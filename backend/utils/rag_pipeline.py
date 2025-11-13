from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from backend.utils.llm import (
    get_chat_model,
    get_embedding_model,
    get_low_cost_chat_model,
    get_summary_model,
)
from backend.utils.memory_service import conversation_memory_service
from backend.utils.ocr_utils import extract_pdf_content, normalize_text

logger = logging.getLogger(__name__)

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "book-movie-chat")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
EMBEDDING_DIMENSION = int(os.getenv("OPENAI_EMBEDDING_DIM", "3072"))

vector_store: Optional[PineconeVectorStore] = None
fallback_store: Optional[Chroma] = None


class QueryType(str, Enum):
    BROAD = "broad"
    SPECIFIC = "specific"


@dataclass
class ChatResult:
    response: str
    sources: List[Dict[str, object]]
    corrected_query: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    query_type: QueryType = QueryType.SPECIFIC


def init_vector_store() -> None:
    """Initialise Pinecone or fall back to a local Chroma store."""
    global vector_store, fallback_store

    try:
        embedding_model = get_embedding_model()
    except ValueError as exc:
        logger.warning("Embedding model could not be initialised: %s", exc)
        vector_store = None
        fallback_store = None
        return

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV")

    if pinecone_api_key and pinecone_env:
        try:
            pc = Pinecone(api_key=pinecone_api_key)
            existing_indexes = pc.list_indexes().names()
            if PINECONE_INDEX_NAME not in existing_indexes:
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=pinecone_env),
                )
            index = pc.Index(PINECONE_INDEX_NAME)
            vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
            fallback_store = None
            logger.info("Connected to Pinecone index '%s'.", PINECONE_INDEX_NAME)
            return
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Falling back to Chroma due to Pinecone error: %s", exc)

    persist_dir = Path(__file__).resolve().parent.parent / "db" / "chroma_store"
    persist_dir.mkdir(parents=True, exist_ok=True)
    fallback_store = Chroma(
        collection_name=PINECONE_INDEX_NAME,
        embedding_function=embedding_model,
        persist_directory=str(persist_dir),
    )
    vector_store = None
    logger.info("Using local Chroma vector store at %s", persist_dir)


async def ingest_document(file: UploadFile) -> Dict[str, object]:
    if not file.filename:
        raise ValueError("Uploaded file must have a filename.")

    file_bytes = await file.read()
    return await ingest_pdf_bytes(file.filename, file_bytes)


async def ingest_pdf_bytes(filename: str, file_bytes: bytes) -> Dict[str, object]:
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    pages = extract_pdf_content(file_bytes)
    if not pages:
        raise ValueError("No readable content found in the PDF.")

    documents: List[Document] = []
    for page in pages:
        text = normalize_text(str(page.get("text", "")))
        if not text and page.get("images"):
            text = (
                "Image-heavy page detected. Refer to the original PDF for visual details."
            )
        if not text:
            continue

        metadata = {
            "source": filename,
            "page": page.get("page_number"),
            "images": page.get("images", []),
        }
        documents.append(Document(page_content=text, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Failed to create chunks from the supplied PDF.")

    _enrich_chunks_with_neighbors(chunks, filename)

    store = _get_active_vector_store()
    await _add_documents(store, chunks)

    return {
        "filename": filename,
        "chunks_indexed": len(chunks),
        "metadata": {
            "pages_processed": len(pages),
            "vector_store": "pinecone" if vector_store is not None else "chroma",
        },
    }


def _enrich_chunks_with_neighbors(chunks: Sequence[Document], filename: str) -> None:
    total = len(chunks)
    base_id = filename.replace(" ", "_").lower()

    for index, chunk in enumerate(chunks):
        metadata = chunk.metadata
        metadata["chunk_index"] = index
        metadata["chunk_total"] = total
        metadata["chunk_id"] = f"{base_id}::chunk-{index}"
        metadata.setdefault("section", f"page-{metadata.get('page', 'unknown')}")

    for index, chunk in enumerate(chunks):
        metadata = chunk.metadata
        prev_chunk = chunks[index - 1] if index > 0 else None
        next_chunk = chunks[index + 1] if index < total - 1 else None

        if prev_chunk:
            metadata["prev_chunk_id"] = prev_chunk.metadata.get("chunk_id")
            metadata["prev_chunk_page"] = prev_chunk.metadata.get("page")
            metadata["prev_chunk_text"] = prev_chunk.page_content
        else:
            metadata["prev_chunk_id"] = None
            metadata["prev_chunk_page"] = None
            metadata["prev_chunk_text"] = ""

        if next_chunk:
            metadata["next_chunk_id"] = next_chunk.metadata.get("chunk_id")
            metadata["next_chunk_page"] = next_chunk.metadata.get("page")
            metadata["next_chunk_text"] = next_chunk.page_content
        else:
            metadata["next_chunk_id"] = None
            metadata["next_chunk_page"] = None
            metadata["next_chunk_text"] = ""


def _get_active_vector_store():
    if vector_store is None and fallback_store is None:
        raise ValueError(
            "Vector store is not initialised. Ensure init_vector_store() ran successfully."
        )
    return vector_store or fallback_store


async def _add_documents(store, documents: Sequence[Document]) -> None:
    ids = [doc.metadata.get("chunk_id") for doc in documents]

    def _run():
        store.add_documents(list(documents), ids=ids)

    await run_in_threadpool(_run)


async def handle_chat(session_id: str, message: str) -> ChatResult:
    if not message.strip():
        raise ValueError("Message cannot be empty.")

    normalized_query = await _normalize_query(message)
    corrected_query = (
        normalized_query if normalized_query.lower() != message.strip().lower() else None
    )

    query_type = await _classify_query(normalized_query)
    top_k = 25 if query_type is QueryType.BROAD else 5

    retrieved_docs = await _retrieve_documents(normalized_query, top_k=top_k)
    expanded_docs = _expand_with_neighbors(retrieved_docs)

    memory = conversation_memory_service.get_memory(session_id)
    history_messages = memory.load_memory_variables({}).get("history", [])
    history_text = _render_history(history_messages)

    response_text = await _map_reduce_answer(
        normalized_query, expanded_docs, history_text, query_type
    )

    conversation_memory_service.save(session_id, normalized_query, response_text)

    suggestions = await generate_suggestions(
        session_id=session_id, query=normalized_query, base_docs=retrieved_docs
    )

    sources_payload = [
        {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "section": doc.metadata.get("section"),
        }
        for doc in retrieved_docs
    ]

    return ChatResult(
        response=response_text,
        corrected_query=corrected_query,
        sources=sources_payload,
        suggestions=suggestions,
        query_type=query_type,
    )


async def _normalize_query(message: str) -> str:
    try:
        llm = get_low_cost_chat_model()
    except ValueError:
        return message.strip()

    system_prompt = (
        "Rewrite the user query so it is well-formed, correctly capitalised, and free of typos. "
        "If the query is already correct, return it unchanged. Respond with only the corrected query."
    )
    result = await llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=message)]
    )
    corrected = result.content.strip()
    return corrected or message.strip()


async def _classify_query(query: str) -> QueryType:
    lowered = query.lower()
    broad_keywords = {"all", "every", "entire", "overview", "summary", "summarize", "list"}
    specific_keywords = {"who", "when", "where", "why", "how", "did", "was", "is"}

    if any(word in lowered for word in broad_keywords):
        heuristic = QueryType.BROAD
    elif any(lowered.startswith(word) for word in specific_keywords):
        heuristic = QueryType.SPECIFIC
    elif len(lowered.split()) <= 6:
        heuristic = QueryType.SPECIFIC
    elif len(lowered.split()) >= 14:
        heuristic = QueryType.BROAD
    else:
        heuristic = QueryType.SPECIFIC

    try:
        llm = get_low_cost_chat_model()
    except ValueError:
        return heuristic

    classification_prompt = (
        "Classify the query as either 'broad' or 'specific'.\n"
        "- Broad: requests comprehensive summaries, timelines, or lists of many items.\n"
        "- Specific: targeted facts, single events, or individual characters.\n"
        "Respond with only the word broad or specific."
    )

    result = await llm.ainvoke(
        [SystemMessage(content=classification_prompt), HumanMessage(content=query)]
    )
    cleaned = result.content.strip().lower()
    if cleaned not in {"broad", "specific"}:
        return heuristic
    return QueryType(cleaned)


async def _retrieve_documents(query: str, top_k: int) -> List[Document]:
    store = _get_active_vector_store()
    retriever = (
        store.as_retriever(search_kwargs={"k": top_k})
        if hasattr(store, "as_retriever")
        else store
    )

    def _run():
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        return retriever.similarity_search(query, k=top_k)

    docs = await run_in_threadpool(_run)
    return docs or []


def _expand_with_neighbors(docs: Sequence[Document]) -> List[Document]:
    expanded: List[Document] = []
    seen_ids = set()

    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id not in seen_ids:
            expanded.append(doc)
            seen_ids.add(chunk_id)

        for prefix in ("prev", "next"):
            neighbor_text = doc.metadata.get(f"{prefix}_chunk_text")
            neighbor_id = doc.metadata.get(f"{prefix}_chunk_id")
            neighbor_page = doc.metadata.get(f"{prefix}_chunk_page")
            if neighbor_text and neighbor_id not in seen_ids:
                neighbor_metadata = {
                    "source": doc.metadata.get("source"),
                    "page": neighbor_page or doc.metadata.get("page"),
                    "section": doc.metadata.get("section"),
                    "chunk_id": neighbor_id,
                    "relation": f"{prefix}_neighbor",
                    "origin_chunk_id": chunk_id,
                }
                expanded.append(Document(page_content=neighbor_text, metadata=neighbor_metadata))
                seen_ids.add(neighbor_id)

    return expanded


def _render_history(messages: Sequence[AIMessage | HumanMessage | SystemMessage]) -> str:
    if not messages:
        return ""
    lines: List[str] = []
    for msg in messages:
        if msg.type == "system":
            continue
        role = msg.type.capitalize()
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


async def _map_reduce_answer(
    query: str,
    documents: Sequence[Document],
    history: str,
    query_type: QueryType,
) -> str:
    if not documents:
        return await _answer_without_context(query, history)

    llm = get_chat_model()
    max_docs = 12 if query_type is QueryType.BROAD else 8
    selected_docs = list(documents[:max_docs])

    async def _map_document(doc: Document) -> str:
        map_prompt = (
            "User Query:\n{query}\n\n"
            "Conversation Summary:\n{history}\n\n"
            "Document Section (include neighbour context if provided):\n{context}\n\n"
            "Extract only the facts that answer the user query. "
            "Quote key phrases and note page numbers if present. "
            "If nothing relevant is found, respond with 'NO_MATCH'."
        )
        context = doc.page_content
        metadata = doc.metadata
        if metadata.get("relation"):
            context = (
                f"(Neighbour chunk from {metadata.get('relation')} of {metadata.get('origin_chunk_id')})\n"
                f"{context}"
            )
        message = map_prompt.format(query=query, history=history or "None", context=context)
        response = await llm.ainvoke([HumanMessage(content=message)])
        return response.content.strip()

    map_results = await asyncio.gather(*(_map_document(doc) for doc in selected_docs))
    filtered_results = [
        result for result in map_results if result and "NO_MATCH" not in result.upper()
    ]

    if not filtered_results:
        return await _answer_without_context(query, history)

    reduce_prompt = (
        "You are BookMovieChat. Merge the findings below into a coherent answer for the user.\n"
        "Ensure the answer is concise, cites page numbers when available, and notes uncertainties.\n"
        "If conflicting details appear, clarify them.\n\n"
        "User Query:\n{query}\n\n"
        "Conversation Summary:\n{history}\n\n"
        "Findings:\n{findings}\n\n"
        "Final Answer:"
    )
    reduce_message = reduce_prompt.format(
        query=query,
        history=history or "None",
        findings="\n\n".join(filtered_results),
    )
    reduce_response = await llm.ainvoke([HumanMessage(content=reduce_message)])
    return reduce_response.content.strip()


async def _answer_without_context(query: str, history: str) -> str:
    llm = get_summary_model()
    prompt = (
        "You are BookMovieChat. The vector database returned no relevant passages. "
        "Use general domain knowledge to answer cautiously, note missing context, and "
        "encourage the user to upload more material if needed.\n\n"
        f"Conversation Summary:\n{history or 'None'}\n\n"
        f"User Query:\n{query}\n"
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip()


async def generate_suggestions(
    session_id: Optional[str] = None,
    query: Optional[str] = None,
    base_docs: Optional[Sequence[Document]] = None,
) -> List[str]:
    default_suggestions = [
        "Summarize the current chapter",
        "List the main themes in this book",
        "Explain the character motivation in this scene",
        "Compare the book and movie adaptations",
        "What happens next in the story?",
    ]

    try:
        llm = get_low_cost_chat_model()
    except ValueError:
        return default_suggestions

    context_parts = []
    if query:
        context_parts.append(f"Most recent user query:\n{query}")
    if session_id:
        history = conversation_memory_service.get_memory(session_id).load_memory_variables({}).get(
            "history", []
        )
        history_text = _render_history(history)
        if history_text:
            context_parts.append("Conversation snapshot:\n" + history_text[-1500:])
    if base_docs:
        titles = {doc.metadata.get("source", "unknown") for doc in base_docs}
        context_parts.append("Relevant sources: " + ", ".join(sorted(titles)))

    prompt = (
        "You assist readers exploring books and movies. Suggest three short, engaging follow-up "
        "questions the user might ask next based on the context. "
        "Return each suggestion on its own line without numbering."
    )

    response = await llm.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content="\n\n".join(context_parts) if context_parts else "No context."),
        ]
    )
    suggestions = [line.strip("- ").strip() for line in response.content.splitlines()]
    cleaned = [s for s in suggestions if s]
    return cleaned or default_suggestions


async def reindex_local_pdfs(root_dir: Path) -> Dict[str, object]:
    directory = root_dir if isinstance(root_dir, Path) else Path(root_dir)
    if not directory.exists():
        raise ValueError(f"Directory '{directory}' does not exist.")

    pdf_files = sorted(directory.glob("**/*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in '{directory}'.")

    results: List[Dict[str, object]] = []
    total_chunks = 0

    for pdf_file in pdf_files:
        file_bytes = pdf_file.read_bytes()
        ingestion = await ingest_pdf_bytes(pdf_file.name, file_bytes)
        results.append(
            {
                "filename": pdf_file.name,
                "chunks_indexed": ingestion["chunks_indexed"],
                "metadata": ingestion["metadata"],
            }
        )
        total_chunks += ingestion["chunks_indexed"]

    return {
        "directory": str(directory),
        "total_files": len(results),
        "total_chunks": total_chunks,
        "files": results,
    }

