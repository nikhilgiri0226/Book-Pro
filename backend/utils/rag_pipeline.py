from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from backend.utils.memory_manager import memory_manager
from backend.utils.ocr_utils import extract_pdf_content, normalize_text

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "book-movie-chat")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")

vector_store: Optional[PineconeVectorStore] = None
fallback_store: Optional[Chroma] = None


@dataclass
class ChatResult:
    answer: str
    sources: List[Dict[str, object]]
    corrected_query: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)


def _require_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured. Set it in your environment variables."
        )
    return api_key


def get_embedding_model() -> OpenAIEmbeddings:
    api_key = _require_openai_api_key()
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)


def get_chat_model() -> ChatOpenAI:
    api_key = _require_openai_api_key()
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    return ChatOpenAI(model=CHAT_MODEL, temperature=temperature, api_key=api_key)


def init_vector_store() -> None:
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
                    dimension=3072,
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
            "source": file.filename,
            "page": page.get("page_number"),
            "images": page.get("images", []),
        }
        documents.append(Document(page_content=text, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Failed to create chunks from the supplied PDF.")

    store = _get_active_vector_store()
    await _add_documents(store, chunks)

    return {
        "filename": file.filename,
        "chunks_indexed": len(chunks),
        "metadata": {
            "pages_processed": len(pages),
            "vector_store": "pinecone"
            if vector_store is not None
            else "chroma",
        },
    }


def _get_active_vector_store():
    if vector_store is None and fallback_store is None:
        raise ValueError(
            "Vector store is not initialised. Ensure init_vector_store() ran successfully."
        )
    return vector_store or fallback_store


async def _add_documents(store, documents: Sequence[Document]) -> None:
    def _run():
        store.add_documents(list(documents))

    await run_in_threadpool(_run)


async def handle_chat(user_id: str, message: str) -> ChatResult:
    if not message.strip():
        raise ValueError("Message cannot be empty.")

    corrected_query = await _correct_query(message)
    query = corrected_query or message

    retrieved_docs = await _retrieve_documents(query)
    history = memory_manager.get_context(user_id)

    response_text = await _generate_response(query, history, retrieved_docs)

    memory_manager.append_message(user_id, "user", message)
    memory_manager.append_message(user_id, "assistant", response_text)

    suggestions = await generate_suggestions(
        user_id=user_id, query=query, base_docs=retrieved_docs
    )

    sources_payload = [
        {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "snippet": doc.page_content[:200],
        }
        for doc in retrieved_docs
    ]

    return ChatResult(
        answer=response_text,
        corrected_query=query if query != message else None,
        sources=sources_payload,
        suggestions=suggestions,
    )


async def _correct_query(message: str) -> Optional[str]:
    try:
        llm = get_chat_model()
    except ValueError:
        return None

    system_prompt = (
        "You act as a spell checker and typo corrector for user search queries. "
        "Return only the corrected query without additional text. "
        "If the query is already correct, return it unchanged."
    )
    result = await llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=message)]
    )
    corrected = result.content.strip()
    if corrected.lower() == message.strip().lower():
        return None
    return corrected


async def _retrieve_documents(query: str, top_k: int = 5) -> List[Document]:
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


async def _generate_response(
    query: str,
    history: Sequence[Dict[str, str]],
    docs: Sequence[Document],
) -> str:
    llm = get_chat_model()
    context_snippets = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', 'N/A')})\n"
        f"{doc.page_content}"
        for doc in docs
    ) or "No relevant context was retrieved. Answer based on general knowledge."

    history_text = "\n".join(
        f"{item['role'].capitalize()}: {item['content']}" for item in history
    )

    system_prompt = (
        "You are BookMovieChat, a helpful assistant that chats with users about books "
        "and movies using uploaded PDFs as references. "
        "Provide accurate answers, cite key details, and highlight if information is "
        "inferred or uncertain. Keep responses concise but welcoming."
    )

    user_prompt = (
        f"Conversation History:\n{history_text or 'No prior context.'}\n\n"
        f"Retrieved Context:\n{context_snippets}\n\n"
        f"User Query:\n{query}\n\n"
        "Compose your answer. Reference the context when relevant and mention the page "
        "numbers. If you lack enough information, say so and suggest uploading "
        "additional material."
    )

    response = await llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    return response.content.strip()


async def generate_suggestions(
    user_id: Optional[str] = None,
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
        llm = get_chat_model()
    except ValueError:
        return default_suggestions

    context = []
    if query:
        context.append(f"User query: {query}")
    if user_id:
        history = memory_manager.get_context(user_id)
        if history:
            context.append(
                "Recent conversation:\n"
                + "\n".join(f"{msg['role']}: {msg['content']}" for msg in history[-4:])
            )
    if base_docs:
        doc_titles = {doc.metadata.get("source", "unknown") for doc in base_docs}
        context.append("Relevant sources: " + ", ".join(sorted(doc_titles)))

    prompt = (
        "Based on the provided context from a conversation about books and movies, "
        "suggest three concise follow-up questions or prompts the user might ask next. "
        "Return them as a plain list separated by newline characters. "
        "If the context is empty, provide three generic but engaging suggestions."
    )

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="\n\n".join(context) if context else "No additional context."),
    ]

    response = await llm.ainvoke(messages)
    suggestions = [line.strip("- ").strip() for line in response.content.splitlines()]
    cleaned = [s for s in suggestions if s]

    return cleaned or default_suggestions

