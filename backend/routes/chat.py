from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.utils.rag_pipeline import (
    ChatResult,
    generate_suggestions,
    handle_chat,
)
from backend.utils.memory_service import conversation_memory_service


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., description="User message for the assistant.")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional identifier to persist conversation context across requests.",
    )


class SourceItem(BaseModel):
    source: Optional[str]
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    section: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    corrected_query: Optional[str] = None
    sources: List[SourceItem] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    query_type: Optional[str] = None


class HistoryResponse(BaseModel):
    session_id: str
    history: List[ChatMessage]


class SuggestionResponse(BaseModel):
    suggestions: List[str]


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    session_id = payload.session_id or "default-session"
    try:
        result: ChatResult = await handle_chat(session_id, payload.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ChatResponse(
        response=result.response,
        corrected_query=result.corrected_query,
        sources=[SourceItem(**source) for source in result.sources],
        suggestions=result.suggestions,
        query_type=result.query_type.value if result.query_type else None,
    )


@router.get("/history", response_model=HistoryResponse)
async def history_endpoint(
    session_id: str = Query(..., description="Session identifier to retrieve history for."),
) -> HistoryResponse:
    history = conversation_memory_service.load_plain_history(session_id)
    return HistoryResponse(session_id=session_id, history=history)


@router.get("/suggest", response_model=SuggestionResponse)
async def suggestion_endpoint(
    session_id: Optional[str] = Query(None, description="Optional session id to tailor suggestions."),
    last_query: Optional[str] = Query(None, description="Optional explicit query to base suggestions on."),
) -> SuggestionResponse:
    suggestions = await generate_suggestions(session_id=session_id, query=last_query)
    return SuggestionResponse(suggestions=suggestions)

