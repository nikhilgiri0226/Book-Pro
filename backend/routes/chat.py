from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.utils.memory_manager import memory_manager
from backend.utils.rag_pipeline import (
    ChatResult,
    generate_suggestions,
    handle_chat,
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user/session.")
    message: str = Field(..., description="User message for the assistant.")


class ChatResponse(BaseModel):
    answer: str
    corrected_query: Optional[str] = None
    sources: List[dict] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    user_id: str
    history: List[ChatMessage]


class SuggestionResponse(BaseModel):
    suggestions: List[str]


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    try:
        result: ChatResult = await handle_chat(payload.user_id, payload.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ChatResponse(
        answer=result.answer,
        corrected_query=result.corrected_query,
        sources=result.sources,
        suggestions=result.suggestions,
    )


@router.get("/history", response_model=HistoryResponse)
async def history_endpoint(user_id: str = Query(..., description="User/session id")) -> HistoryResponse:
    history = memory_manager.get_history(user_id)
    return HistoryResponse(user_id=user_id, history=history)


@router.get("/suggest", response_model=SuggestionResponse)
async def suggestion_endpoint(
    user_id: Optional[str] = Query(None, description="Optional user/session id to tailor suggestions."),
    last_query: Optional[str] = Query(None, description="Optional explicit query to base suggestions on."),
) -> SuggestionResponse:
    suggestions = await generate_suggestions(user_id=user_id, query=last_query)
    return SuggestionResponse(suggestions=suggestions)

