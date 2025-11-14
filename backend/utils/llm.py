from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _require_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured. Set it in your environment variables."
        )
    return api_key


@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    api_key = _require_openai_api_key()
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    return OpenAIEmbeddings(model=model, api_key=api_key)


def _build_chat_model(model_name: str, temperature: float) -> ChatOpenAI:
    api_key = _require_openai_api_key()
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)


def get_chat_model() -> ChatOpenAI:
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    return _build_chat_model(model, temperature)


def get_low_cost_chat_model() -> ChatOpenAI:
    model = os.getenv("OPENAI_CORRECTION_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_CORRECTION_TEMPERATURE", "0.0"))
    return _build_chat_model(model, temperature)


def get_summary_model() -> ChatOpenAI:
    model = os.getenv("OPENAI_SUMMARY_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    temperature = float(os.getenv("OPENAI_SUMMARY_TEMPERATURE", "0.0"))
    return _build_chat_model(model, temperature)

