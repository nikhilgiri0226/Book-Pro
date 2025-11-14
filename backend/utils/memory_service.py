from __future__ import annotations

import os
from threading import Lock
from typing import Dict, List

from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.utils.llm import get_summary_model
from backend.utils.memory_manager import memory_manager

DEFAULT_SYSTEM_PROMPT = (
    "You are BookMovieChat's memory. Summarize conversations while preserving key facts, "
    "character names, timelines, and unanswered questions. Keep summaries concise and factual."
)


class ConversationMemoryService:
    def __init__(self) -> None:
        self._memories: Dict[str, ConversationSummaryBufferMemory] = {}
        self._lock = Lock()

    def get_memory(self, session_id: str) -> ConversationSummaryBufferMemory:
        with self._lock:
            if session_id not in self._memories:
                self._memories[session_id] = self._create_memory(session_id)
            return self._memories[session_id]

    def _create_memory(self, session_id: str) -> ConversationSummaryBufferMemory:
        max_tokens = int(os.getenv("MEMORY_TOKEN_LIMIT", "1500"))
        memory = ConversationSummaryBufferMemory(
            llm=get_summary_model(),
            max_token_limit=max_tokens,
            return_messages=True,
            memory_key="history",
            input_key="input",
            output_key="output",
            ai_prefix="Assistant",
            human_prefix="User",
        )

        conversation = memory_manager.get_history(session_id)
        if conversation:
            for message in conversation:
                role = message.get("role")
                content = message.get("content", "")
                if not content:
                    continue
                if role == "user":
                    memory.chat_memory.add_message(HumanMessage(content=content))
                elif role == "assistant":
                    memory.chat_memory.add_message(AIMessage(content=content))
                elif role == "system":
                    memory.chat_memory.add_message(SystemMessage(content=content))

        if not any(isinstance(msg, SystemMessage) for msg in memory.chat_memory.messages):
            memory.chat_memory.add_message(SystemMessage(content=DEFAULT_SYSTEM_PROMPT))

        return memory

    def save(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        memory = self.get_memory(session_id)
        memory.save_context({"input": user_message}, {"output": assistant_message})
        memory_manager.append_message(session_id, "user", user_message)
        memory_manager.append_message(session_id, "assistant", assistant_message)

    def load_plain_history(self, session_id: str) -> List[Dict[str, str]]:
        return memory_manager.get_history(session_id)


conversation_memory_service = ConversationMemoryService()

