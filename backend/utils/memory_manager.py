from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Dict, List


class MemoryManager:
    def __init__(self, history_path: Path, window_size: int = 10) -> None:
        self.history_path = history_path
        self.window_size = window_size
        self._history: Dict[str, List[Dict[str, str]]] = {}
        self._lock = Lock()

    def load_history(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_path.exists():
            self.history_path.write_text("{}", encoding="utf-8")

        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._history = {
                    user_id: list(messages)
                    for user_id, messages in data.items()
                    if isinstance(messages, list)
                }
            else:  # pragma: no cover - defensive
                self._history = {}
        except json.JSONDecodeError:  # pragma: no cover - defensive
            self._history = {}

    def get_history(self, user_id: str) -> List[Dict[str, str]]:
        return self._history.get(user_id, [])

    def get_context(self, user_id: str) -> List[Dict[str, str]]:
        history = self.get_history(user_id)
        return history[-self.window_size :]

    def append_message(self, user_id: str, role: str, content: str) -> None:
        with self._lock:
            messages = self._history.setdefault(user_id, [])
            messages.append({"role": role, "content": content})
            self._history[user_id] = messages[-self.window_size :]
            self._persist()

    def overwrite_history(self, user_id: str, messages: List[Dict[str, str]]) -> None:
        with self._lock:
            self._history[user_id] = messages[-self.window_size :]
            self._persist()

    def _persist(self) -> None:
        serialized = json.dumps(self._history, ensure_ascii=False, indent=2)
        self.history_path.write_text(serialized, encoding="utf-8")


history_file = Path(__file__).resolve().parent.parent / "db" / "history.json"
memory_manager = MemoryManager(history_file)

