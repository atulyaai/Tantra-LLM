from __future__ import annotations

from typing import List, Dict, Any


class WorkingMemory:
    """Stub: context window manager with importance-based trimming."""

    def __init__(self, max_tokens: int = 8192):
        self.max_tokens = max_tokens
        self.items: List[Dict[str, Any]] = []

    def add(self, content: str, importance: float = 0.5):
        self.items.append({"content": content, "importance": float(importance)})
        self._trim()

    def get_context(self) -> str:
        return "\n".join(i["content"] for i in self.items)

    def _trim(self):
        # TODO: Replace length heuristic with tokenizer-aware trimming
        while sum(len(i["content"]) for i in self.items) > self.max_tokens and self.items:
            self.items.sort(key=lambda x: x["importance"])  # ascending
            self.items.pop(0)


