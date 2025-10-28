from __future__ import annotations

from typing import List

from .working_memory import WorkingMemory
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory


class MemoryManager:
    """Stub: coordinates working, episodic, and semantic memory."""

    def __init__(self, wm_tokens: int = 8192):
        self.working = WorkingMemory(max_tokens=wm_tokens)
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()

    def recall(self, query: str) -> List[str]:
        facts = ["; ".join(f) for f in self.semantic.query(query)]
        episodes = self.episodic.retrieve(query)
        return facts + episodes

    def consider_store(self, text: str, importance: float = 0.5):
        if importance >= 0.7:
            self.episodic.store(summary=text, importance=importance)
        else:
            self.working.add(text, importance=importance)


