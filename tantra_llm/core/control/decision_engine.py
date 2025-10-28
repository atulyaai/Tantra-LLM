from __future__ import annotations

from typing import Dict, Any


class DecisionEngine:
    """Stub: determines recall depth, mode selection, and storage decisions."""

    def __init__(self, personality_layer, memory_manager):
        self.personality = personality_layer
        self.memory = memory_manager

    def decide(self, user_text: str) -> Dict[str, Any]:
        mode = self.personality.select_mode(user_text)
        recalls = self.memory.recall(user_text)
        return {"mode": mode, "recall": recalls}


