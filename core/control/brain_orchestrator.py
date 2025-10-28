from __future__ import annotations

from typing import Optional


class BrainOrchestrator:
    """Stub: perception → decision → response → reflection → memory."""

    def __init__(self, perception, decision_engine, response_generator, memory_manager):
        self.perception = perception
        self.decision = decision_engine
        self.response = response_generator
        self.memory = memory_manager

    def step(self, text: Optional[str] = None, image=None, audio=None) -> str:
        p = self.perception.perceive(text=text, image=image, audio=audio)
        d = self.decision.decide(text or "")
        out = self.response.generate(p, d)
        self.memory.consider_store(text or "", importance=0.6)
        return out


