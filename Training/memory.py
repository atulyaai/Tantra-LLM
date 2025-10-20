from typing import Any, Dict, List


class Memory:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.short_history: List[Dict[str, Any]] = []

    def build_context(self, user_text: str) -> str:
        max_turns = int(self.config.get("memory", {}).get("short", {}).get("max_turns", 10))
        recent = self.short_history[-max_turns:]
        parts: List[str] = []
        for turn in recent:
            parts.append(f"User: {turn['user']}")
            parts.append(f"Assistant: {turn['assistant']}")
        return "\n".join(parts)

    def observe_turn(self, user_text: str, assistant_text: str, traces: List[Dict[str, Any]]):
        self.short_history.append({"user": user_text, "assistant": assistant_text})


