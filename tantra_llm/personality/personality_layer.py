from __future__ import annotations

from typing import Dict


AUTO_CUES = {
    "DirectAssertive": ["just give me", "final answer", "short"],
    "MentorBuilder": ["how do i", "i'm unsure", "can you guide"],
    "CriticalChallenger": ["are you sure", "prove", "seems wrong"],
    "CreativeExplorer": ["ideas", "alternatives", "brainstorm", "creative"],
}

OVERRIDES = {
    "mode: direct": "DirectAssertive",
    "mode: mentor": "MentorBuilder",
    "mode: critical": "CriticalChallenger",
    "mode: creative": "CreativeExplorer",
}


class PersonalityLayer:
    """Stub: selects mode (auto + overrides) and parameterizes decoding/prefixes."""

    def __init__(self, config: Dict):
        self.config = config
        self.default_mode = "DirectAssertive"
        self.session_mode: str | None = None

    def select_mode(self, user_text: str) -> str:
        lt = user_text.lower()
        for k, v in OVERRIDES.items():
            if k in lt:
                self.session_mode = v
                return v
        if self.session_mode:
            return self.session_mode
        for mode, cues in AUTO_CUES.items():
            if any(c in lt for c in cues):
                return mode
        return self.default_mode

    def parameterize(self, mode: str) -> Dict:
        tones = self.config.get("tones", {})
        params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "prompt_prefix": tones.get("default", {}).get("prompt_prefix", ""),
        }
        mapping = {
            "DirectAssertive": "concise",
            "MentorBuilder": "mentor",
            "CriticalChallenger": "critical",
            "CreativeExplorer": "creative",
        }
        tone = mapping.get(mode, "default")
        preset = tones.get(tone, tones.get("default", {}))
        params["prompt_prefix"] = preset.get("prompt_prefix", params["prompt_prefix"])
        return params


