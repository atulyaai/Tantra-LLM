from __future__ import annotations

"""
Parameterizer: converts personality outputs into prompt prefixes, control tokens,
and decoding parameters (temperature, top_p, max_tokens, penalties).

# DESIGN QUESTION:
- Confirm decoding ranges per mode; specify persistence rules for overrides.
"""

from typing import Dict


class Parameterizer:
    """Stub interface for mapping persona decisions to decoding params and prompts."""

    def __init__(self, config: Dict):
        self.config = config

    def build(self, mode: str, style_hints: Dict) -> Dict:
        return {"prompt_prefix": "", "temperature": 0.7, "top_p": 0.9, "max_tokens": 512}


