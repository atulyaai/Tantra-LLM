from __future__ import annotations

"""Safety Module: deny-list + toxicity checks."""

from typing import Dict


class SafetyModule:
    """Evaluate safety and return action (pass/modify/deny)."""

    def __init__(self):
        self.deny_keywords = [
            "illegal", "harmful", "violent", "explosive", "weapon",
            "drug", "suicide", "hack", "virus", "malware"
        ]
        self.toxicity_patterns = [
            "hate", "discrimination", "racism", "sexism", "bigotry"
        ]

    def evaluate(self, draft: str, context: Dict) -> Dict:
        """Return action (pass/modify/deny) + reasons."""
        draft_lower = draft.lower()
        
        # Check deny list
        for kw in self.deny_keywords:
            if kw in draft_lower:
                return {"action": "deny", "reasons": [f"Contains dangerous keyword: {kw}"]}
        
        # Check toxicity
        for pattern in self.toxicity_patterns:
            if pattern in draft_lower:
                return {"action": "modify", "reasons": [f"Contains toxic content: {pattern}"]}
        
        # Pass if safe
        return {"action": "pass", "reasons": []}


