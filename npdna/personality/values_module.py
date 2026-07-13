from __future__ import annotations

"""
Values Module: outputs normalized value vector guiding style/safety.

# DESIGN QUESTION:
- Confirm parameter budget split among Values/Style/Safety (<50M total).
"""

from typing import Dict


class ValuesModule:
    """Stub interface for computing value scores from context."""

    def __init__(self):
        pass

    def forward(self, context: Dict) -> Dict[str, float]:
        return {}


