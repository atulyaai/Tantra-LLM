from __future__ import annotations

"""
Dynamic compute routing: fast/medium/deep paths based on query complexity.

# DESIGN QUESTION:
- Define complexity scoring and path thresholds.
"""

from typing import Dict


class ComputeRouter:
    """Stub selecting compute path according to complexity score."""

    def __init__(self):
        pass

    def route(self, meta: Dict) -> str:
        return "medium"


