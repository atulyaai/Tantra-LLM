from __future__ import annotations

"""
Batch-free token streaming interface for low-latency decoding.

# DESIGN QUESTION:
- Define streaming API for SpikingBrain generate step and partial results.
"""


class TokenStreamer:
    """Stub for streaming decode callbacks."""

    def __init__(self):
        pass

    def on_token(self, token_id: int):
        """TODO: Emit tokens to client incrementally."""
        pass


