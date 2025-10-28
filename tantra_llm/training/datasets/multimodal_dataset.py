from __future__ import annotations

"""Dataset stub for image+caption and audio+transcription pairs.

# DESIGN QUESTION:
- Confirm data formats, max lengths, and alignment strategy.
"""

from typing import Any, Dict, List


class MultimodalDataset:
    """Stub dataset returning dicts with image/audio/text entries."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


