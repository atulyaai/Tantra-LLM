from __future__ import annotations

"""Trainer stub for fusion projection layers (freeze base models).

# DESIGN QUESTION:
- Confirm losses: token cross-entropy + alignment (e.g., cosine) weights.
"""

from typing import Any
from .training_config import FusionTrainingConfig


class FusionTrainer:
    """Stub trainer coordinating dataset, projectors, and optimization."""

    def __init__(self, config: FusionTrainingConfig, vision_projector: Any, audio_projector: Any):
        self.config = config
        self.vision_projector = vision_projector
        self.audio_projector = audio_projector

    def fit(self, dataset):
        pass


