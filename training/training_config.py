from __future__ import annotations

"""Training config for fusion projection layers only."""

from dataclasses import dataclass


@dataclass
class FusionTrainingConfig:
    lr: float = 1e-4
    batch_size: int = 8
    epochs: int = 3
    weight_decay: float = 0.01
    grad_accum: int = 1
    grad_clip: float = 1.0
    warmup_steps: int = 10
    dropout: float = 0.1
