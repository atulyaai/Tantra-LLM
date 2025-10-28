from __future__ import annotations

"""Training config for fusion projection layers only.

# DESIGN QUESTION:
- Confirm batch size, lr, schedule, and freeze policies for base models.
"""

from dataclasses import dataclass


@dataclass
class FusionTrainingConfig:
    lr: float = 1e-4
    batch_size: int = 8
    epochs: int = 3
    weight_decay: float = 0.01
    grad_accum: int = 1


