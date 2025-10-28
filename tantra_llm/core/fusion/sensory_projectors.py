from __future__ import annotations

import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    """Stub: Vd → D projection (confirm dims)."""

    def __init__(self, vision_dim: int, model_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AudioProjector(nn.Module):
    """Stub: Wd → D projection (confirm dims)."""

    def __init__(self, audio_dim: int, model_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(audio_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


