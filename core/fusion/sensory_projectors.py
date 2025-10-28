from __future__ import annotations

import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    """Vd → D projection with basic shape handling and validation."""

    def __init__(self, vision_dim: int, model_dim: int):
        super().__init__()
        self.vision_dim = vision_dim
        self.model_dim = model_dim
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B, Vd] or [Vd]; reshape to [B, Vd]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.size(-1) == self.vision_dim, f"Expected last dim {self.vision_dim}, got {x.size(-1)}"
        y = self.proj(x)
        assert y.size(-1) == self.model_dim, f"Projected dim mismatch: {y.size(-1)} != {self.model_dim}"
        return y


class AudioProjector(nn.Module):
    """Wd → D projection with basic shape handling and validation."""

    def __init__(self, audio_dim: int, model_dim: int):
        super().__init__()
        self.audio_dim = audio_dim
        self.model_dim = model_dim
        self.proj = nn.Sequential(
            nn.Linear(audio_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B, Wd] or [Wd]; reshape to [B, Wd]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.size(-1) == self.audio_dim, f"Expected last dim {self.audio_dim}, got {x.size(-1)}"
        y = self.proj(x)
        assert y.size(-1) == self.model_dim, f"Projected dim mismatch: {y.size(-1)} != {self.model_dim}"
        return y


