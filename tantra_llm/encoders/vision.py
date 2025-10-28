from __future__ import annotations

from typing import Any

import torch


class VisionEncoder:
    """Stub wrapper for Long-ViT-like encoder; returns embeddings tensor."""

    def __init__(self, embed_dim: int = 1024):
        self.embed_dim = embed_dim

    def __call__(self, image: Any) -> torch.Tensor:
        return torch.zeros(1, self.embed_dim)


