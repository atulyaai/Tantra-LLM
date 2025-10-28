from __future__ import annotations

import torch.nn as nn


class AdapterLayer(nn.Module):
    """Stub bottleneck adapter (personality/style injection in fusion)."""

    def __init__(self, model_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(model_dim, bottleneck_dim, bias=False)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, model_dim, bias=False)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


