import math
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        # depthwise causal conv (approx selective scan context mixing)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=d_conv, groups=d_model, padding=d_conv - 1)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        h = self.norm(x)
        u, v = self.in_proj(h).chunk(2, dim=-1)  # GLU-like gating
        # depthwise conv across time
        y = self.dwconv(u.transpose(1, 2))[:, :, : u.size(1)]  # [B, C, T] -> causal trim
        y = y.transpose(1, 2)
        y = y * torch.sigmoid(v)
        y = self.out_proj(y)
        y = self.dropout(y)
        return x + y  # residual


class MambaDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, d_state: int, d_conv: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, dropout=dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.out(x)  # [B, T, V]
        return logits


def build_from_config(cfg: Dict[str, Any], vocab_size: int) -> MambaDecoder:
    d_model = int(cfg.get("d_model", 256))
    n_layers = int(cfg.get("n_layers", 6))
    d_state = int(cfg.get("d_state", 64))
    d_conv = int(cfg.get("d_conv", 4))
    dropout = float(cfg.get("dropout", 0.0))
    return MambaDecoder(vocab_size, d_model, n_layers, d_state, d_conv, dropout)


