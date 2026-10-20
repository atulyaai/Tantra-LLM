"""Product-key expert layer: thousands of tiny experts, a handful active per token (CPU-friendly).

Each expert is one hidden neuron (a down vector u_i and an up vector v_i). A token picks its
top-k experts out of N = n_keys**2 with product keys (two sub-key searches of n_keys each, then
top-k over k*k candidates), so lookup costs O(sqrt(N)) instead of O(N). Only the selected rows of
the expert tables are read, and with sparse gradients only those rows are updated. Total
parameters grow with N (knowledge capacity) while compute per token stays roughly flat.
Based on product-key memories (Lample et al. 2019) and PEER (He 2024).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ProductKeyExperts(nn.Module):
    def __init__(self, d_model: int, n_keys: int = 128, heads: int = 4, top_k: int = 16,
                 key_dim: int = 64, sparse: bool = True):
        super().__init__()
        self.n_keys, self.heads, self.top_k, self.key_dim = n_keys, heads, top_k, key_dim
        self.n_experts = n_keys * n_keys
        self.query = nn.Linear(d_model, heads * key_dim, bias=False)
        self.keys = nn.Parameter(torch.randn(heads, 2, n_keys, key_dim // 2) / math.sqrt(key_dim // 2))
        # Expert tables as embeddings so backward touches only the selected rows (sparse=True).
        self.down = nn.Embedding(self.n_experts, d_model, sparse=sparse)
        self.up = nn.Embedding(self.n_experts, d_model, sparse=sparse)
        nn.init.normal_(self.down.weight, std=d_model ** -0.5)
        nn.init.zeros_(self.up.weight)  # starts as a no-op, so it can be added to a trained model safely

    def route(self, x: Tensor):
        """Top-k (scores, expert ids) per head for x of shape [T, d]."""
        T = x.shape[0]
        q = self.query(x).view(T, self.heads, 2, self.key_dim // 2)
        s1 = torch.einsum('thd,hnd->thn', q[:, :, 0], self.keys[:, 0])   # [T, H, n_keys]
        s2 = torch.einsum('thd,hnd->thn', q[:, :, 1], self.keys[:, 1])
        v1, i1 = s1.topk(self.top_k, -1)
        v2, i2 = s2.topk(self.top_k, -1)
        cand = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(-2)         # [T, H, k*k]
        ids = (i1.unsqueeze(-1) * self.n_keys + i2.unsqueeze(-2)).flatten(-2)
        scores, pick = cand.topk(self.top_k, -1)
        return scores, ids.gather(-1, pick)                                # [T, H, k]

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        scores, ids = self.route(x)
        gate = scores.softmax(-1)                                          # [T, H, k]
        down = self.down(ids)                                              # [T, H, k, d]
        h = F.gelu(torch.einsum('td,thkd->thk', x, down)) * gate
        out = torch.einsum('thk,thkd->td', h, self.up(ids))
        return out.reshape(shape)

    def active_params_per_token(self, d_model: int) -> int:
        return self.heads * self.top_k * 2 * d_model
