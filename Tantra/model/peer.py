"""Product-key expert layer: thousands of tiny experts, a handful active per token (CPU-friendly).

Each expert is one hidden neuron (a down vector u_i and an up vector v_i). A token picks its
top-k experts out of N = n_keys**2 with product keys (two sub-key searches of n_keys each, then
top-k over k*k candidates), so lookup costs O(sqrt(N)) instead of O(N). Only the selected rows of
the expert tables are read, and with sparse gradients only those rows are updated. Total
parameters grow with N (knowledge capacity) while compute per token stays roughly flat.
Based on product-key memories (Lample et al. 2019) and PEER (He 2024).

Improvements:
- Explicit total / active parameter helpers
- Optional activation counter for later pruning of redundant experts
- Safer init comments and sparse-gradient notes
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ProductKeyExperts(nn.Module):
    def __init__(self, d_model: int, n_keys: int = 128, heads: int = 4, top_k: int = 16,
                 key_dim: int = 64, sparse: bool = True, track_activations: bool = False):
        super().__init__()
        self.n_keys, self.heads, self.top_k, self.key_dim = n_keys, heads, top_k, key_dim
        self.n_experts = n_keys * n_keys
        self.track_activations = track_activations
        self.query = nn.Linear(d_model, heads * key_dim, bias=False)
        self.keys = nn.Parameter(torch.randn(heads, 2, n_keys, key_dim // 2) / math.sqrt(key_dim // 2))
        # Expert tables as embeddings so backward touches only the selected rows (sparse=True).
        self.down = nn.Embedding(self.n_experts, d_model, sparse=sparse)
        self.up = nn.Embedding(self.n_experts, d_model, sparse=sparse)
        nn.init.normal_(self.down.weight, std=d_model ** -0.5)
        nn.init.zeros_(self.up.weight)  # starts as a no-op, so it can be added to a trained model safely

        if track_activations:
            self.register_buffer("activation_counts", torch.zeros(self.n_experts, dtype=torch.long), persistent=False)

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
        expert_ids = ids.gather(-1, pick)
        if self.track_activations and self.training:
            flat_ids = expert_ids.reshape(-1)
            self.activation_counts.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.long))
        return scores, expert_ids

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        scores, ids = self.route(x)
        gate = scores.softmax(-1)                                          # [T, H, k]
        down = self.down(ids)                                              # [T, H, k, d]
        h = F.gelu(torch.einsum('td,thkd->thk', x, down)) * gate
        # Weighted sum of the chosen up-vectors without materialising a [T, H, k, d] tensor.
        T = ids.shape[0]
        out = F.embedding_bag(ids.reshape(T, -1), self.up.weight, per_sample_weights=h.reshape(T, -1),
                              mode='sum', sparse=self.up.sparse)
        return out.reshape(shape)

    def active_params_per_token(self, d_model: int) -> int:
        return self.heads * self.top_k * 2 * d_model

    def total_expert_params(self) -> int:
        return 2 * self.n_experts * self.down.embedding_dim

    def usage_stats(self):
        if not self.track_activations:
            return None
        counts = self.activation_counts
        used = (counts > 0).sum().item()
        return {
            "n_experts": self.n_experts,
            "used_experts": used,
            "unused_experts": self.n_experts - used,
            "max_activations": counts.max().item(),
            "mean_activations": counts.float().mean().item(),
        }

    def prune_unused(self, min_activations: int = 1) -> int:
        """Return number of experts that would be pruned (does not mutate weights; for analysis)."""
        if not self.track_activations:
            return 0
        return int((self.activation_counts < min_activations).sum().item())
