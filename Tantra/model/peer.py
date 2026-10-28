"""Product-key expert layer: thousands of tiny experts, a handful active per token (CPU-friendly).

Each expert is one hidden neuron (a down vector u_i and an up vector v_i). A token picks its
top-k experts out of N = n_keys**2 with product keys (two sub-key searches of n_keys each, then
top-k over k*k candidates), so lookup costs O(sqrt(N)) instead of O(N). Only the selected rows of
the expert tables are read, and with sparse gradients only those rows are updated. Total
parameters grow with N (knowledge capacity) while compute per token stays roughly flat.
Based on product-key memories (Lample et al. 2019) and PEER (He 2024).

CPU size path (optional, inference):
- factor_low_rank_: shared basis + per-expert codes (about d/rank smaller)
- quantize_int8_: per-row INT8 + scale on whatever table is stored (~4x)
- compact_used_: drop experts that were never (or rarely) activated
Together these are the practical route toward ~100:1 expert-table size on CPU.
Float training path is unchanged until one of those methods is called.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _row_int8(weight: Tensor):
    """Per-row INT8. Returns (codes int8 [N, D], scales float [N])."""
    scale = weight.detach().abs().amax(dim=-1).clamp_min(1e-8) / 127.0
    codes = torch.round(weight.detach() / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return codes, scale


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

        self.low_rank = False
        self.quantized = False
        self.compacted = False

    def route(self, x: Tensor):
        """Top-k (scores, expert ids) per head for x of shape [T, d]."""
        T = x.shape[0]
        q = self.query(x).view(T, self.heads, 2, self.key_dim // 2)
        s1 = torch.einsum('thd,hnd->thn', q[:, :, 0], self.keys[:, 0])
        s2 = torch.einsum('thd,hnd->thn', q[:, :, 1], self.keys[:, 1])
        v1, i1 = s1.topk(self.top_k, -1)
        v2, i2 = s2.topk(self.top_k, -1)
        cand = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(-2)
        ids = (i1.unsqueeze(-1) * self.n_keys + i2.unsqueeze(-2)).flatten(-2)
        scores, pick = cand.topk(self.top_k, -1)
        expert_ids = ids.gather(-1, pick)
        if self.track_activations and self.training:
            flat_ids = expert_ids.reshape(-1)
            self.activation_counts.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.long))
        return scores, expert_ids

    def _remap(self, ids: Tensor) -> Tensor:
        if not self.compacted:
            return ids
        return self.id_map[ids]

    def _gather_table(self, ids: Tensor, which: str) -> Tensor:
        """Selected rows [T, H, k, d] from float, INT8, or low-rank storage."""
        ids = self._remap(ids)
        if self.low_rank:
            basis = getattr(self, f"{which}_basis")
            if self.quantized:
                code = getattr(self, f"{which}_code_q")
                scale = getattr(self, f"{which}_code_scale")
                gathered = code[ids].float() * scale[ids].unsqueeze(-1)
            else:
                gathered = getattr(self, f"{which}_code")[ids]
            return gathered @ basis
        if self.quantized:
            q = getattr(self, f"{which}_q")
            scale = getattr(self, f"{which}_scale")
            return q[ids].float() * scale[ids].unsqueeze(-1)
        table = self.down if which == "down" else self.up
        return table(ids)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        scores, ids = self.route(x)
        gate = scores.softmax(-1)
        down = self._gather_table(ids, "down")
        h = F.gelu(torch.einsum('td,thkd->thk', x, down)) * gate
        if self.low_rank or self.quantized or self.compacted:
            up = self._gather_table(ids, "up")
            out = torch.einsum('thk,thkd->td', h, up)
        else:
            T = ids.shape[0]
            out = F.embedding_bag(
                ids.reshape(T, -1), self.up.weight,
                per_sample_weights=h.reshape(T, -1),
                mode='sum', sparse=self.up.sparse,
            )
        return out.reshape(shape)

    def active_params_per_token(self, d_model: int) -> int:
        if self.low_rank:
            rank = self.down_basis.shape[0]
            return self.heads * self.top_k * 2 * rank + 2 * rank * d_model
        return self.heads * self.top_k * 2 * d_model

    def total_expert_params(self) -> int:
        if self.low_rank:
            return int(self.down_code.numel() + self.up_code.numel()
                       + self.down_basis.numel() + self.up_basis.numel())
        if self.compacted:
            return int(self.down.weight.numel() + self.up.weight.numel())
        return 2 * self.n_experts * self.down.embedding_dim

    def stored_bytes(self) -> int:
        """Bytes holding expert tables (not the router). Each tensor is counted once."""
        seen = set()
        n = 0
        names = [
            "down_q", "up_q", "down_scale", "up_scale",
            "down_code", "up_code", "down_code_q", "up_code_q",
            "down_code_scale", "up_code_scale",
            "down_basis", "up_basis", "id_map",
        ]
        if not self.quantized and not self.low_rank:
            names.extend(["down.weight", "up.weight"])
        for name in names:
            if "." in name:
                mod, attr = name.split(".")
                t = getattr(getattr(self, mod, None), attr, None)
            else:
                t = getattr(self, name, None)
            if t is None or not torch.is_tensor(t):
                continue
            key = t.data_ptr()
            if key in seen:
                continue
            seen.add(key)
            n += t.numel() * t.element_size()
        return int(n)

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
        """How many experts fall below min_activations. Does not change weights."""
        if not self.track_activations:
            return 0
        return int((self.activation_counts < min_activations).sum().item())

    @torch.no_grad()
    def factor_low_rank_(self, rank: int, sample: Optional[int] = None) -> dict:
        """Replace expert tables with a shared basis plus per-expert codes.

        Reconstruction is W ≈ codes @ basis. Storage ratio is about d_model / rank
        when N is large. Call quantize_int8_() afterwards for another ~4x.
        """
        if self.quantized or self.compacted:
            raise RuntimeError("factor_low_rank_ expects the original float tables")
        rank = int(rank)
        d = self.down.embedding_dim
        if rank < 1 or rank >= d:
            raise ValueError(f"rank must be in 1..{d - 1}")

        def _factor(weight: Tensor):
            n = weight.shape[0]
            take = n if sample is None else min(n, int(sample))
            idx = torch.randperm(n, device=weight.device)[:take]
            _, _, vh = torch.linalg.svd(weight[idx].float(), full_matrices=False)
            basis = vh[:rank].contiguous()
            codes = (weight.float() @ basis.T).contiguous()
            err = (weight.float() - codes @ basis).pow(2).mean().sqrt()
            ref = weight.float().pow(2).mean().sqrt().clamp_min(1e-8)
            return codes, basis, float((err / ref).item())

        down_code, down_basis, down_err = _factor(self.down.weight.data)
        up_code, up_basis, up_err = _factor(self.up.weight.data)
        self.down_code = nn.Parameter(down_code)
        self.up_code = nn.Parameter(up_code)
        self.down_basis = nn.Parameter(down_basis)
        self.up_basis = nn.Parameter(up_basis)
        # Free the full float tables so size matches the factored form.
        self.down = nn.Embedding(1, 1)
        self.up = nn.Embedding(1, 1)
        self.low_rank = True
        return {"rank": rank, "down_rel_rmse": down_err, "up_rel_rmse": up_err,
                "stored_bytes": self.stored_bytes()}

    @torch.no_grad()
    def quantize_int8_(self) -> dict:
        """Pack stored expert rows or low-rank codes to per-row INT8. Inference only."""
        if self.quantized:
            return {"stored_bytes": self.stored_bytes(), "already": True}
        before = self.stored_bytes()
        if self.low_rank:
            dq, ds = _row_int8(self.down_code.data)
            uq, us = _row_int8(self.up_code.data)
            self.register_buffer("down_code_q", dq)
            self.register_buffer("up_code_q", uq)
            self.register_buffer("down_code_scale", ds)
            self.register_buffer("up_code_scale", us)
            del self.down_code
            del self.up_code
        else:
            dq, ds = _row_int8(self.down.weight.data)
            uq, us = _row_int8(self.up.weight.data)
            self.register_buffer("down_q", dq)
            self.register_buffer("up_q", uq)
            self.register_buffer("down_scale", ds)
            self.register_buffer("up_scale", us)
        self.quantized = True
        after = self.stored_bytes()
        return {"before_bytes": before, "stored_bytes": after, "ratio": before / max(after, 1)}

    @torch.no_grad()
    def compact_used_(self, min_activations: int = 1) -> dict:
        """Drop experts below min_activations. Routing ids stay valid via id_map.

        Index 0 of the new tables is a reserved zero row. Unused ids map to it.
        """
        if not self.track_activations:
            raise RuntimeError("compact_used_ needs track_activations=True")
        if self.low_rank or self.quantized or self.compacted:
            raise RuntimeError("compact_used_ expects the original float tables")
        keep = (self.activation_counts >= min_activations).nonzero().flatten()
        n_keep = int(keep.numel())
        d = self.down.embedding_dim
        device = self.down.weight.device
        id_map = torch.zeros(self.n_experts, dtype=torch.long, device=device)
        if n_keep:
            id_map[keep] = torch.arange(1, n_keep + 1, device=device)
        down_w = torch.zeros(n_keep + 1, d, device=device)
        up_w = torch.zeros(n_keep + 1, d, device=device)
        if n_keep:
            down_w[1:] = self.down.weight.data[keep]
            up_w[1:] = self.up.weight.data[keep]
        before = self.stored_bytes()
        self.down = nn.Embedding.from_pretrained(down_w, freeze=False, sparse=True)
        self.up = nn.Embedding.from_pretrained(up_w, freeze=False, sparse=True)
        self.register_buffer("id_map", id_map)
        self.compacted = True
        after = self.stored_bytes()
        return {
            "kept": n_keep,
            "dropped": self.n_experts - n_keep,
            "before_bytes": before,
            "stored_bytes": after,
            "ratio": before / max(after, 1),
        }
