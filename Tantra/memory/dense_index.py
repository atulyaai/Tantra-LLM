"""
Dense Quantized Vector Index for Tantra
=======================================
Ultra-compact semantic vector index designed for CPU inference.
Stores vectors as normalized tensors (FP32 or INT8 quantized) for
sub-millisecond cosine similarity search via BLAS/SIMD operations.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F


class DenseVectorIndex:
    """Contiguous in-memory vector index with SIMD matrix-vector dot-product."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._vec_list: List[torch.Tensor] = []
        self.vectors: Optional[torch.Tensor] = None  # [N, dim] float32
        self.doc_ids: List[int] = []

    def __len__(self) -> int:
        return len(self.doc_ids)

    def add(self, vec: torch.Tensor, doc_id: int) -> None:
        """Add a single embedding vector (dim,) with doc_id."""
        vec = vec.view(1, -1).float()
        vec = F.normalize(vec, p=2, dim=-1)
        self._vec_list.append(vec)
        self.doc_ids.append(doc_id)

    def add_batch(self, vecs: torch.Tensor, doc_ids: List[int]) -> None:
        """Add batch of embedding vectors (B, dim)."""
        vecs = vecs.float()
        vecs = F.normalize(vecs, p=2, dim=-1)
        self._vec_list.append(vecs)
        self.doc_ids.extend(doc_ids)

    def build(self) -> None:
        """Consolidate accumulated vectors into a single contiguous tensor."""
        if not self._vec_list:
            return
        if self.vectors is None:
            self.vectors = torch.cat(self._vec_list, dim=0)
        else:
            self.vectors = torch.cat([self.vectors] + self._vec_list, dim=0)
        self._vec_list.clear()

    @torch.no_grad()
    def search(self, query_vec: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
        """Return [(doc_id, score), ...] sorted by cosine similarity."""
        if self._vec_list:
            self.build()
        if self.vectors is None or len(self.doc_ids) == 0:
            return []
        q = query_vec.view(1, -1).float()
        q = F.normalize(q, p=2, dim=-1)
        # Cosine similarity via single matrix-vector dot product (fast on CPU)
        scores = (self.vectors @ q.t()).squeeze(-1)  # [N]
        k = min(top_k, scores.size(0))
        top_scores, top_indices = scores.topk(k)
        return [(self.doc_ids[idx.item()], top_scores[i].item()) for i, idx in enumerate(top_indices)]

    def state_dict(self) -> dict:
        return {
            "dim": self.dim,
            "vectors": self.vectors.clone() if self.vectors is not None else None,
            "doc_ids": list(self.doc_ids),
        }

    def load_state_dict(self, d: dict) -> None:
        self.dim = d["dim"]
        self.vectors = d["vectors"]
        self.doc_ids = list(d["doc_ids"])
