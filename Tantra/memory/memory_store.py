"""
Tantra High-Density RAM Memory Store
====================================
Decoupled high-density memory architecture for CPU-first models.
Combines:
  1. Inverted BM25 Lexical Index (sub-millisecond keyword lookup)
  2. Dense Quantized Semantic Vector Index (cosine similarity on CPU)
  3. Hybrid Fusion Ranking (Lexical + Semantic)
  4. Ultra-compact in-memory fact storage with disk snapshotting
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from .dense_index import DenseVectorIndex
from .inverted_index import InvertedBM25Index, tokenize_query

N_EMBED_DIM = 64
N_HASH_BUCKETS = 1 << 16


def text_to_dense_vector(text: str, dim: int = N_EMBED_DIM) -> torch.Tensor:
    """Generate a fast, deterministic dense semantic vector without requiring a GPU.
    Uses subword hashing + character n-grams projected to a compact unit sphere."""
    words = tokenize_query(text)
    if not words:
        words = [text.lower().strip() or "empty"]
    vec = torch.zeros(dim, dtype=torch.float32)
    # Word features
    for w in words:
        h = zlib.crc32(w.encode("utf-8"))
        sign = 1.0 if (h & 0x10000) else -1.0
        vec[h % dim] += sign
    # Character trigram / 4-gram features
    padded = f" {' '.join(words)} "
    p_len = len(padded)
    for n in (3, 4):
        for i in range(p_len - n + 1):
            h = zlib.crc32(padded[i : i + n].encode("utf-8"))
            sign = 0.5 if (h & 0x10000) else -0.5
            vec[h % dim] += sign
    return F.normalize(vec.unsqueeze(0), p=2, dim=-1).squeeze(0)


class TantraMemoryStore:
    """High-density RAM memory store with hybrid sparse + dense retrieval."""

    def __init__(self, dim: int = N_EMBED_DIM, alpha: float = 0.6):
        """
        Args:
            dim: Dimension of semantic embedding vectors.
            alpha: Weight for lexical BM25 score (1 - alpha for dense vector score).
        """
        self.dim = dim
        self.alpha = alpha
        self.inverted = InvertedBM25Index()
        self.dense = DenseVectorIndex(dim=dim)
        from .triple_store import TripleStore
        self.triples = TripleStore()
        # doc_id -> (question, answer, expert, metadata)
        self.documents: Dict[int, Tuple[str, str, str, dict]] = {}
        self.next_id: int = 0

    def __len__(self) -> int:
        return len(self.documents)

    def add_fact(self, question: str, answer: str, expert: str = "gk_general",
                 metadata: Optional[dict] = None) -> int:
        """Add a factual entry to the memory store."""
        doc_id = self.next_id
        self.next_id += 1
        meta = metadata or {}
        self.documents[doc_id] = (question.strip(), answer.strip(), expert.strip(), meta)

        # Index in Inverted BM25
        self.inverted.add_document(doc_id, question)

        # Index in Dense Vector Index
        vec = text_to_dense_vector(question, dim=self.dim)
        self.dense.add(vec, doc_id)
        return doc_id

    add = add_fact

    def load_from_datasets(self, data_dirs: Sequence[Union[str, Path]] = ("Datasets/curated",),
                           learned_path: Optional[Union[str, Path]] = "Datasets/learned/facts.jsonl",
                           max_per_file: Optional[int] = None) -> int:
        """Ingest all knowledge files from curated datasets and learned facts."""
        count = 0
        for d in data_dirs:
            dp = Path(d)
            if not dp.exists():
                continue
            for f in dp.glob("knowledge/*.jsonl"):
                expert_name = f.stem
                file_count = 0
                try:
                    for line in f.read_text(encoding="utf-8").splitlines():
                        if max_per_file is not None and file_count >= max_per_file:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if "messages" in row and len(row["messages"]) >= 2:
                            q = row["messages"][-2]["content"]
                            a = row["messages"][-1]["content"]
                            self.add_fact(q, a, expert_name)
                            count += 1
                            file_count += 1
                except Exception:
                    continue

        if learned_path:
            lp = Path(learned_path)
            if lp.exists():
                try:
                    for line in lp.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        q = row.get("q", "")
                        a = row.get("a", "")
                        expert = row.get("expert", "gk_general")
                        if q and a:
                            self.add_fact(q, a, expert, metadata={"learned": True})
                            count += 1
                except Exception:
                    pass
        return count

    def search_hybrid(self, query: str, top_k: int = 3) -> List[Tuple[str, str, str, float]]:
        """Hybrid search combining inverted lexical BM25 and dense semantic similarity.
        Returns list of (question, answer, expert, combined_score)."""
        results: List[Tuple[str, str, str, float]] = []
        # Fast path: DNA triple match
        triple_hit = self.triples.query_natural(query)
        if triple_hit:
            s, p, o = triple_hit
            results.append((query, f"The {p} of {s} is {o}.", "gk_general", 1.0))
            if top_k == 1:
                return results

        # 1. Lexical BM25 search
        bm25_matches = self.inverted.search(query, top_k=top_k * 3)
        max_bm25 = max([score for _, score in bm25_matches], default=1.0)
        norm_bm25 = {doc_id: (score / max(1e-5, max_bm25)) for doc_id, score in bm25_matches}

        # 2. Dense semantic vector search
        q_vec = text_to_dense_vector(query, dim=self.dim)
        dense_matches = self.dense.search(q_vec, top_k=top_k * 3)
        dense_scores = {doc_id: max(0.0, score) for doc_id, score in dense_matches}

        # 3. Combine candidate IDs with query coverage weighting
        q_tokens = set(tokenize_query(query))
        total_q = max(1, len(q_tokens))
        candidate_ids = set(norm_bm25.keys()) | set(dense_scores.keys())
        scored: List[Tuple[int, float]] = []
        for doc_id in candidate_ids:
            lex_score = norm_bm25.get(doc_id, 0.0)
            sem_score = dense_scores.get(doc_id, 0.0)
            doc_q_tokens = set(tokenize_query(self.documents[doc_id][0]))
            coverage = len(q_tokens & doc_q_tokens) / total_q if q_tokens else 1.0
            combined = (self.alpha * lex_score + (1.0 - self.alpha) * sem_score) * coverage
            scored.append((doc_id, combined))

        scored.sort(key=lambda kv: kv[1], reverse=True)
        results = []
        for doc_id, score in scored[:top_k]:
            q, a, e, _ = self.documents[doc_id]
            results.append((q, a, e, score))
        return results

    def search(self, query: str) -> Optional[Tuple[str, str, str, float]]:
        """Drop-in replacement method matching KnowledgeIndex.search API.
        Returns (question, answer, expert, confidence) or None."""
        matches = self.search_hybrid(query, top_k=1)
        if not matches:
            return None
        return matches[0]

    def save(self, path: Union[str, Path]) -> None:
        """Save packed snapshot to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "dim": self.dim,
            "alpha": self.alpha,
            "next_id": self.next_id,
            "documents": self.documents,
            "inverted": self.inverted.state_dict(),
            "dense": self.dense.state_dict(),
            "triples": self.triples.state_dict(),
        }
        torch.save(state, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TantraMemoryStore":
        """Load packed snapshot from disk."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        store = cls(dim=state["dim"], alpha=state["alpha"])
        store.next_id = state["next_id"]
        store.documents = state["documents"]
        store.inverted.load_state_dict(state["inverted"])
        store.dense.load_state_dict(state["dense"])
        if "triples" in state:
            store.triples.load_state_dict(state["triples"])
        return store
