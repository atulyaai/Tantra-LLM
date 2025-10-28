from __future__ import annotations

from typing import List, Dict, Any, Optional


class EpisodicMemory:
    """Stub: vector-store backed episodic memory (to integrate with Chroma/FAISS)."""

    def __init__(self):
        self.docs: List[Dict[str, Any]] = []

    def store(self, summary: str, embedding: Optional[List[float]] = None, importance: float = 0.5, tags: Optional[List[str]] = None):
        self.docs.append({
            "summary": summary,
            "embedding": embedding or [],
            "importance": float(importance),
            "tags": tags or [],
        })

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        # TODO: replace lexical overlap with vector similarity
        scored = []
        q = set(query.lower().split())
        for d in self.docs:
            s = set(str(d.get("summary", "")).lower().split())
            score = len(q.intersection(s)) * (d.get("importance", 0.5))
            if score > 0:
                scored.append((score, d["summary"]))
        scored.sort(reverse=True)
        return [s for _, s in scored[:top_k]]


