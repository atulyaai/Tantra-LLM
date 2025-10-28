from __future__ import annotations

from typing import List, Dict, Any, Optional


class EpisodicMemory:
    """Episodic memory with retrieval that influences response generation.
    
    Stores:
    - Conversation conclusions
    - User preferences
    - Interpretations/personality notes
    
    Retrieval:
    - Similarity-based search
    - Returns top-k most relevant memories
    """

    def __init__(self):
        self.docs: List[Dict[str, Any]] = []

    def store(self, summary: str, embedding: Optional[List[float]] = None, importance: float = 0.5, tags: Optional[List[str]] = None):
        """Store a memory episode."""
        self.docs.append({
            "summary": summary,
            "embedding": embedding or [],
            "importance": float(importance),
            "tags": tags or [],
        })

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top-k most relevant memories for the query."""
        if not self.docs:
            return []
        scored = []
        q = set(query.lower().split())
        for d in self.docs:
            s = set(str(d.get("summary", "")).lower().split())
            score = len(q.intersection(s)) * (d.get("importance", 0.5))
            if score > 0:
                scored.append((score, d["summary"]))
        scored.sort(reverse=True)
        return [s for _, s in scored[:top_k]]
    
    def clear(self):
        """Clear all memories."""
        self.docs.clear()
    
    def size(self) -> int:
        """Return number of stored episodes."""
        return len(self.docs)

