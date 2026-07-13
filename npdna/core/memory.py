import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from npdna.atulya_core.protocol.memory import MemoryStore, MemoryChunk

class InMemoryVectorStore(MemoryStore):
    """
    A concrete reference implementation of the MemoryStore protocol.
    Uses basic cosine similarity over word-overlap frequency vectors for deterministic semantic search.
    """
    
    def __init__(self, embed_dim: int = 4096):
        self.embed_dim = embed_dim
        # List of indexed memories, each dict has "content", "metadata", "embedding" (tensor [embed_dim])
        self.registry: List[Dict[str, Any]] = []

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Generates a normalized bag-of-words frequency vector representing the text.
        Ensures semantic overlap between queries and matching documents.
        """
        words = text.lower().replace(".", "").replace(",", "").split()
        embedding = torch.zeros(self.embed_dim)
        
        for w in words:
            # Hash word deterministically to a slot in the embedding dimension
            idx = sum(ord(c) * (i + 1) for i, c in enumerate(w)) % self.embed_dim
            embedding[idx] += 1.0
            
        # Add epsilon to prevent divide-by-zero on empty input
        embedding += 1e-6
        return F.normalize(embedding, dim=-1)

    async def retrieve(self, query: str, k: int = 5) -> List[MemoryChunk]:
        """Retrieve top k most relevant memory chunks using cosine similarity."""
        if not self.registry:
            return []

        query_embed = self._get_text_embedding(query)
        
        # Build embedding matrix
        stored_embeds = torch.stack([m["embedding"] for m in self.registry]) # [N, embed_dim]
        
        # Calculate cosine similarity: dot product of normalized vectors
        scores = torch.matmul(stored_embeds, query_embed) # [N]
        
        # Sort and get top k
        top_k_val, top_k_idx = torch.topk(scores, min(k, len(self.registry)))
        
        results = []
        for score, idx in zip(top_k_val.tolist(), top_k_idx.tolist()):
            item = self.registry[idx]
            results.append(MemoryChunk(
                content=item["content"],
                score=round(score, 4),
                metadata=item["metadata"]
            ))
        return results

    async def write(self, content: str, metadata: Dict[str, Any]) -> None:
        """Indexes new memory content block and its metadata."""
        embedding = self._get_text_embedding(content)
        self.registry.append({
            "content": content,
            "metadata": metadata,
            "embedding": embedding
        })

    async def consolidate(self) -> None:
        """Performs mock vector index consolidation and cleanup."""
        # Clean up duplicates
        seen = set()
        unique_registry = []
        for item in self.registry:
            if item["content"] not in seen:
                seen.add(item["content"])
                unique_registry.append(item)
        self.registry = unique_registry
        print(f"[MemoryStore] Consolidated vectors. Total index size: {len(self.registry)}")
