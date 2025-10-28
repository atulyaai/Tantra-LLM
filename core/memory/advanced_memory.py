"""
Advanced Memory System
Implements episodic and semantic memory with vector search and graph structures.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel


@dataclass
class MemoryItem:
    """A single memory item."""
    content: str
    timestamp: datetime
    importance: float
    modality: str  # 'text', 'vision', 'audio'
    metadata: Dict[str, Any]
    embedding: Optional[torch.Tensor] = None


class EpisodicMemory:
    """Episodic memory for storing specific experiences."""
    
    def __init__(self, embedding_dim: int = 768, max_memories: int = 10000):
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.memories: List[MemoryItem] = []
        self.embeddings: torch.Tensor = torch.empty(0, embedding_dim)
        self.importance_scores: torch.Tensor = torch.empty(0)
        
        # Load sentence transformer for embeddings
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            # Fallback to basic embedding
            self.encoder = None
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding."""
        if self.encoder is not None:
            embedding = self.encoder.encode(text)
            return torch.tensor(embedding, dtype=torch.float32)
        else:
            # Simple fallback: use character-based encoding
            chars = [ord(c) for c in text[:self.embedding_dim]]
            while len(chars) < self.embedding_dim:
                chars.append(0)
            return torch.tensor(chars[:self.embedding_dim], dtype=torch.float32)
    
    def store(self, content: str, importance: float = 0.5, modality: str = 'text', metadata: Dict = None) -> None:
        """Store a new memory."""
        if metadata is None:
            metadata = {}
        
        # Create memory item
        memory = MemoryItem(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            modality=modality,
            metadata=metadata
        )
        
        # Encode content
        memory.embedding = self._encode_text(content)
        
        # Add to memory list
        self.memories.append(memory)
        
        # Update embeddings tensor
        if self.embeddings.numel() == 0:
            self.embeddings = memory.embedding.unsqueeze(0)
        else:
            self.embeddings = torch.cat([self.embeddings, memory.embedding.unsqueeze(0)], dim=0)
        
        # Update importance scores
        if self.importance_scores.numel() == 0:
            self.importance_scores = torch.tensor([importance])
        else:
            self.importance_scores = torch.cat([self.importance_scores, torch.tensor([importance])])
        
        # Trim if over capacity
        if len(self.memories) > self.max_memories:
            self._trim_memories()
    
    def _trim_memories(self) -> None:
        """Trim memories based on importance and recency."""
        if len(self.memories) <= self.max_memories:
            return
        
        # Calculate combined scores (importance + recency)
        now = datetime.now()
        scores = []
        for memory in self.memories:
            # Recency score (newer = higher)
            recency = 1.0 / (1.0 + (now - memory.timestamp).total_seconds() / 3600)  # Hours
            combined_score = memory.importance * 0.7 + recency * 0.3
            scores.append(combined_score)
        
        # Keep top memories
        scores = torch.tensor(scores)
        _, indices = torch.topk(scores, self.max_memories)
        
        # Update memory list and tensors
        self.memories = [self.memories[i] for i in indices]
        self.embeddings = self.embeddings[indices]
        self.importance_scores = self.importance_scores[indices]
    
    def recall(self, query: str, top_k: int = 5) -> List[str]:
        """Recall relevant memories."""
        if len(self.memories) == 0:
            return []
        
        # Encode query
        query_embedding = self._encode_text(query)
        
        # Compute similarities
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.embeddings,
            dim=1
        )
        
        # Weight by importance
        weighted_scores = similarities * self.importance_scores
        
        # Get top-k memories
        _, indices = torch.topk(weighted_scores, min(top_k, len(self.memories)))
        
        return [self.memories[i].content for i in indices]


class SemanticMemory:
    """Semantic memory for storing facts and concepts."""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.facts: Dict[str, MemoryItem] = {}
        self.embeddings: torch.Tensor = torch.empty(0, embedding_dim)
        self.fact_keys: List[str] = []
        
        # Load sentence transformer for embeddings
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            self.encoder = None
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding."""
        if self.encoder is not None:
            embedding = self.encoder.encode(text)
            return torch.tensor(embedding, dtype=torch.float32)
        else:
            # Simple fallback
            chars = [ord(c) for c in text[:self.embedding_dim]]
            while len(chars) < self.embedding_dim:
                chars.append(0)
            return torch.tensor(chars[:self.embedding_dim], dtype=torch.float32)
    
    def store_fact(self, key: str, content: str, importance: float = 0.8) -> None:
        """Store a semantic fact."""
        memory = MemoryItem(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            modality='text',
            metadata={'type': 'fact', 'key': key}
        )
        
        memory.embedding = self._encode_text(content)
        
        # Update or add fact
        if key in self.facts:
            # Update existing fact
            old_index = self.fact_keys.index(key)
            self.facts[key] = memory
            self.embeddings[old_index] = memory.embedding
        else:
            # Add new fact
            self.facts[key] = memory
            self.fact_keys.append(key)
            
            if self.embeddings.numel() == 0:
                self.embeddings = memory.embedding.unsqueeze(0)
            else:
                self.embeddings = torch.cat([self.embeddings, memory.embedding.unsqueeze(0)], dim=0)
    
    def recall_facts(self, query: str, top_k: int = 5) -> List[str]:
        """Recall relevant facts."""
        if len(self.facts) == 0:
            return []
        
        # Encode query
        query_embedding = self._encode_text(query)
        
        # Compute similarities
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.embeddings,
            dim=1
        )
        
        # Get top-k facts
        _, indices = torch.topk(similarities, min(top_k, len(self.facts)))
        
        return [self.facts[self.fact_keys[i]].content for i in indices]
    
    def get_fact(self, key: str) -> Optional[str]:
        """Get a specific fact by key."""
        return self.facts.get(key, {}).content if key in self.facts else None


class WorkingMemory:
    """Working memory for current context."""
    
    def __init__(self, max_tokens: int = 8192):
        self.max_tokens = max_tokens
        self.context: List[str] = []
        self.current_tokens = 0
    
    def add_context(self, text: str) -> None:
        """Add text to working memory."""
        # Simple token counting (approximate)
        tokens = len(text.split())
        
        # Add to context
        self.context.append(text)
        self.current_tokens += tokens
        
        # Trim if over capacity
        while self.current_tokens > self.max_tokens and len(self.context) > 1:
            removed = self.context.pop(0)
            self.current_tokens -= len(removed.split())
    
    def get_context(self) -> str:
        """Get current context."""
        return " ".join(self.context)
    
    def clear(self) -> None:
        """Clear working memory."""
        self.context = []
        self.current_tokens = 0


class AdvancedMemoryManager:
    """Advanced memory manager combining all memory types."""
    
    def __init__(self, embedding_dim: int = 768, max_episodic: int = 10000):
        self.episodic = EpisodicMemory(embedding_dim, max_episodic)
        self.semantic = SemanticMemory(embedding_dim)
        self.working = WorkingMemory()
        
        # Memory consolidation parameters
        self.consolidation_threshold = 0.7
        self.consolidation_frequency = 100  # Consolidate every 100 new memories
        self.memory_count = 0
    
    def store(self, content: str, importance: float = 0.5, modality: str = 'text', metadata: Dict = None) -> None:
        """Store content in appropriate memory system."""
        if metadata is None:
            metadata = {}
        
        # Store in episodic memory
        self.episodic.store(content, importance, modality, metadata)
        
        # Add to working memory
        self.working.add_context(content)
        
        # Increment counter
        self.memory_count += 1
        
        # Consolidate if needed
        if self.memory_count % self.consolidation_frequency == 0:
            self._consolidate_memories()
    
    def recall(self, query: str, top_k: int = 5) -> List[str]:
        """Recall relevant information from all memory systems."""
        # Get episodic memories
        episodic_recall = self.episodic.recall(query, top_k // 2)
        
        # Get semantic facts
        semantic_recall = self.semantic.recall_facts(query, top_k // 2)
        
        # Combine and return
        return episodic_recall + semantic_recall
    
    def _consolidate_memories(self) -> None:
        """Consolidate important episodic memories to semantic memory."""
        # Get high-importance memories
        high_importance = [
            memory for memory in self.episodic.memories
            if memory.importance > self.consolidation_threshold
        ]
        
        # Convert to semantic facts
        for memory in high_importance:
            # Create a semantic key
            key = f"fact_{memory.timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Store as semantic fact
            self.semantic.store_fact(key, memory.content, memory.importance)
    
    def get_working_context(self) -> str:
        """Get current working memory context."""
        return self.working.get_context()
    
    def clear_working_memory(self) -> None:
        """Clear working memory."""
        self.working.clear()
    
    def store_fact(self, key: str, content: str, importance: float = 0.8) -> None:
        """Store a semantic fact directly."""
        self.semantic.store_fact(key, content, importance)
    
    def get_fact(self, key: str) -> Optional[str]:
        """Get a specific fact."""
        return self.semantic.get_fact(key)
    
    def consider_store(self, content: str, importance: float = 0.5) -> None:
        """Consider storing content in memory (alias for store)."""
        self.store(content, importance)