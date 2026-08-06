"""Cognition: compute routing, dynamic context, planning, nervous system, memory store,
and Memory Cortex (external knowledge store for NP-DNA).

Single flat module (previously ``core/compute_routing.py``, ``core/dynamic_context.py``,
``core/planning.py``, ``core/nervous_system.py``, ``core/memory.py``,
and ``cortex.py``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path as _Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from npdna.schema import (
    IDENTITY,
    MemoryChunk,
    MemoryStore,
    SystemPulse,
    TaskPlan,
    TaskStep,
)
from .architecture import CortexConfig


class ComputeRouter:
    """Routes queries to fast/medium/deep compute paths dynamically based on complexity and recorded history."""

    def __init__(self):
        self.fast_max_tokens = 50
        self.medium_max_tokens = 200
        self.deep_max_tokens = 500

        # Performance history: key is (path, provider), value is List[Tuple[latency_ms, cost]]
        self.history: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self.history_limit = 10

        # Load identity configuration
        latency_pref = IDENTITY.get("latency_vs_precision", {})
        self.fast_threshold = latency_pref.get("fast_threshold_ms", 500)
        self.medium_threshold = latency_pref.get("medium_threshold_ms", 2000)
        self.deep_threshold = latency_pref.get("deep_threshold_ms", 10000)

    def record_performance(self, path: str, provider: str, latency_ms: float, cost: float):
        """Records latency and cost metrics for a given path and provider to adapt routing thresholds."""
        key = (path, provider)
        if key not in self.history:
            self.history[key] = []

        self.history[key].append((latency_ms, cost))
        if len(self.history[key]) > self.history_limit:
            self.history[key].pop(0)

    def get_average_performance(self, path: str, provider: str) -> Tuple[float, float]:
        """Returns the rolling average (latency_ms, cost) for the specified path and provider."""
        key = (path, provider)
        records = self.history.get(key, [])
        if not records:
            # Return default baseline estimates
            baselines = {
                "fast": (150.0, 0.0),
                "medium": (800.0, 0.001),
                "deep": (4000.0, 0.005)
            }
            return baselines.get(path, (500.0, 0.002))

        avg_latency = sum(r[0] for r in records) / len(records)
        avg_cost = sum(r[1] for r in records) / len(records)
        return avg_latency, avg_cost

    def analyze_complexity(self, query: str, context_len: int = 0) -> float:
        """Score query complexity (0=simple, 1=complex)."""
        complexity = 0.0

        # Length heuristic
        if len(query) > 200:
            complexity += 0.3

        # Question marks suggest reasoning needed
        if "?" in query:
            complexity += 0.2

        # Complex keywords
        complex_keywords = ["explain", "how", "why", "analyze", "compare", "design", "plan", "summarize", "describe"]
        if any(kw in query.lower() for kw in complex_keywords):
            complexity += 0.3

        # Mode-specific complexity
        if "mode:" in query.lower():
            complexity += 0.1  # Mode switching is simple

        # Context length
        if context_len > 10000:
            complexity += 0.2

        return min(complexity, 1.0)

    def select_path(self, query: str, provider: str = "local", context_len: int = 0) -> Literal["fast", "medium", "deep"]:
        """Return fast/medium/deep path based on complexity, adjusted for recorded performance history."""
        score = self.analyze_complexity(query, context_len)

        # Determine baseline path recommendation
        if score < 0.3:
            base_path = "fast"
        elif score < 0.7:
            base_path = "medium"
        else:
            base_path = "deep"

        # Self-adaptation check: check if the recommended path has been under pressure (slow/costly)
        avg_latency, _ = self.get_average_performance(base_path, provider)

        # Shorten under pressure check
        if base_path == "deep" and avg_latency > self.deep_threshold:
            # Fallback to medium path to conserve system resources
            print(f"[AdaptiveRouter] Pressure detected on 'deep' path (avg latency={avg_latency:.1f}ms > threshold={self.deep_threshold}ms). Downgrading route to 'medium'.")
            return "medium"

        if base_path == "medium" and avg_latency > self.medium_threshold:
            print(f"[AdaptiveRouter] Pressure detected on 'medium' path (avg latency={avg_latency:.1f}ms > threshold={self.medium_threshold}ms). Downgrading route to 'fast'.")
            return "fast"

        return base_path

    def get_max_tokens(self, path: str) -> int:
        """Get max_tokens for selected path."""
        mapping = {
            "fast": self.fast_max_tokens,
            "medium": self.medium_max_tokens,
            "deep": self.deep_max_tokens,
        }
        return mapping.get(path, self.medium_max_tokens)

    def get_max_context(self, path: str) -> int:
        """Get max context window for selected path."""
        mapping = {
            "fast": 2048,
            "medium": 4096,
            "deep": 8192,
        }
        return mapping.get(path, 4096)


class DynamicContextManager:
    """Dynamic context sizing with importance-based trimming.

    Provides:
    - Short contexts (2K-4K tokens) for quick responses
    - Long contexts (16K-32K tokens) for reasoning/planning
    - Sliding window attention with recurrent state caching
    """

    def __init__(
        self,
        max_short: int = 4096,
        max_long: int = 32768,
        importance_threshold: float = 0.5,
    ):
        self.max_short = max_short
        self.max_long = max_long
        self.importance_threshold = importance_threshold
        self.recurrent_state: Optional[Dict[str, Any]] = None

    def select_window(self, task_metadata: Dict[str, Any]) -> int:
        """Return target context length based on task characteristics.

        Args:
            task_metadata: Dict with keys like 'urgency', 'complexity', 'type'

        Returns:
            Target context length in tokens
        """
        urgency = task_metadata.get("urgency", 0.5)
        complexity = task_metadata.get("complexity", 0.5)
        task_type = task_metadata.get("type", "unknown")

        # Fast path: simple queries
        if urgency > 0.8 and complexity < 0.3:
            return self.max_short

        # Deep path: complex reasoning
        if task_type in ["plan", "analyze", "reasoning"]:
            return self.max_long

        # Medium path: default
        return (self.max_short + self.max_long) // 2

    def trim(self, token_ids: List[int], target_len: int) -> List[int]:
        """Trim token sequence to fit target_len with sliding window."""
        if len(token_ids) <= target_len:
            return token_ids

        # Sliding window truncation: keep the most recent tokens (end of sequence)
        return token_ids[-target_len:]

    def update_recurrent_state(self, kv_cache: Optional[Dict[str, torch.Tensor]] = None):
        """Update recurrent state for KV-cache reuse.

        Args:
            kv_cache: Key-value cache from transformer attention
        """
        self.recurrent_state = kv_cache or {}

    def get_recurrent_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get current recurrent state.

        Returns:
            KV-cache dict or None
        """
        return self.recurrent_state


class EventBus:
    """
    Async pub/sub event bus. Broadcasts state changes across all modular organs.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 2020):
        self.host = host
        self.port = port
        self.subscribers = {}

    def subscribe(self, event_type: str, callback: callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    async def emit(self, event_type: str, data: dict):
        if event_type in self.subscribers:
            for cb in self.subscribers[event_type]:
                await cb(data)

    async def heartbeat(self):
        """Main 20Hz Loop — reports real system load via psutil."""
        import psutil
        while True:
            try:
                cpu_load = float(psutil.cpu_percent(interval=None))
                mem_usage = float(psutil.virtual_memory().percent)
            except Exception:
                cpu_load, mem_usage = 0.0, 0.0
            pulse = SystemPulse(cpu_load=cpu_load, mem_usage=mem_usage, active_modules=[])
            await self.emit("pulse", pulse.model_dump())
            await asyncio.sleep(0.05)


class InMemoryVectorStore(MemoryStore):
    """
    A concrete reference implementation of the MemoryStore protocol.
    Uses feature hashing (word tokens + character 3-grams) over embedding dimensions for robust semantic/lexical search.
    """

    def __init__(self, embed_dim: int = 4096):
        self.embed_dim = embed_dim
        # List of indexed memories, each dict has "content", "metadata", "embedding" (tensor [embed_dim])
        self.registry: List[Dict[str, Any]] = []

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Generates a normalized feature-hashing vector (word tokens + character n-grams)
        representing the text across the embedding space.
        """
        text_clean = text.lower().strip()
        if not text_clean:
            return F.normalize(torch.full((self.embed_dim,), 1e-6), dim=-1)

        words = text_clean.replace(".", "").replace(",", "").replace("!", "").split()
        embedding = torch.zeros(self.embed_dim)

        # Word-level feature hashing
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % self.embed_dim
            embedding[h] += 1.0

        # Character 3-gram feature hashing for subword matching
        for i in range(len(text_clean) - 2):
            gram = text_clean[i:i+3]
            h = int(hashlib.sha256(gram.encode("utf-8")).hexdigest(), 16) % self.embed_dim
            embedding[h] += 0.5

        embedding += 1e-6
        return F.normalize(embedding, dim=-1)

    async def retrieve(self, query: str, k: int = 5) -> List[MemoryChunk]:
        """Retrieve top k most relevant memory chunks using cosine similarity."""
        if not self.registry:
            return []

        query_embed = self._get_text_embedding(query)

        # Build embedding matrix
        stored_embeds = torch.stack([m["embedding"] for m in self.registry])  # [N, embed_dim]

        # Calculate cosine similarity: dot product of normalized vectors
        scores = torch.matmul(stored_embeds, query_embed)  # [N]

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


# ── Fast Response Memory (from response_memory.py) ──────────────────────────


class FastResponseMemory:
    """Associative Q&A memory with directly written, normalized key weights."""

    def __init__(self, dimension: int = 1024, threshold: float = 0.94) -> None:
        self.dimension = dimension
        self.threshold = threshold
        self.keys = torch.empty((0, dimension), dtype=torch.float32)
        self.questions: list[str] = []
        self.answers: list[str] = []

    def _encode(self, question: str) -> torch.Tensor:
        words = re.findall(r"[\w']+", question.lower())
        features = words + [f"{a}|{b}" for a, b in zip(words, words[1:])]
        vector = torch.zeros(self.dimension, dtype=torch.float32)
        for feature in features:
            index = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "little") % self.dimension
            vector[index] += 1.0
        return vector / vector.norm().clamp_min(1.0)

    def write(self, question: str, answer: str) -> None:
        normalized = question.strip()
        try:
            index = self.questions.index(normalized)
        except ValueError:
            self.questions.append(normalized)
            self.answers.append(answer.strip())
            self.keys = torch.cat([self.keys, self._encode(normalized).unsqueeze(0)])
        else:
            self.answers[index] = answer.strip()
            self.keys[index] = self._encode(normalized)

    def match(self, question: str) -> tuple[str, float] | None:
        if not self.answers:
            return None
        scores = self.keys @ self._encode(question)
        score, index = scores.max(dim=0)
        if float(score) < self.threshold:
            return None
        return self.answers[int(index)], float(score)

    def save(self, path: str | _Path) -> None:
        torch.save({
            "dimension": self.dimension, "threshold": self.threshold,
            "keys": self.keys, "questions": self.questions, "answers": self.answers,
        }, _Path(path))

    @classmethod
    def load(cls, path: str | _Path) -> "FastResponseMemory":
        state = torch.load(_Path(path), map_location="cpu", weights_only=True)
        memory = cls(dimension=int(state["dimension"]), threshold=float(state["threshold"]))
        memory.keys = state["keys"].float()
        memory.questions = list(state["questions"])
        memory.answers = list(state["answers"])
        return memory


# ── Memory Cortex (from cortex.py) ─────────────────────────────────────────

logger = logging.getLogger(__name__)


@dataclass
class CortexEntry:
    """A single knowledge entry in the Cortex."""

    key: Tensor            # Query/key vector (dim,)
    value: Tensor          # Value vector (dim,)
    topic: str = ""
    topics: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    source: str = ""
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


class MemoryCortex(torch.nn.Module):
    """External vector memory.  Store and retrieve knowledge without retraining.

    Args:
        config: Cortex configuration (dim, max entries, top_k).
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        self.config = config
        self.entries: list[CortexEntry] = []

        # Projection layer: hidden_state -> query vector
        self.query_proj = torch.nn.Linear(config.dim, config.dim, bias=False)
        self.value_proj = torch.nn.Linear(config.dim, config.dim, bias=False)
        self._last_top_indices = None
        self._last_top_scores = None
        self._keys_cache: Tensor | None = None
        self._values_cache: Tensor | None = None
        self._cache_dirty = True

    @property
    def size(self) -> int:
        return len(self.entries)

    def store(
        self,
        key: Tensor,
        value: Tensor | None = None,
        topic: str = "",
        source: str = "",
    ) -> int:
        """Store a knowledge entry.  Returns entry index."""
        if value is None:
            value = key.clone()

        key = key.detach().float().cpu()
        value = value.detach().float().cpu()

        # Enforce max capacity -- evict least-accessed entry
        if self.size >= self.config.max_entries:
            self._evict_least_used()

        entry = CortexEntry(key=key, value=value, topic=topic, topics=[topic] if topic else [], source=source)
        self.entries.append(entry)
        self._invalidate_cache()
        return self.size - 1

    def store_batch(
        self,
        keys: Tensor,
        values: Tensor | None = None,
        topic: str = "",
        source: str = "",
    ) -> list[int]:
        """Store multiple knowledge entries at once.

        Args:
            keys: (batch, dim) tensor of key vectors.
            values: Optional (batch, dim) tensor of value vectors.
            topic: Topic label applied to all entries.
            source: Source string applied to all entries.

        Returns:
            List of entry indices.
        """
        if values is None:
            values = keys.clone()
        indices = []
        for i in range(keys.shape[0]):
            idx = self.store(keys[i], values[i], topic=topic, source=source)
            indices.append(idx)
        return indices

    def _invalidate_cache(self) -> None:
        self._keys_cache = None
        self._values_cache = None
        self._cache_dirty = True

    def _stacked_vectors(self, device: torch.device) -> tuple[Tensor, Tensor]:
        if (
            self._cache_dirty
            or self._keys_cache is None
            or self._values_cache is None
            or self._keys_cache.device != device
        ):
            self._keys_cache = torch.stack([e.key for e in self.entries]).to(device)
            self._values_cache = torch.stack([e.value for e in self.entries]).to(device)
            self._cache_dirty = False
        return self._keys_cache, self._values_cache

    def retrieve(self, query: Tensor, top_k: int | None = None) -> tuple[Tensor, Tensor]:
        """Find most relevant knowledge for a query.

        Args:
            query: Query vector (dim,) or (batch, dim).
            top_k: Number of entries to retrieve.

        Returns:
            (values, scores) -- retrieved value vectors and similarity scores.
        """
        if self.size == 0:
            self._last_top_indices = None
            self._last_top_scores = None
            dim = self.config.dim
            k = top_k or self.config.top_k
            if query.dim() == 1:
                return torch.zeros(k, dim, device=query.device, dtype=query.dtype), torch.zeros(k, device=query.device, dtype=query.dtype)
            return torch.zeros(query.size(0), k, dim, device=query.device, dtype=query.dtype), torch.zeros(query.size(0), k, device=query.device, dtype=query.dtype)

        top_k = min(top_k or self.config.top_k, self.size)

        # Cosine similarity
        is_1d = query.dim() == 1
        if is_1d:
            query = query.unsqueeze(0)
        keys, values = self._stacked_vectors(query.device)

        query_norm = torch.nn.functional.normalize(query, dim=-1)   # (B, dim)
        keys_norm = torch.nn.functional.normalize(keys, dim=-1)     # (N, dim)
        scores = query_norm @ keys_norm.T  # (B, N)

        top_scores, top_indices = torch.topk(scores, top_k, dim=-1)  # (B, k)

        # Expand indices for gather
        expanded = top_indices.unsqueeze(-1).expand(-1, -1, values.size(-1))
        top_values = torch.gather(
            values.unsqueeze(0).expand(query.size(0), -1, -1),
            dim=1,
            index=expanded,
        )  # (B, k, dim)

        # Update access counts
        if not getattr(self, "_is_sleeping", False):
            for idx_row in top_indices:
                for idx in idx_row:
                    self.entries[idx.item()].access_count += 1

        if not self.training:
            self._last_top_indices = top_indices.detach().cpu()
            self._last_top_scores = top_scores.detach().cpu()
        else:
            self._last_top_indices = None
            self._last_top_scores = None

        if is_1d:
            return top_values.squeeze(0), top_scores.squeeze(0)
        return top_values, top_scores

    def augment(self, hidden: Tensor) -> Tensor:
        """Augment a hidden state with retrieved Cortex knowledge.

        Args:
            hidden: Hidden state (batch, seq_len, dim) or (batch, dim).

        Returns:
            Augmented hidden state, same shape as input.
        """
        if self.size == 0:
            return hidden

        squeeze_seq = False
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
            squeeze_seq = True

        B, T, D = hidden.shape
        query = self.query_proj(hidden.reshape(-1, D))  # (B*T, D)
        values, scores = self.retrieve(query)           # (B*T, k, D), (B*T, k)
        relevant = scores.max(dim=-1).values >= self.config.min_relevance
        if not relevant.any():
            return hidden.squeeze(1) if squeeze_seq else hidden

        # Soft attention over retrieved values
        attn = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B*T, k, 1)
        context = (values * attn).sum(dim=1)                 # (B*T, D)
        context = context * relevant.to(context.dtype).unsqueeze(-1)
        context = self.value_proj(context)

        augmented = hidden + context.reshape(B, T, D)

        if squeeze_seq:
            augmented = augmented.squeeze(1)

        return augmented

    def _evict_least_used(self) -> None:
        """Remove the least-accessed entry."""
        if not self.entries:
            return
        min_idx = min(range(len(self.entries)), key=lambda i: self._importance_score(self.entries[i]))
        removed = self.entries.pop(min_idx)
        self._invalidate_cache()
        logger.debug("Cortex evicted entry (topic=%s, accesses=%d)", removed.topic, removed.access_count)

    @staticmethod
    def _importance_score(entry: CortexEntry) -> float:
        age_hours = max(0.0, (time.time() - entry.created_at) / 3600)
        relationship_boost = len(entry.related) * 0.25
        topic_boost = len(entry.topics) * 0.1
        return entry.access_count + relationship_boost + topic_boost - age_hours * 0.01

    def link_entries(self, source_idx: int, target_idx: int) -> None:
        if not (0 <= source_idx < self.size and 0 <= target_idx < self.size):
            return
        source = self.entries[source_idx]
        target = self.entries[target_idx]
        target_id = str(target_idx)
        source_id = str(source_idx)
        if target_id not in source.related:
            source.related.append(target_id)
        if source_id not in target.related:
            target.related.append(source_id)

    def prune_by_importance(self, max_entries: int | None = None) -> int:
        max_entries = max_entries or self.config.max_entries
        if self.size <= max_entries:
            return 0
        before = self.size
        self.entries.sort(key=self._importance_score, reverse=True)
        self.entries = self.entries[:max_entries]
        self._invalidate_cache()
        return before - self.size

    def store_from_text(self, text: str, encoder_fn: Any, topic: str = "") -> int:
        """Convenience: encode text and store it.

        Args:
            text: Raw text to store.
            encoder_fn: Callable that converts text -> tensor (dim,).
            topic: Topic label.

        Returns:
            Entry index.
        """
        with torch.no_grad():
            vec = encoder_fn(text)
        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec, dtype=torch.float32)
        return self.store(vec, topic=topic, source=text[:200])

    def sleep_cycle(
        self,
        similarity_threshold: float = 0.90,
        max_capacity: int | None = None,
        core: Any | None = None,
    ) -> dict[str, int]:
        """Perform a consolidation pass to merge duplicate facts and enforce capacity.

        Uses pure PyTorch operations to compute cosine similarities and greedily group entries.
        """
        if self.size == 0:
            return {"before": 0, "after": 0, "merged": 0, "evicted": 0, "active_writeback": 0}

        max_capacity = max_capacity or self.config.max_entries
        old_size = self.size

        # Compute normalized keys once; similarity rows are produced in bounded blocks.
        with torch.no_grad():
            keys = torch.stack([e.key for e in self.entries])  # (N, dim)
            keys_norm = torch.nn.functional.normalize(keys, dim=-1)  # (N, dim)

        # Bound temporary similarity memory to roughly 32 MiB of float32 values.
        row_block_size = max(1, min(1024, (8 * 1024 * 1024) // old_size))
        visited = set()
        consolidated_entries: list[CortexEntry] = []

        for block_start in range(0, old_size, row_block_size):
            block_end = min(block_start + row_block_size, old_size)
            with torch.no_grad():
                similarity_rows = keys_norm[block_start:block_end] @ keys_norm.T

            for i in range(block_start, block_end):
                if i in visited:
                    continue

                row = similarity_rows[i - block_start]
                cluster_indices = []
                for j in range(i, old_size):
                    if j not in visited and row[j].item() >= similarity_threshold:
                        cluster_indices.append(j)
                        visited.add(j)

                if not cluster_indices:
                    continue

                if len(cluster_indices) == 1:
                    consolidated_entries.append(self.entries[i])
                    continue

                # Merge cluster
                clustered_entries = [self.entries[idx] for idx in cluster_indices]
            
                # Key/Value is average
                merged_key = torch.stack([e.key for e in clustered_entries]).mean(dim=0)
                merged_value = torch.stack([e.value for e in clustered_entries]).mean(dim=0)
            
                # Topic is the most common non-generic topic, breaking ties by input order
                topics = [e.topic for e in clustered_entries if e.topic and e.topic.lower() != "general"]
                if topics:
                    counts = {}
                    for t in topics:
                        counts[t] = counts.get(t, 0) + 1
                    merged_topic = max(counts, key=counts.get)
                else:
                    merged_topic = clustered_entries[0].topic or "General"
                
                # Source: Concatenate unique text snippets or pick the longest one
                sources = []
                for e in clustered_entries:
                    if e.source and e.source not in sources:
                        sources.append(e.source)
                if sources:
                    # Pick the longest source snippet to retain detail
                    merged_source = max(sources, key=len)
                else:
                    merged_source = ""
                
                merged_created = min(e.created_at for e in clustered_entries)
                merged_access = sum(e.access_count for e in clustered_entries)

                merged_entry = CortexEntry(
                    key=merged_key,
                    value=merged_value,
                    topic=merged_topic,
                    source=merged_source,
                    created_at=merged_created,
                    access_count=merged_access,
                )
                consolidated_entries.append(merged_entry)

        after_merge_size = len(consolidated_entries)
        merged_count = old_size - after_merge_size

        # 3. Enforce max capacity (evict LFU if size exceeds capacity)
        evicted_count = 0
        if after_merge_size > max_capacity:
            # Sort by access_count descending, keep top max_capacity
            consolidated_entries.sort(key=lambda e: e.access_count, reverse=True)
            evicted_count = after_merge_size - max_capacity
            consolidated_entries = consolidated_entries[:max_capacity]

        # 4. Active Write-Back (Fact consolidation into model weights)
        consolidated_to_weights = 0
        if core is not None and hasattr(core, "model"):
            high_freq_entries = [e for e in consolidated_entries if e.access_count >= 5 and e.source]
            if high_freq_entries:
                model = core.model
                # Temporary optimizer for local fine-tuning
                opt = torch.optim.SGD(model.parameters(), lr=1e-3)
                loss_fn = torch.nn.CrossEntropyLoss()
                
                self._is_sleeping = True
                try:
                    was_training = model.training
                    model.train()
                    for entry in high_freq_entries:
                        try:
                            token_ids = core.encode(entry.source, allow_growth=False)
                            if len(token_ids) > 1:
                                input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long, device=model.embedding.weight.device)
                                target_ids = torch.tensor([token_ids[1:]], dtype=torch.long, device=model.embedding.weight.device)
                                
                                # 3 steps of local weight updates
                                for _ in range(3):
                                    opt.zero_grad()
                                    logits, balance_loss = model(input_ids)
                                    loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                                    total_loss = loss + balance_loss * 0.05
                                    total_loss.backward()
                                    opt.step()
                                    
                                # Decay access count since it's consolidated
                                entry.access_count = max(0, entry.access_count - 5)
                                consolidated_to_weights += 1
                        except Exception as ex:
                            logger.warning("Failed to consolidate fact '%s' to weights: %s", entry.source[:30], ex)
                finally:
                    opt.zero_grad(set_to_none=True)
                    self._is_sleeping = False
                    if not was_training:
                        model.eval()

        self.entries = consolidated_entries
        self._invalidate_cache()
        logger.info(
            "Cortex sleep consolidation complete: before=%d, after=%d, merged=%d, evicted=%d, active_writeback=%d",
            old_size, self.size, merged_count, evicted_count, consolidated_to_weights
        )
        return {
            "before": old_size,
            "after": self.size,
            "merged": merged_count,
            "evicted": evicted_count,
            "active_writeback": consolidated_to_weights,
        }

    def save(self, path: str | _Path) -> None:
        """Save cortex entries to disk as a .pt checkpoint."""
        path = _Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [
                {
                    "key": e.key,
                    "value": e.value,
                    "topic": e.topic,
                    "topics": e.topics,
                    "source": e.source,
                    "created_at": e.created_at,
                    "access_count": e.access_count,
                }
                for e in self.entries
            ],
            "config": {
                "dim": self.config.dim,
                "max_entries": self.config.max_entries,
            },
        }
        torch.save(data, path)
        logger.info(f"[Cortex] Saved {len(self.entries)} entries to {path}")

    @classmethod
    def load_from_disk(cls, path: str | _Path, config: CortexConfig) -> 'MemoryCortex':
        """Load a cortex from a saved .pt checkpoint."""
        path = _Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Cortex checkpoint not found: {path}")
        cortex = cls(config)
        # Cortex files contain tensors and plain data only; never unpickle code.
        data = torch.load(path, map_location="cpu", weights_only=True)
        for e in data.get("entries", []):
            entry = CortexEntry(
                key=e["key"],
                value=e["value"],
                topic=e.get("topic", ""),
                topics=e.get("topics", []),
                source=e.get("source", ""),
                created_at=e.get("created_at", 0.0),
                access_count=e.get("access_count", 0),
            )
            cortex.entries.append(entry)
        cortex._invalidate_cache()
        logger.info(f"[Cortex] Loaded {len(cortex.entries)} entries from {path}")
        return cortex

    @classmethod
    def load(cls, path: "str | _Path", config: "CortexConfig") -> "MemoryCortex":
        """Load cortex from path.
        
        Handles:
          - path/cortex.pt  -> load from .pt file
          - path/           -> directory: look for cortex.pt inside, else empty cortex
        """
        path = _Path(path)
        # If a .pt file, load directly
        if path.suffix == ".pt" and path.is_file():
            return cls.load_from_disk(path, config)
        # If a directory, look for cortex.pt inside it
        if path.is_dir():
            pt_file = path / "cortex.pt"
            if pt_file.exists():
                return cls.load_from_disk(pt_file, config)
            # Legacy: empty cortex (directory exists but no .pt inside)
            logger.info("[Cortex] No cortex.pt found in %s; starting with empty cortex.", path)
            return cls(config)
        # Fallback: try load_from_disk directly (handles .pt path)
        return cls.load_from_disk(path, config)


class CortexAutoStore:
    """Auto-store intermediate representations during training."""

    def __init__(self, data_dir: str | _Path = "assets/cortex"):
        self.data_dir = _Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self):
        store_file = self.data_dir / "cortex_store.json"
        if store_file.exists():
            try:
                self._store = json.loads(store_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
                logger.warning("CortexAutoStore: corrupt store file %s - starting fresh (%s)", store_file, exc)
                self._store = {}

    def _save(self):
        store_file = self.data_dir / "cortex_store.json"
        store_file.write_text(json.dumps(self._store, indent=2), encoding="utf-8")

    def auto_store(self, layer_name: str, representations: dict[str, Any], step: int):
        key = f"{layer_name}_step_{step}"
        self._store[key] = {
            "layer": layer_name,
            "step": step,
            "timestamp": time.time(),
            "representation_keys": list(representations.keys()),
            "representation_sizes": {k: len(v) if hasattr(v, "__len__") else 1 for k, v in representations.items()},
        }
        if len(self._store) > 100:
            oldest = sorted(self._store.keys(), key=lambda k: self._store[k].get("timestamp", 0))[:10]
            for key in oldest:
                del self._store[key]
        self._save()

    def retrieve(self, layer_name: str, step: int) -> dict[str, Any] | None:
        return self._store.get(f"{layer_name}_step_{step}")

    def get_stats(self) -> dict[str, Any]:
        return {"entries": len(self._store), "layers": list({v.get("layer") for v in self._store.values()})}
