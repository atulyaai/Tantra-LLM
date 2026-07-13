"""Memory Cortex â€” external knowledge store for NP-DNA.

Stores knowledge as vectors.  The model queries the Cortex during inference
to retrieve relevant facts.  Adding knowledge = adding vectors (zero training).

This is how a 50M param model accesses unlimited knowledge:
  1M Cortex entries â‰ˆ 100B params of stored factual knowledge.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .config import CortexConfig

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

        # Projection layer: hidden_state â†’ query vector
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

        # Enforce max capacity â€” evict least-accessed entry
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
            (values, scores) â€” retrieved value vectors and similarity scores.
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
            encoder_fn: Callable that converts text â†’ tensor (dim,).
            topic: Topic label.

        Returns:
            Entry index.
        """
        with torch.no_grad():
            vec = encoder_fn(text)
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

    def save(self, path: str | Path) -> None:
        """Save Cortex to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save entries as tensor + metadata
        if self.entries:
            keys = torch.stack([e.key for e in self.entries])
            values = torch.stack([e.value for e in self.entries])
            torch.save({"keys": keys, "values": values}, path / "cortex_vectors.pt")

            meta = [
                {
                    "topic": e.topic,
                    "topics": e.topics,
                    "related": e.related,
                    "source": e.source,
                    "created_at": e.created_at,
                    "access_count": e.access_count,
                }
                for e in self.entries
            ]
            (path / "cortex_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

        # Save projection weights
        torch.save(
            {
                "query_proj": self.query_proj.state_dict(),
                "value_proj": self.value_proj.state_dict(),
            },
            path / "cortex_projections.pt",
        )
        logger.info("Cortex saved: %d entries to %s", self.size, path)

    @classmethod
    def load(cls, path: str | Path, config: CortexConfig) -> "MemoryCortex":
        """Load Cortex from disk."""
        path = Path(path)
        cortex = cls(config)

        proj_path = path / "cortex_projections.pt"
        if proj_path.exists():
            state = torch.load(proj_path, map_location="cpu", weights_only=True)
            cortex.query_proj.load_state_dict(state["query_proj"])
            cortex.value_proj.load_state_dict(state["value_proj"])

        vec_path = path / "cortex_vectors.pt"
        meta_path = path / "cortex_meta.json"
        if vec_path.exists() and meta_path.exists():
            vecs = torch.load(vec_path, map_location="cpu", weights_only=True)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            keys = vecs["keys"]
            values = vecs["values"]
            for i, m in enumerate(meta):
                entry = CortexEntry(
                    key=keys[i],
                    value=values[i],
                    topic=m.get("topic", ""),
                    topics=m.get("topics", []),
                    related=m.get("related", []),
                    source=m.get("source", ""),
                    created_at=m.get("created_at", 0.0),
                    access_count=m.get("access_count", 0),
                )
                cortex.entries.append(entry)
            cortex._invalidate_cache()

        logger.info("Cortex loaded: %d entries from %s", cortex.size, path)
        return cortex


class CortexAutoStore:
    """Auto-store intermediate representations during training."""

    def __init__(self, data_dir: str | Path = "assets/cortex"):
        self.data_dir = Path(data_dir)
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
