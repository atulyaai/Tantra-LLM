"""
tantra/moe.py — Mixture of Experts. Contains: ExpertMeta, ExpertRegistry, MoERouter, LoadBalancer, LazyExpertLoader.
"""
from __future__ import annotations

import collections
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from Tantra.codec import DNACodec
from Tantra.config import ALRAConfig, MoEConfig, NeuroCoreConfig
from Tantra.model import ALRAAttention, NeuroCoreModel
from Tantra.utils import get_logger

log = get_logger(__name__)

# ── ExpertMeta & ExpertRegistry ───────────────────────────────────────────────

@dataclass
class ExpertMeta:
    """Metadata for one expert network."""
    expert_id: int
    name: str
    dna_path: str                        # path to compressed .dna weight file
    specialization: str = "general"      # learned domain e.g. 'code','math','language'
    param_count: int = 0
    compressed_size_bytes: int = 0
    usage_count: int = 0                 # how many tokens routed here (for load stats)
    last_used_step: int = 0


class ExpertRegistry:
    """
    Maintains metadata for all MoE expert networks.
    Persisted as a JSON registry file alongside the expert .dna files.
    """

    def __init__(self, expert_dir: str, num_experts: int) -> None:
        self._dir = expert_dir
        self._num_experts = num_experts
        self._registry: dict[int, ExpertMeta] = {}
        self._registry_path = os.path.join(expert_dir, "registry.json")
        os.makedirs(expert_dir, exist_ok=True)

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, meta: ExpertMeta) -> None:
        """Add or update an expert's metadata."""
        self._registry[meta.expert_id] = meta
        self._save()

    def register_new(
        self,
        expert_id: int,
        specialization: str = "general",
        param_count: int = 0,
    ) -> ExpertMeta:
        """Create and register a new expert with auto-generated path."""
        dna_path = os.path.join(self._dir, f"expert_{expert_id:04d}.dna")
        meta = ExpertMeta(
            expert_id=expert_id,
            name=f"expert_{expert_id:04d}",
            dna_path=dna_path,
            specialization=specialization,
            param_count=param_count,
        )
        self.register(meta)
        return meta

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, expert_id: int) -> Optional[ExpertMeta]:
        """Return expert metadata by ID, or None if not registered."""
        return self._registry.get(expert_id)

    def all_experts(self) -> list[ExpertMeta]:
        """Return all registered experts sorted by ID."""
        return sorted(self._registry.values(), key=lambda m: m.expert_id)

    def exists_on_disk(self, expert_id: int) -> bool:
        """True if the .dna file for this expert exists on disk."""
        meta = self.get(expert_id)
        if meta is None:
            return False
        path = os.path.join(self._dir, os.path.basename(meta.dna_path))
        return os.path.isfile(path)

    # ── Usage Tracking ────────────────────────────────────────────────────────

    def record_usage(self, expert_id: int, step: int) -> None:
        """Increment usage counter for load balancing analysis."""
        meta = self._registry.get(expert_id)
        if meta:
            meta.usage_count += 1
            meta.last_used_step = step

    def usage_stats(self) -> dict:
        """Return usage distribution across all experts."""
        total = sum(m.usage_count for m in self._registry.values()) or 1
        return {
            m.expert_id: {
                "count": m.usage_count,
                "fraction": m.usage_count / total,
                "specialization": m.specialization,
            }
            for m in self.all_experts()
        }

    def most_used(self, n: int = 10) -> list[ExpertMeta]:
        """Return top-n most-used experts."""
        return sorted(
            self._registry.values(),
            key=lambda m: m.usage_count,
            reverse=True,
        )[:n]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save registry to JSON file."""
        self._save()

    def _save(self) -> None:
        data = {}
        for k, v in self._registry.items():
            d = asdict(v)
            d["dna_path"] = os.path.basename(v.dna_path)
            data[str(k)] = d
        with open(self._registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load registry from JSON file (if it exists)."""
        if not os.path.isfile(self._registry_path):
            log.info("No expert registry found — starting fresh")
            return
        with open(self._registry_path) as f:
            data = json.load(f)
        self._registry = {
            int(k): ExpertMeta(**v) for k, v in data.items()
        }
        log.info(f"Loaded registry with {len(self._registry)} experts")

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"ExpertRegistry(experts={len(self._registry)}, dir={self._dir!r})"


# ── MoERouter ─────────────────────────────────────────────────────────────────

class MoERouter(nn.Module):
    """
    Selects which expert(s) to route each token to.
    Uses a small ALRA attention layer so routing is context-aware,
    not just based on the single token embedding.
    """

    def __init__(self, config: MoEConfig, embed_dim: int):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        num_heads = 8 if embed_dim % 8 == 0 else (4 if embed_dim % 4 == 0 else 1)
        head_dim = embed_dim // num_heads
        alra_cfg = ALRAConfig(
            dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.context_layer = ALRAAttention(alra_cfg)
        
        # Router projection
        self.router_weights = nn.Linear(embed_dim, self.num_experts, bias=False)

    def forward(
        self, x: Tensor, state: dict | None = None
    ) -> tuple[Tensor, Tensor, dict | None]:
        """
        x: [B, T, D]
        Returns:
            routing_weights: [B, T, top_k] probabilities
            selected_experts: [B, T, top_k] integer expert IDs
            new_state: updated ALRA state for generation
        """
        # Guard: empty input
        if x.numel() == 0:
            B = x.shape[0] if x.dim() >= 1 else 0
            T = x.shape[1] if x.dim() >= 2 else 0
            weights = torch.empty(B, T, self.top_k, device=x.device)
            experts = torch.empty(B, T, self.top_k, device=x.device, dtype=torch.long)
            return weights, experts, state

        # Get context-aware representation
        context_x, new_state = self.context_layer(x, state=state)
        
        # Compute logits for each expert
        logits = self.router_weights(context_x)  # [B, T, num_experts]
        
        # Top-k selection
        routing_weights, selected_experts = torch.topk(logits, self.top_k, dim=-1)
        
        # Normalize weights (softmax over top_k)
        routing_weights = torch.softmax(routing_weights, dim=-1)
        
        return routing_weights, selected_experts, new_state

    def load_balancing_loss(self, selected_experts: Tensor) -> Tensor:
        """
        Compute load balancing loss to prevent expert collapse.
        selected_experts: [B, T, top_k]
        """
        if selected_experts.numel() == 0:
            return torch.tensor(0.0, device=selected_experts.device)
        # Count fraction of tokens routed to each expert in this batch
        expert_counts = torch.bincount(
            selected_experts.flatten(), minlength=self.num_experts
        ).float()
        total = expert_counts.sum().clamp(min=1.0)
        expert_fraction = expert_counts / total
        
        # Ideal fraction is uniform
        ideal_fraction = 1.0 / self.num_experts
        
        # MSE loss vs ideal uniform distribution
        return torch.mean((expert_fraction - ideal_fraction) ** 2) * self.config.load_balance_coeff


# ── LoadBalancer ──────────────────────────────────────────────────────────────

class LoadBalancer(nn.Module):
    """
    Computes auxiliary loss to ensure all experts are utilized equally.
    Prevents the router from collapsing and always picking the same few experts.
    """
    def __init__(self, num_experts: int, coeff: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.coeff = coeff

    def forward(self, routing_probs: Tensor) -> Tensor:
        """
        routing_probs: [B, T, num_experts] probabilities from router before top-k.
        Returns scalar loss.
        """
        if routing_probs.numel() == 0 or self.coeff == 0.0:
            return torch.tensor(0.0, device=routing_probs.device)
            
        # Average probability routed to each expert across the batch/sequence
        # shape: [num_experts]
        mean_probs = routing_probs.mean(dim=(0, 1))
        
        # Ideal is uniform distribution: 1 / num_experts
        ideal = 1.0 / self.num_experts
        
        # Mean squared error
        loss = torch.mean((mean_probs - ideal) ** 2)
        
        return loss * self.coeff


# ── LazyExpertLoader ──────────────────────────────────────────────────────────

class LazyExpertLoader:
    """
    Manages loading experts from .dna files into RAM with an LRU cache.
    Prevents OOM by keeping only `cache_size` experts in RAM.
    """

    def __init__(
        self,
        moe_config: MoEConfig,
        model_config: NeuroCoreConfig,
        registry: ExpertRegistry,
        dna_codec: DNACodec,
    ) -> None:
        self._cfg = moe_config
        self._model_cfg = model_config
        self._registry = registry
        self._codec = dna_codec
        self._cache_size = moe_config.expert_cache_size
        
        # Thread-safe LRU cache
        self._cache: collections.OrderedDict[int, nn.Module] = collections.OrderedDict()
        self._lock = threading.Lock()

    def get_expert(self, expert_id: int, device: str = "cpu") -> nn.Module:
        """Get expert from RAM if cached, else load from disk."""
        with self._lock:
            if expert_id in self._cache:
                # Cache hit: move to end (most recently used)
                expert = self._cache.pop(expert_id)
                self._cache[expert_id] = expert
                return expert.to(device, non_blocking=True) if "cuda" in str(device) else expert.to(device)

        # Cache miss: load from disk
        return self._load_expert(expert_id, device)

    def _evict_lru_if_needed(self):
        """Evict the least recently used expert from the cache if we exceed cache size."""
        while len(self._cache) > self._cache_size:
            # popitem(last=False) removes the first item inserted (least recently used)
            self._cache.popitem(last=False)

    def _load_expert(self, expert_id: int, device: str = "cpu") -> nn.Module:
        """Load expert from disk, decompress, instantiate, and cache."""
        meta = self._registry.get(expert_id)
        if not meta:
            raise ValueError(f"Expert ID {expert_id} not found in registry.")

        log.debug(f"Cache miss: Loading expert {expert_id} from {meta.dna_path}")
        
        # 1. Instantiate empty expert architecture
        # An expert is just a NeuroCoreModel stack (without embedding/lm_head if shared)
        # For simplicity in this implementation, we assume full model per expert.
        expert = NeuroCoreModel(self._model_cfg)
        
        # 2. Decompress .dna into state_dict tensors
        if self._registry.exists_on_disk(expert_id):
            try:
                state_dict_flat = self._codec.decompress(meta.dna_path)
                model_sd = expert.state_dict()
                restored_sd = {}
                for k, param_tensor in model_sd.items():
                    if k in state_dict_flat:
                        flat_t = state_dict_flat[k]
                        if flat_t.shape == param_tensor.shape:
                            restored_sd[k] = flat_t.to(param_tensor.dtype)
                        elif flat_t.numel() == param_tensor.numel():
                            restored_sd[k] = flat_t.view(param_tensor.shape).to(param_tensor.dtype)
                if restored_sd:
                    expert.load_state_dict(restored_sd, strict=False)
                    log.info(f"Loaded {len(restored_sd)} decompressed weight tensors into Expert {expert_id}")
            except Exception as e:
                log.warning(f"Could not load decompressed DNA weights for Expert {expert_id} ({e}). Using default init.")
        else:
            log.warning(f"DNA file not found for expert {expert_id}, using default init.")

        # Hybrid GPU offload with pinned memory and CUDA streams
        if "cuda" in str(device) and torch.cuda.is_available():
            expert_cpu = expert.to("cpu")
            # Pin memory for faster CPU -> GPU transfer
            for param in expert_cpu.parameters():
                if param.data.is_contiguous():
                    param.data = param.data.pin_memory()
            
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                expert_gpu = expert_cpu.to(device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(stream)
            final_expert = expert_gpu
        else:
            final_expert = expert.to(device)

        # 3. Add to cache (thread-safe)
        with self._lock:
            self._cache[expert_id] = final_expert
            self._evict_lru_if_needed()
            
        return final_expert

    def save_expert(self, expert_id: int, expert: nn.Module) -> str:
        """Compress expert state_dict into .dna file on disk."""
        meta = self._registry.get(expert_id)
        if not meta:
            meta = self._registry.register_new(expert_id, specialization="general")

        dna_path = os.path.join(self._registry._dir, f"expert_{expert_id:04d}.dna")
        
        try:
            from Tantra.codec import MultimodalWeightFormatter, CompressionConfig
            formatter = MultimodalWeightFormatter(CompressionConfig(zstd_level=3))
            
            tensor_weights = {}
            for k, v in expert.state_dict().items():
                if isinstance(v, torch.Tensor):
                    if v.dim() == 1:
                        tensor_weights[k] = v.unsqueeze(0).float().cpu()
                    elif v.dim() == 2:
                        tensor_weights[k] = v.float().cpu()
                    else:
                        tensor_weights[k] = v.view(v.size(0), -1).float().cpu()
                        
            stats = formatter.format_weights(tensor_weights, dna_path)
            meta.compressed_size_bytes = os.path.getsize(dna_path)
            meta.dna_path = os.path.basename(dna_path)
            self._registry.register(meta)
            log.info(f"Compressed Expert {expert_id} -> {dna_path} ({meta.compressed_size_bytes / 1024:.1f} KB)")
        except Exception as e:
            log.warning(f"Could not compress Expert {expert_id} to DNA: {e}")

        return dna_path

    def save_all_active_experts(self) -> None:
        """Save all currently cached experts to DNA files during checkpointing."""
        with self._lock:
            for eid, expert in self._cache.items():
                self.save_expert(eid, expert)
