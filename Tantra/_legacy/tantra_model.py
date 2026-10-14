"""
Tantra Dense-MoA Model
=======================

Frozen base model + trainable LoRA adapters per expert.
All adapters always active (dense), weighted by router.
CPU-optimized: no sparse branching, single fused matmul.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── LoRA Adapter ────────────────────────────────────────────────────────────────

class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation module.
    Trains only low-rank matrices A (d x r) and B (r x d).
    Output: x + scale * (x @ A @ B)
    """
    
    def __init__(
        self,
        d_model: int,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        # LoRA matrices - shared across target modules for parameter efficiency
        n_targets = len(self.target_modules)
        self.lora_A = nn.Parameter(torch.randn(n_targets, d_model, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(n_targets, rank, d_model))
        
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, x: Tensor, module_idx: int) -> Tensor:
        """Apply LoRA for specific target module."""
        A = self.lora_A[module_idx]  # [D, r]
        B = self.lora_B[module_idx]  # [r, D]
        
        lora_out = torch.matmul(torch.matmul(x, A), B) * self.scale
        
        if self.dropout_layer:
            lora_out = self.dropout_layer(lora_out)
        
        return lora_out
    
    def num_parameters(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()
    
    def resize_rank(self, new_rank: int) -> None:
        """Dynamically resize LoRA rank (expand or shrink)."""
        old_rank = self.rank
        if new_rank == old_rank:
            return
        
        n_targets = len(self.target_modules)
        device = self.lora_A.device
        dtype = self.lora_A.dtype
        
        if new_rank > old_rank:
            new_A = torch.zeros(n_targets, self.d_model, new_rank, device=device, dtype=dtype)
            new_A[:, :, :old_rank] = self.lora_A.data
            new_A[:, :, old_rank:] = torch.randn(n_targets, self.d_model, new_rank - old_rank, device=device, dtype=dtype) * 0.01
            
            new_B = torch.zeros(n_targets, new_rank, self.d_model, device=device, dtype=dtype)
            new_B[:, :old_rank, :] = self.lora_B.data
        else:
            new_A = self.lora_A.data[:, :, :new_rank]
            new_B = self.lora_B.data[:, :new_rank, :]
        
        self.lora_A.data = new_A
        self.lora_B.data = new_B
        self.rank = new_rank
        self.scale = self.alpha / new_rank


# ── Fused LoRA Layer (CPU-optimized) ──────────────────────────────────────────

class FusedLoRALayer(nn.Module):
    """
    Applies ALL adapters in a single batched matmul.
    No Python loop, no branching - pure tensor ops for CPU speed.
    """
    
    def __init__(self, d_model: int, adapter_ranks: Dict[str, int], target_modules: List[str]):
        super().__init__()
        self.d_model = d_model
        self.adapter_names = list(adapter_ranks.keys())
        self.n_adapters = len(self.adapter_names)
        self.target_modules = target_modules
        self.n_targets = len(target_modules)
        
        max_rank = max(adapter_ranks.values())
        
        self.adapter_A = nn.Parameter(
            torch.randn(self.n_adapters, self.n_targets, d_model, max_rank) * 0.01
        )
        self.adapter_B = nn.Parameter(
            torch.zeros(self.n_adapters, self.n_targets, max_rank, d_model)
        )
        self.adapter_ranks = nn.Parameter(
            torch.tensor([adapter_ranks[name] for name in self.adapter_names]),
            requires_grad=False
        )
        self.adapter_scales = nn.Parameter(
            torch.tensor([1.0 / adapter_ranks[name] for name in self.adapter_names]),
            requires_grad=False
        )
        
        rank_mask_A = torch.zeros(self.n_adapters, self.n_targets, d_model, max_rank)
        rank_mask_B = torch.zeros(self.n_adapters, self.n_targets, max_rank, d_model)
        for i, rank in enumerate(adapter_ranks.values()):
            rank_mask_A[i, :, :, :rank] = 1.0
            rank_mask_B[i, :, :rank, :] = 1.0
        self.register_buffer("rank_mask_A", rank_mask_A)
        self.register_buffer("rank_mask_B", rank_mask_B)
    
    def forward(self, x: Tensor, router_weights: Tensor) -> Tensor:
        """
        x: [B, T, D] - hidden states from base model
        router_weights: [B, n_adapters] - softmax weights from router
        Returns: [B, T, D] - weighted sum of all adapter outputs
        """
        batch_size, T, D = x.shape
        
        # Apply rank masks
        A = self.adapter_A * self.rank_mask_A  # [n_adapters, n_targets, D, max_rank]
        B_mat = self.adapter_B * self.rank_mask_B  # [n_adapters, n_targets, max_rank, D]
        scales = self.adapter_scales.view(1, 1, self.n_adapters, 1)  # [1, 1, n_adapters, 1]
        
        outputs = []
        for t_idx in range(self.n_targets):
            A_t = A[:, t_idx]  # [n_adapters, D, max_rank]
            B_t = B_mat[:, t_idx]  # [n_adapters, max_rank, D]
            
            # x @ A_t: [B, T, D] @ [n_adapters, D, max_rank] = [B, T, n_adapters, max_rank]
            xA = torch.einsum('btd,adr->btar', x, A_t)  # [B, T, n_adapters, max_rank]
            # xA @ B_t: [B, T, n_adapters, max_rank] @ [n_adapters, max_rank, D] = [B, T, n_adapters, D]
            xAB = torch.einsum('btar,ard->btad', xA, B_t)  # [B, T, n_adapters, D]
            
            # Apply scale
            xAB = xAB * scales  # [B, T, n_adapters, D]
            outputs.append(xAB)
        
        # Sum over targets: [B, T, n_adapters, D]
        adapter_out = outputs[0]
        for o in outputs[1:]:
            adapter_out = adapter_out + o
        
        # Weight by router: [B, n_adapters] -> [B, 1, n_adapters, 1]
        router_w = router_weights.reshape(batch_size, 1, self.n_adapters, 1)
        
        # Weighted sum: [B, T, n_adapters, D] * [B, 1, n_adapters, 1] -> [B, T, D]
        expert_contribution = (adapter_out * router_w).sum(dim=2)
        
        return expert_contribution
    
    def get_adapter_params(self, adapter_name: str) -> Tuple[Tensor, Tensor]:
        """Extract A, B for a specific adapter (for saving/loading)."""
        idx = self.adapter_names.index(adapter_name)
        rank = self.adapter_ranks[idx].item()
        return self.adapter_A[idx, :, :, :rank], self.adapter_B[idx, :, :rank, :]
    
    def set_adapter_params(self, adapter_name: str, A: Tensor, B: Tensor) -> None:
        """Set A, B for a specific adapter."""
        idx = self.adapter_names.index(adapter_name)
        rank = A.shape[-1]
        self.adapter_A.data[idx, :, :, :rank] = A
        self.adapter_B.data[idx, :, :rank, :] = B
        self.adapter_ranks.data[idx] = rank
        self.adapter_scales.data[idx] = 1.0 / rank
        self.rank_mask_A.data[idx, :, :, :rank] = 1.0
        self.rank_mask_A.data[idx, :, :, rank:] = 0.0
        self.rank_mask_B.data[idx, :, :rank, :] = 1.0
        self.rank_mask_B.data[idx, :, rank:, :] = 0.0
    
    def resize_expert_rank(self, expert_name: str, new_rank: int) -> None:
        """Resize a specific expert's LoRA rank."""
        idx = self.adapter_names.index(expert_name)
        old_rank = self.adapter_ranks[idx].item()
        if new_rank == old_rank:
            return
        
        device = self.adapter_A.device
        dtype = self.adapter_A.dtype
        
        if new_rank > old_rank:
            self.adapter_A.data[idx, :, :, old_rank:new_rank] = torch.randn(
                self.n_targets, self.d_model, new_rank - old_rank, device=device, dtype=dtype) * 0.01
            self.adapter_B.data[idx, :, old_rank:new_rank, :] = 0.0
        else:
            self.adapter_A.data[idx, :, :, new_rank:] = 0.0
            self.adapter_B.data[idx, :, new_rank:, :] = 0.0
        
        self.adapter_ranks.data[idx] = new_rank
        self.adapter_scales.data[idx] = 1.0 / new_rank
        self.rank_mask_A.data[idx, :, :, :new_rank] = 1.0
        self.rank_mask_A.data[idx, :, :, new_rank:] = 0.0
        self.rank_mask_B.data[idx, :, :new_rank, :] = 1.0
        self.rank_mask_B.data[idx, :, new_rank:, :] = 0.0


# ── Hierarchical Router ────────────────────────────────────────────────────────

class HierarchicalRouter(nn.Module):
    """
    Two-level routing:
    Level 1: Domain router (8 domains)
    Level 2: Specialist router per domain (variable experts)
    
    Output: flat weights over all experts [B, n_total_experts]
    """
    
    def __init__(
        self,
        d_model: int,
        domain_config: Dict[str, int],
        hidden_dim: int = 256,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.domains = list(domain_config.keys())
        self.domain_experts = domain_config
        self.n_domains = len(self.domains)
        self.temperature = temperature
        
        # Domain router
        self.domain_router = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_domains),
        )
        
        # Specialist routers per domain
        self.specialist_routers = nn.ModuleDict()
        for domain, n_experts in domain_config.items():
            self.specialist_routers[domain] = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_experts),
            )
        
        # Build flat expert index mapping
        self.expert_to_domain = {}
        self.expert_to_local_idx = {}
        self.domain_offsets = {}
        offset = 0
        for domain in self.domains:
            self.domain_offsets[domain] = offset
            n = self.domain_experts[domain]
            for i in range(n):
                self.expert_to_domain[offset + i] = domain
                self.expert_to_local_idx[offset + i] = i
            offset += n
        self.n_total_experts = offset
    
    def forward(self, x: Tensor, return_analysis: bool = False) -> Tuple[Tensor, Optional[Dict]]:
        x_pooled = x.mean(dim=1)  # [B, D]
        B = x_pooled.shape[0]
        
        # Level 1: Domain routing
        domain_logits = self.domain_router(x_pooled) / self.temperature
        domain_weights = F.softmax(domain_logits, dim=-1)
        
        # Level 2: Specialist routing per domain
        expert_weights = torch.zeros(B, self.n_total_experts, device=x.device, dtype=x.dtype)
        
        for d_idx, domain in enumerate(self.domains):
            spec_logits = self.specialist_routers[domain](x_pooled) / self.temperature
            spec_weights = F.softmax(spec_logits, dim=-1)
            
            offset = self.domain_offsets[domain]
            n = self.domain_experts[domain]
            expert_weights[:, offset:offset+n] = (
                domain_weights[:, d_idx:d_idx+1] * spec_weights
            )
        
        if not return_analysis:
            return expert_weights, None
        
        # Analysis for interpretability
        top_k = expert_weights.topk(5, dim=-1)
        analysis = {
            'expert_weights': expert_weights.detach().cpu().numpy(),
            'domain_weights': domain_weights.detach().cpu().numpy(),
            'top_experts': [
                {self._expert_name(i): float(w) for i, w in zip(idx, val)}
                for idx, val in zip(top_k.indices, top_k.values)
            ],
            'entropy': -(expert_weights * expert_weights.clamp(min=1e-8).log()).sum(-1).mean().item(),
            'certainty': expert_weights.max(-1).values.mean().item(),
        }
        return expert_weights, analysis
    
    def _expert_name(self, flat_idx) -> str:
        idx = flat_idx.item() if hasattr(flat_idx, 'item') else flat_idx
        domain = self.expert_to_domain[idx]
        local = self.expert_to_local_idx[idx]
        return f"{domain}_{local}"


# ── Tantra Dense-MoA Model ────────────────────────────────────────────────────

@dataclass
class TantraConfig:
    """Configuration for Tantra Dense-MoA model."""
    
    base_model_path: str = ""
    base_dim: int = 1024
    base_layers: int = 24
    base_heads: int = 16
    vocab_size: int = 65536
    
    adapter_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    router_hidden_dim: int = 256
    router_temperature: float = 1.0
    
    taxonomy: Dict = field(default_factory=dict)
    
    freeze_base: bool = True
    adapter_lr: float = 1e-3
    router_lr: float = 5e-4
    
    quantization: str = "int8"
    fuse_adapters: bool = True
    
    domain_config: Dict[str, int] = field(default_factory=dict)
    
    @classmethod
    def from_taxonomy(cls, taxonomy_path: str, base_model_path: str = "") -> "TantraConfig":
        import yaml
        with open(taxonomy_path) as f:
            taxonomy = yaml.safe_load(f)
        
        domain_config = {}
        for domain, info in taxonomy.items():
            if domain in ["conversation", "knowledge", "reasoning", "safety", "tasks",
                          "agentic", "tool_calling", "structured_output", "rag", "code_agent",
                          "multilingual", "translation", "long_context"]:
                domain_config[domain] = info["count"]
        
        return cls(
            base_model_path=base_model_path,
            taxonomy=taxonomy,
            domain_config=domain_config,
        )


class _RMSNorm(nn.Module):
    """RMSNorm with learnable scale (matches checkpoint format: gamma, beta, w_scale)."""
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.w_scale = nn.Linear(dim, 1, bias=True)
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        return self.gamma * (x * rms) + self.beta


class _Attn(nn.Module):
    """Gated attention matching checkpoint format (w_q, w_k, w_v, w_o, w_gate)."""
    def __init__(self, d: int, heads: int = 16):
        super().__init__()
        self.heads = heads
        self.head_dim = d // heads
        self.w_q = nn.Linear(d, d)
        self.w_k = nn.Linear(d, d)
        self.w_v = nn.Linear(d, d)
        self.w_o = nn.Linear(d, d)
        self.w_gate = nn.Linear(d, heads)
    def forward(self, x, mask=None, state=None):
        B, T, D = x.shape
        q = self.w_q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            att = att + mask
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).contiguous()  # [B, T, heads, head_dim]
        # Per-head gate: [B, T, heads] -> [B, T, heads, 1]
        gate = torch.sigmoid(self.w_gate(x)).unsqueeze(-1)
        y = y * gate  # broadcast over head_dim
        y = y.view(B, T, D)
        return self.w_o(y), state


class _MLP(nn.Module):
    """SwiGLU MLP matching checkpoint format (w_up, w_gate, w_down)."""
    def __init__(self, d: int, hidden: int = 4096):
        super().__init__()
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class _BaseLayer(nn.Module):
    """Transformer layer: norm1 + attn + norm2 + mlp."""
    def __init__(self, d: int, heads: int = 16):
        super().__init__()
        self.norm1 = _RMSNorm(d)
        self.attn = _Attn(d, heads)
        self.norm2 = _RMSNorm(d)
        self.mlp = _MLP(d)
    def forward(self, x, mask=None, state=None):
        h, state = self.attn(self.norm1(x), mask=mask, state=state)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, state


class _LatentHeader(nn.Module):
    """Reasoning latent header (reasoning_norm, reasoning_proj, gate)."""
    def __init__(self, d: int):
        super().__init__()
        self.reasoning_norm = _RMSNorm(d)
        self.reasoning_proj = nn.Linear(d, d)
        self.gate = nn.Linear(d * 2, d)
    def forward(self, x):
        h = F.silu(self.reasoning_proj(self.reasoning_norm(x)))
        return self.gate(torch.cat([x, h], dim=-1))


class _BaseModuleWrapper(nn.Module):
    """
    Wraps base model weights loaded from legacy checkpoint.
    Exposes: embed, layers, final_norm, latent_header, output_proj, reasoning_depth.
    """
    def __init__(self, state_dict: Dict[str, Tensor]):
        super().__init__()
        # Infer dimensions from checkpoint
        d_model = state_dict["embed.weight"].shape[1]
        vocab_size = state_dict["embed.weight"].shape[0]
        output_dim = state_dict["output_proj.weight"].shape[0]

        # Find layer count
        layer_ids = {int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")}
        n_layers = max(layer_ids) + 1 if layer_ids else 0

        # Find attention heads from w_gate shape
        heads = 16  # default; infer from w_gate if possible
        attn_gate_key = "layers.0.attn.w_gate.weight"
        if attn_gate_key in state_dict:
            heads = state_dict[attn_gate_key].shape[0]

        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed.weight = nn.Parameter(state_dict["embed.weight"].clone())

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = _BaseLayer(d_model, heads)
            prefix = f"layers.{i}."
            own_sd = layer.state_dict()
            remapped = {}
            for k, v in own_sd.items():
                ck = prefix + k
                if ck in state_dict and state_dict[ck].shape == v.shape:
                    remapped[k] = state_dict[ck].clone()
            layer.load_state_dict(remapped, strict=False)
            self.layers.append(layer)

        self.final_norm = _RMSNorm(d_model)
        fn_keys = {"gamma": "final_norm.gamma", "beta": "final_norm.beta",
                   "w_scale.weight": "final_norm.w_scale.weight", "w_scale.bias": "final_norm.w_scale.bias"}
        fn_sd = {k: state_dict[v].clone() for k, v in fn_keys.items() if v in state_dict}
        self.final_norm.load_state_dict(fn_sd, strict=False)

        self.latent_header = _LatentHeader(d_model)
        lh_map = {}
        for k in state_dict:
            if k.startswith("latent_header."):
                lk = k[len("latent_header."):]
                lh_map[lk] = state_dict[k].clone()
        self.latent_header.load_state_dict(lh_map, strict=False)

        self.output_proj = nn.Linear(d_model, output_dim, bias=False)
        if "output_proj.weight" in state_dict:
            self.output_proj.weight = nn.Parameter(state_dict["output_proj.weight"].clone())

        self.reasoning_depth = 1 if any(k.startswith("latent_header.") for k in state_dict) else 0

        # Freeze
        for p in self.parameters():
            p.requires_grad = False


class TantraModel(nn.Module):
    """
    Tantra Dense-MoA Model.
    
    Architecture:
    - Frozen base transformer (loaded from checkpoint)
    - Fused LoRA adapters for all experts (always active, weighted)
    - Hierarchical router (domain -> specialist)
    
    Forward: base(x) + sum_i(router_weight_i * adapter_i(base_output))
    """
    
    def __init__(self, config: TantraConfig):
        super().__init__()
        self.config = config
        self.d_model = config.base_dim
        
        # Load base model (frozen)
        self.base_model = self._load_base_model(config.base_model_path)
        if config.freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False
        
        # Build adapter ranks from taxonomy
        adapter_ranks = {}
        for domain, info in config.taxonomy.items():
            if "experts" in info:
                for expert in info["experts"]:
                    adapter_ranks[expert] = info["rank"]
        
        # Fused LoRA layer
        self.adapters = FusedLoRALayer(
            d_model=self.d_model,
            adapter_ranks=adapter_ranks,
            target_modules=config.adapter_target_modules,
        )
        
        # Hierarchical router
        self.router = HierarchicalRouter(
            d_model=self.d_model,
            domain_config=config.domain_config,
            hidden_dim=config.router_hidden_dim,
            temperature=config.router_temperature,
        )
        
        # LM head (from base model)
        self.lm_head = self.base_model.output_proj
        
        # Expert names in order
        self.expert_names = list(adapter_ranks.keys())
        self.n_experts = len(self.expert_names)
        
        # Adapter health monitors
        self.register_buffer("adapter_usage", torch.zeros(self.n_experts))
        self.register_buffer("adapter_steps", torch.zeros(self.n_experts))
    
    def _load_base_model(self, path: str) -> nn.Module:
        """Load base Tantra model and freeze it."""
        import torch
        from Tantra.model.tantra_model import TantraConfig, TantraModel

        if path and os.path.exists(path):
            import Tantra.config as _cfg
            safe_globals = [getattr(_cfg, n) for n in dir(_cfg) if n.endswith("Config")]
            torch.serialization.add_safe_globals(safe_globals)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)

            # Build a lightweight wrapper exposing the checkpoint's modules directly
            base = _BaseModuleWrapper(state_dict)
            return base
        else:
            return TantraModel(TantraConfig())
    
    def forward(
        self,
        input_ids: Tensor,
        return_router_analysis: bool = False,
    ) -> Tuple[Tensor, Optional[Dict]]:
        """
        input_ids: [B, T]
        Returns: logits [B, T, vocab_size], router_analysis (optional)
        """
        # Base forward (frozen) - get hidden states before output projection
        with torch.no_grad() if self.config.freeze_base else torch.enable_grad():
            hidden = self._get_base_hidden(input_ids)
        
        # Router weights
        router_weights, analysis = self.router(hidden, return_analysis=return_router_analysis)
        
        # Fused adapter forward
        adapter_contribution = self.adapters(hidden, router_weights)
        
        # Combine
        final_hidden = hidden + adapter_contribution
        
        # LM head
        logits = self.lm_head(final_hidden)
        
        # Track usage
        if self.training:
            self.adapter_usage += router_weights.sum(dim=0).detach()
            self.adapter_steps += router_weights.shape[0]
        
        return logits, analysis
    
    def _get_base_hidden(self, input_ids: Tensor) -> Tensor:
        """Extract hidden states from base model before output projection."""
        x = self.base_model.embed(input_ids)
        
        for layer in self.base_model.layers:
            x, _ = layer(x, mask=None, state=None)
        
        x = self.base_model.final_norm(x)
        if self.base_model.reasoning_depth > 0:
            x = self.base_model.latent_header(x)
            x = self.base_model.final_norm(x)
        
        return x
    
    def get_expert_usage(self) -> Dict[str, float]:
        """Return average usage per expert."""
        usage = self.adapter_usage / self.adapter_steps.clamp(min=1)
        return {name: float(usage[i]) for i, name in enumerate(self.expert_names)}
    
    def resize_expert(self, expert_name: str, new_rank: int) -> None:
        """Resize a specific expert's LoRA rank."""
        self.adapters.resize_expert_rank(expert_name, new_rank)
    
    def add_expert(self, expert_name: str, rank: int) -> None:
        """Add a new expert (requires router expansion)."""
        pass


# ── Export / Quantization ──────────────────────────────────────────────────────

def export_Tantra_int8(model: TantraModel, output_path: str) -> None:
    """Export model with INT8 quantized base, FP16 adapters."""
    import copy
    
    export_model = copy.deepcopy(model)
    export_model.eval()
    
    from torch.quantization import quantize_dynamic
    export_model.base_model = quantize_dynamic(
        export_model.base_model,
        {nn.Linear},
        dtype=torch.qint8
    )
    
    if model.config.fuse_adapters:
        export_model.adapters.fuse_weights()
    
    torch.save({
        "model_state_dict": export_model.state_dict(),
        "config": model.config,
        "quantization": "int8_base_fp16_adapters",
        "expert_names": model.expert_names,
    }, output_path)


# ── Factory Function ──────────────────────────────────────────────────────────

def build_tantra_model(
    taxonomy_path: str,
    base_model_path: str = "",
    device: str = "cpu",
) -> TantraModel:
    """Build Tantra model from taxonomy and base checkpoint."""
    config = TantraConfig.from_taxonomy(taxonomy_path, base_model_path)
    model = TantraModel(config).to(device)
    return model