"""
tantra/model.py — NeuroCore neural architecture with Multi-Token Prediction (MTP).
Contains: DynamicScaleNorm, RotaryPositionalEncoding, ALRAAttention, SparseGatedProjection, NeuroCoreBlock, NeuroCoreModel.
"""

from typing import Optional, List, Dict, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from Tantra.config import NeuroCoreConfig, ALRAConfig, SGPConfig, NeuroCoreBlockConfig
from Tantra.utils import elu_plus_one, top_k_mask


# ── DynamicScaleNorm ──

class DynamicScaleNorm(nn.Module):
    """
    DSN: (x - mean) / std * sigmoid(W*x + b) * gamma + beta
    Learned scale adapts to input magnitude dynamically.
    Drop-in replacement for nn.LayerNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w_scale = nn.Linear(dim, 1, bias=True)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        scale = torch.sigmoid(self.w_scale(x))
        return x_norm * scale * self.gamma + self.beta


# ── RotaryPositionalEncoding ──

class RotaryPositionalEncoding:
    """Rotary embeddings: rotate Q and K based on position."""
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: int = 10000):
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._cache: Dict[Tuple[str, torch.dtype], Tuple[Tensor, Tensor]] = {}

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        seq_len = max(1, seq_len)
        key = (str(device), dtype)
        cached = self._cache.get(key)
        if cached is None or seq_len > cached[0].shape[0]:
            build_len = max(seq_len, 2048)
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
            t = torch.arange(build_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos_c = emb.cos().to(dtype)
            sin_c = emb.sin().to(dtype)
            self._cache[key] = (cos_c, sin_c)
            return cos_c[:seq_len], sin_c[:seq_len]
        return cached[0][:seq_len], cached[1][:seq_len]

    def _rotate_half(self, x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply(self, q: Tensor, k: Tensor, seq_len: int, offset: int = 0) -> Tuple[Tensor, Tensor]:
        cos, sin = self.get_cos_sin(offset + seq_len, q.device, q.dtype)
        cos = cos[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)

        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        return q_rotated, k_rotated


# ── ALRAAttention ──

class ALRAAttention(nn.Module):
    """
    Adaptive Linear Resonance Attention.
    O(n*d^2) complexity vs standard O(n^2*d).
    Uses learned forget gate for adaptive context window.
    """
    def __init__(self, config: ALRAConfig):
        super().__init__()
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.use_forget_gate = config.use_forget_gate
        self.eps = 1e-6

        assert self.dim == self.num_heads * self.head_dim, "dim must be num_heads * head_dim"

        self.w_q = nn.Linear(self.dim, self.dim)
        self.w_k = nn.Linear(self.dim, self.dim)
        self.w_v = nn.Linear(self.dim, self.dim)
        self.w_o = nn.Linear(self.dim, self.dim)
        
        if self.use_forget_gate:
            self.w_gate = nn.Linear(self.dim, self.num_heads)

        self.rope = RotaryPositionalEncoding(self.head_dim)
        
    def _apply_kernel(self, x: Tensor) -> Tensor:
        return elu_plus_one(x)

    def forward(
        self, 
        x: Tensor,
        mask: Optional[Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[Tensor, Optional[dict]]:
        B, T, D = x.shape
        
        Q = self.w_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        past_len = state.get("step", 0) if state is not None else 0
        Q, K = self.rope.apply(Q, K, T, offset=past_len)

        Q = self._apply_kernel(Q)
        K = self._apply_kernel(K)

        gates = None
        if self.use_forget_gate:
            gates = torch.sigmoid(self.w_gate(x)).transpose(1, 2)

        if state is not None or T == 1:
            if state is None:
                state = {}
            out, new_state = self._sequential_forward(Q, K, V, gates, state)
            if new_state is not None:
                new_state["step"] = past_len + T
        else:
            out = self._parallel_forward(Q, K, V, gates)
            new_state = None

        out = out.transpose(1, 2).reshape(B, T, self.dim)
        out = self.w_o(out)
        return out, new_state

    def _parallel_forward(self, Q: Tensor, K: Tensor, V: Tensor, gates: Optional[Tensor]) -> Tensor:
        """
        Chunked blockwise scan for linear O(1) memory complexity.
        For sequences <= 256 (e.g. pre-training), uses a fully vectorized O(T^2) causal matrix 
        to bypass the Python GIL and extreme loop overhead.
        """
        B, H, T, Dh = Q.shape
        
        if T <= 2048:
            # Fast vectorized causal path (O(1) memory graph overhead on GPU/CPU)
            if gates is not None:
                log_g = torch.log(gates.clamp(min=1e-6))
                cum_log_g = torch.cumsum(log_g, dim=-1)
                diff = cum_log_g.unsqueeze(-1) - cum_log_g.unsqueeze(-2)
                diff = diff.clamp(max=0.0, min=-50.0)
                mask = torch.tril(torch.ones(T, T, device=Q.device, dtype=torch.bool))
                D = torch.exp(diff) * mask.to(Q.dtype)
            else:
                D = torch.tril(torch.ones(T, T, device=Q.device, dtype=Q.dtype))
                
            attn = torch.matmul(Q, K.transpose(-2, -1))
            if D.dim() == 4:
                attn = attn * D
            else:
                attn = attn * D.unsqueeze(0).unsqueeze(0)
                
            num = torch.matmul(attn, V)
            den = attn.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            out = torch.nan_to_num(num / den, nan=0.0, posinf=1.0, neginf=-1.0)
            return out

        chunk_size = 256
        outs = []
        S = torch.zeros(B, H, Dh, Dh, device=Q.device, dtype=Q.dtype)
        z = torch.zeros(B, H, Dh, device=Q.device, dtype=Q.dtype)
        
        for c in range(0, T, chunk_size):
            end_c = min(c + chunk_size, T)
            Q_c = Q[:, :, c:end_c, :]
            K_c = K[:, :, c:end_c, :]
            V_c = V[:, :, c:end_c, :]
            gates_c = gates[:, :, c:end_c] if gates is not None else None
            
            for t_i in range(end_c - c):
                Q_t = Q_c[:, :, t_i]
                K_t = K_c[:, :, t_i]
                V_t = V_c[:, :, t_i]
                
                KV_t = K_t.unsqueeze(-1) * V_t.unsqueeze(-2)
                
                if gates_c is not None:
                    g_t = gates_c[:, :, t_i].unsqueeze(-1)
                    S = S * g_t.unsqueeze(-1) + KV_t
                    z = z * g_t + K_t
                else:
                    S = S + KV_t
                    z = z + K_t
                    
                num = torch.matmul(Q_t.unsqueeze(-2), S).squeeze(-2)
                den = (Q_t * z).sum(dim=-1, keepdim=True) + self.eps
                out_t = torch.nan_to_num(num / den, nan=0.0, posinf=1.0, neginf=-1.0)
                outs.append(out_t)
                
        out = torch.stack(outs, dim=2)
        return out

    def _sequential_forward(self, Q: Tensor, K: Tensor, V: Tensor, gates: Optional[Tensor], state: dict) -> Tuple[Tensor, dict]:
        S = state.get('S')
        z = state.get('z')
        B, H, _, Dh = Q.shape
        
        if S is None:
            S = torch.zeros(B, H, Dh, Dh, device=Q.device, dtype=Q.dtype)
            state['S'] = S
        if z is None:
            z = torch.zeros(B, H, Dh, device=Q.device, dtype=Q.dtype)
            state['z'] = z
            
        Q_t = Q.squeeze(2)
        K_t = K.squeeze(2)
        V_t = V.squeeze(2)
        
        KV_t = K_t.unsqueeze(-1) * V_t.unsqueeze(-2)
        
        if gates is not None:
            gate_t = gates.squeeze(2).unsqueeze(-1)
            S.mul_(gate_t.unsqueeze(-1)).add_(KV_t)
            z.mul_(gate_t).add_(K_t)
        else:
            S.add_(KV_t)
            z.add_(K_t)
            
        num = torch.matmul(Q_t.unsqueeze(2), S).squeeze(2)
        den = (Q_t * z).sum(dim=-1, keepdim=True) + self.eps
        out = torch.nan_to_num(num / den, nan=0.0, posinf=1.0, neginf=-1.0).unsqueeze(2)
        
        return out, state


class CausalSelfAttention(nn.Module):
    """Standard causal attention for controlled CPU comparisons with ALRA."""
    def __init__(self, config: ALRAConfig):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = config.dim, config.num_heads, config.head_dim
        self.w_q = nn.Linear(self.dim, self.dim)
        self.w_k = nn.Linear(self.dim, self.dim)
        self.w_v = nn.Linear(self.dim, self.dim)
        self.w_o = nn.Linear(self.dim, self.dim)
        self.rope = RotaryPositionalEncoding(self.head_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, state: Optional[dict] = None) -> Tuple[Tensor, Optional[dict]]:
        batch, tokens, _ = x.shape
        q = self.w_q(x).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if state is not None and "k" in state and state["k"] is not None:
            past_len = state["k"].shape[2]

        q, k = self.rope.apply(q, k, tokens, offset=past_len)

        if state is not None:
            if "k" in state and state["k"] is not None:
                k = torch.cat([state["k"], k], dim=2)
                v = torch.cat([state["v"], v], dim=2)
            state["k"] = k
            state["v"] = v

        if past_len == 0:
            is_causal = (tokens > 1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal)
        else:
            if tokens == 1:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            else:
                q_len = tokens
                kv_len = k.shape[2]
                causal_mask = torch.tril(torch.ones(q_len, kv_len, device=q.device, dtype=torch.bool), diagonal=kv_len - q_len)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, dropout_p=0.0)

        return self.w_o(out.transpose(1, 2).reshape(batch, tokens, self.dim)), state


# ── SparseGatedProjection ──

class SparseGatedProjection(nn.Module):
    """SGP: brain-inspired sparse FFN with top-k% active neurons."""
    def __init__(self, config: SGPConfig):
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = self.dim * config.expansion
        self.k = max(1, int(self.hidden_dim * config.sparsity))
        
        self.w_up = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w_down = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.w_gate = nn.Linear(self.dim, self.hidden_dim, bias=True)
        
        if config.activation == "silu":
            self.act = F.silu
        elif config.activation == "relu":
            self.act = F.relu
        else:
            self.act = F.gelu
            
        self._last_active_ratio = 0.0

    def forward(self, x: Tensor) -> Tensor:
        gates = torch.sigmoid(self.w_gate(x))
        mask = top_k_mask(gates, self.k)
        
        if self.training:
            mask_float = mask.to(x.dtype)
            mask_st = mask_float.detach() - gates.detach() + gates
            up = self.act(self.w_up(x))
            hidden = up * mask_st
        else:
            mask_float = mask.to(x.dtype)
            up = self.act(self.w_up(x))
            hidden = up * mask_float

            
        return self.w_down(hidden)

    def get_activation_stats(self) -> dict:
        return {"active_ratio": self._last_active_ratio, "target_ratio": self.k / self.hidden_dim}


class SwiGLUProjection(nn.Module):
    """Dense CPU-friendly gated MLP; avoids top-k sorting and masking overhead."""
    def __init__(self, config: SGPConfig):
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = self.dim * config.expansion
        self.w_up = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w_gate = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w_down = nn.Linear(self.hidden_dim, self.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Top1MoEProjection(nn.Module):
    """Actual top-1 MoE: selected token groups run through separate MLP experts."""
    def __init__(self, config: SGPConfig, num_experts: int, balance_coeff: float = 0.01):
        super().__init__()
        self.num_experts = max(2, num_experts)
        self.balance_coeff = balance_coeff
        self.router = nn.Linear(config.dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLUProjection(config) for _ in range(self.num_experts))
        self.last_aux_loss: Optional[Tensor] = None
        self.last_usage: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        router_logits = self.router(flat)
        probabilities = torch.softmax(router_logits, dim=-1)
        selected = probabilities.argmax(dim=-1)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            positions = selected == expert_id
            if positions.any():
                # Probability keeps a differentiable router path while the
                # discrete top-1 decision provides true conditional compute.
                output[positions] = expert(flat[positions]) * probabilities[positions, expert_id].unsqueeze(-1)
        mean_probability = probabilities.mean(dim=0)
        usage = torch.bincount(selected, minlength=self.num_experts).to(probabilities.dtype) / max(1, selected.numel())
        self.last_usage = usage.detach()
        self.last_aux_loss = self.balance_coeff * self.num_experts * torch.sum(mean_probability * usage)
        return output.reshape(original_shape)


# ── NeuroCoreBlock ──

class NeuroCoreBlock(nn.Module):
    """Full NeuroCore block: x -> DSN -> ALRA -> residual -> DSN -> SGP/MoE -> residual -> output."""
    def __init__(self, config: NeuroCoreBlockConfig, layer_idx: int, moe_config: Optional[Any] = None, use_moe: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.pre_norm = config.pre_norm
        self.use_moe = use_moe
        dim = config.alra.dim
        
        self.norm1 = DynamicScaleNorm(dim)
        self.attn = CausalSelfAttention(config.alra) if config.alra.attention_kind == "causal" else ALRAAttention(config.alra)
        self.norm2 = DynamicScaleNorm(dim)
        if use_moe and moe_config is not None and getattr(moe_config, "real_top1", False):
            self.mlp = Top1MoEProjection(config.sgp, moe_config.num_experts, moe_config.load_balance_coeff)
        elif config.sgp.implementation == "swiglu":
            self.mlp = SwiGLUProjection(config.sgp)
        else:
            self.mlp = SparseGatedProjection(config.sgp)

        if self.use_moe and moe_config is not None and not getattr(moe_config, "real_top1", False):
            from Tantra.moe import MoERouter
            self.router = MoERouter(moe_config, embed_dim=dim)
        else:
            self.router = None

    def forward(
        self, 
        x: Tensor,
        mask: Optional[Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[Tensor, Optional[dict]]:
        if self.pre_norm:
            norm_x = self.norm1(x)
            attn_out, new_state = self.attn(norm_x, mask=mask, state=state)
            x = x + attn_out
            
            norm_x2 = self.norm2(x)
            if self.router is not None:
                routing_weights, selected_experts, _ = self.router(norm_x2)
                mlp_out = self.mlp(norm_x2) * routing_weights.mean(dim=-1, keepdim=True)
            else:
                mlp_out = self.mlp(norm_x2)
            x = x + mlp_out
        else:
            attn_out, new_state = self.attn(x, mask=mask, state=state)
            x = self.norm1(x + attn_out)
            if self.router is not None:
                routing_weights, selected_experts, _ = self.router(x)
                mlp_out = self.mlp(x) * routing_weights.mean(dim=-1, keepdim=True)
            else:
                mlp_out = self.mlp(x)
            x = self.norm2(x + mlp_out)
            
        return x, new_state


# ── LatentCoTHeader ──

class LatentCoTHeader(nn.Module):
    """
    Latent Chain-of-Thought (CoT) Reasoning Header.
    Applies recurrent depth iterations on model hidden states to allow latent reasoning steps
    prior to final token prediction.
    """
    def __init__(self, dim: int, reasoning_depth: int = 3):
        super().__init__()
        self.dim = dim
        self.reasoning_depth = reasoning_depth
        self.reasoning_norm = DynamicScaleNorm(dim)
        self.reasoning_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, x: Tensor) -> Tensor:
        state = x
        for _ in range(self.reasoning_depth):
            normed = self.reasoning_norm(state)
            delta = F.silu(self.reasoning_proj(normed))
            g = torch.sigmoid(self.gate(torch.cat([state, delta], dim=-1)))
            state = torch.clamp(state + g * delta, min=-100.0, max=100.0)
        return state


# ── NeuroCoreModel with Multi-Token Prediction (MTP) & Latent CoT ──

class NeuroCoreModel(nn.Module):
    """Full NeuroCore language model with Multi-Token Prediction (MTP) heads and Latent Reasoning Headers."""

    def __init__(self, config: NeuroCoreConfig, use_mtp: bool = True, reasoning_depth: int = 3, use_moe: bool = False):
        super().__init__()
        self.config = config
        self.dim = config.block.alra.dim
        self.vocab_size = config.vocab.vocab_size
        self.use_mtp = use_mtp
        self.reasoning_depth = reasoning_depth
        self.use_moe = use_moe

        self.embed = nn.Embedding(self.vocab_size, self.dim)
        nn.init.normal_(self.embed.weight, std=0.02)

        self.layers = nn.ModuleList([
            NeuroCoreBlock(
                config.block,
                layer_idx=i,
                moe_config=config.moe if (use_moe or getattr(config.moe, "num_experts", 1) > 1) else None,
                use_moe=use_moe or (getattr(config.moe, "num_experts", 1) > 1 and i % 2 == 1)
            )
            for i in range(config.block.num_layers)
        ])

        self.final_norm = DynamicScaleNorm(self.dim)
        self.latent_header = LatentCoTHeader(self.dim, reasoning_depth=reasoning_depth)
        
        # Primary head (predicts t+1)
        self.output_proj = nn.Linear(self.dim, self.vocab_size, bias=False)
        self.output_proj.weight = self.embed.weight

        # Auxiliary MTP head (predicts t+2 for DeepSeek-style Multi-Token Prediction)
        if self.use_mtp:
            self.mtp_head = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.shared_multimodal_weights: Dict[str, torch.Tensor] = {}

        # ── Dedicated specialist layers (one fixed layer per category) ──
        # Unlike residual adapters that touch every block, each category owns
        # ONE transformer layer that runs once past the shared base stack. Only
        # the routed category's layer executes, keeping per-request compute at
        # base + 1 layer. Each layer's output is gate-interpolated with the base
        # residual (zero gate = identity), so a fresh category is an exact
        # pass-through and never perturbs the base until it is trained.
        self.category_layers = nn.ModuleDict()
        # One scalar residual gate per specialist layer (parameter list mirrors
        # the per-category stack length). Gates are zero-initialised so a
        # freshly installed category is a literal identity pass-through: it
        # does not perturb the base until its dataset actually trains it.
        self.category_gates = nn.ModuleDict()
        self.active_category: Optional[str] = None

    def add_category_layers(self, categories: List[str], depth: int = 1, clone_layer_index: int = -1) -> None:
        """Add a stack of dedicated specialist layers per category, cloned from a base block.

        ``depth`` is the initial capacity (1 = one fixed layer). The stack can
        later grow (harder categories) or shrink (idle/over-provisioned) without
        changing any tensor shapes, so checkpoints stay compatible.
        Each layer carries a zero-initialised residual gate, so an untrained
        category is an exact identity (output equals the base) until its
        dataset trains it — see :meth:`forward` for the gate interpolation.
        """
        moe_config = self.config.moe if (self.use_moe or getattr(self.config.moe, "num_experts", 1) > 1) else None
        for category in categories:
            if not category or category in self.category_layers:
                continue
            if not category.replace("_", "").isalnum():
                raise ValueError(f"Invalid category layer name: {category!r}")
            stack = nn.ModuleList()
            for _ in range(max(1, depth)):
                layer = NeuroCoreBlock(
                    self.config.block,
                    layer_idx=clone_layer_index if clone_layer_index >= 0 else len(self.layers),
                    moe_config=moe_config,
                    use_moe=self.use_moe,
                )
                if 0 <= clone_layer_index < len(self.layers):
                    with torch.no_grad():
                        layer.load_state_dict(self.layers[clone_layer_index].state_dict())
                stack.append(layer)
            self.category_layers[category] = stack
            self.category_gates[category] = nn.ParameterList(
                [nn.Parameter(torch.zeros((), dtype=torch.get_default_dtype()), requires_grad=True)
                 for _ in range(len(stack))]
            )

    def grow_category(self, category: str, cap: int) -> bool:
        """Append one more specialist layer to a category (up to ``cap`` total)."""
        import copy
        if category not in self.category_layers:
            return False
        stack = self.category_layers[category]
        if len(stack) >= cap:
            return False
        source = stack[-1]
        new_layer = copy.deepcopy(source)
        with torch.no_grad():
            for p in new_layer.parameters():
                p.data.add_(torch.randn_like(p.data) * 0.001)
        stack.append(new_layer)
        self.category_gates[category].append(
            nn.Parameter(torch.zeros((), dtype=torch.get_default_dtype()), requires_grad=True)
        )
        return True

    def shrink_category(self, category: str, floor: int = 1) -> bool:
        """Remove one specialist layer from a category (down to ``floor`` total)."""
        if category not in self.category_layers:
            return False
        stack = self.category_layers[category]
        if len(stack) <= floor:
            return False
        stack.pop(len(stack) - 1)
        gates = list(self.category_gates[category])
        self.category_gates[category] = nn.ParameterList(gates[:-1])
        return True

    def category_depth(self, category: str) -> int:
        if category not in self.category_layers:
            return 0
        return len(self.category_layers[category])

    def sync_category_gates_from_checkpoint(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Preserve behaviour of trained categories from older checkpoints.

        Checkpoints written before residual gates existed contain
        ``category_layers.<name>.*`` but no ``category_gates.<name>.*``.  With
        gates defaulting to 0 those trained categories would silently become
        no-ops, so we open the gates (1.0 = full block output, matching the
        legacy behaviour) for any installed category that has layer weights in
        the checkpoint but no gate tensors.
        """
        import re
        layer_keys = {re.match(r"category_layers\.([^.]+)\.", k).group(1)
                      for k in state_dict if re.match(r"category_layers\.([^.]+)\.", k)}
        gated = {re.match(r"category_gates\.([^.]+)\.", k).group(1)
                 for k in state_dict if re.match(r"category_gates\.([^.]+)\.", k)}
        with torch.no_grad():
            for name in layer_keys - gated:
                gates = self.category_gates.get(name)
                if gates is not None:
                    for gate in gates:
                        gate.fill_(1.0)


    def freeze_for_category(self, category: str) -> None:
        """Freeze the shared base and every other category; train one specialist layer only."""
        if category not in self.category_layers:
            raise KeyError(f"Category layer {category!r} is not installed")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.category_layers[category].parameters():
            parameter.requires_grad_(True)
        for parameter in self.category_gates[category]:
            parameter.requires_grad_(True)
        self.active_category = category

    def get_multimodal_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract text, audio, image, and video weight slices from unified embedding space
        or return bound shared multimodal weights.
        """
        vocab_cfg = self.config.vocab
        w = self.embed.weight.detach()
        weights = {
            "text": w[vocab_cfg.text_range_start : vocab_cfg.text_range_end + 1].clone(),
            "audio": w[vocab_cfg.audio_range_start : vocab_cfg.audio_range_end + 1].clone(),
            "image": w[vocab_cfg.image_range_start : vocab_cfg.image_range_end + 1].clone(),
            "video": w[vocab_cfg.video_range_start : vocab_cfg.video_range_end + 1].clone(),
        }
        weights.update(self.shared_multimodal_weights)
        return weights

    def get_aux_loss(self) -> Tensor:
        """Aggregate real-MoE router balancing losses for training."""
        losses = [
            layer.mlp.last_aux_loss
            for layer in self.layers
            if isinstance(getattr(layer, "mlp", None), Top1MoEProjection)
            and layer.mlp.last_aux_loss is not None
        ]
        if not losses:
            return torch.zeros((), device=self.embed.weight.device)
        return torch.stack(losses).sum()

    def bind_multimodal_weights(self, weights_dict: Dict[str, torch.Tensor]) -> None:
        """
        Bind/share text, audio, image, and video weight matrices across unified model embedding space.
        """
        vocab_cfg = self.config.vocab
        with torch.no_grad():
            for mod, tensor in weights_dict.items():
                self.shared_multimodal_weights[mod] = tensor
                if mod == "text":
                    sl = slice(vocab_cfg.text_range_start, vocab_cfg.text_range_end + 1)
                    if tensor.shape[0] == (vocab_cfg.text_range_end - vocab_cfg.text_range_start + 1) and tensor.shape[1] == self.dim:
                        self.embed.weight[sl].copy_(tensor)
                elif mod == "audio":
                    sl = slice(vocab_cfg.audio_range_start, vocab_cfg.audio_range_end + 1)
                    if tensor.shape[0] == (vocab_cfg.audio_range_end - vocab_cfg.audio_range_start + 1) and tensor.shape[1] == self.dim:
                        self.embed.weight[sl].copy_(tensor)
                elif mod == "image":
                    sl = slice(vocab_cfg.image_range_start, vocab_cfg.image_range_end + 1)
                    if tensor.shape[0] == (vocab_cfg.image_range_end - vocab_cfg.image_range_start + 1) and tensor.shape[1] == self.dim:
                        self.embed.weight[sl].copy_(tensor)
                elif mod == "video":
                    sl = slice(vocab_cfg.video_range_start, vocab_cfg.video_range_end + 1)
                    if tensor.shape[0] == (vocab_cfg.video_range_end - vocab_cfg.video_range_start + 1) and tensor.shape[1] == self.dim:
                        self.embed.weight[sl].copy_(tensor)

    def export_multimodal_weights(self, formatter: Any, output_path: str, dict_data: Optional[bytes] = None) -> Any:
        """Export model multimodal weight space into encrypted DNA-AI representation format."""
        weights = self.get_multimodal_weights()
        return formatter.format_weights(weights, output_path, dict_data=dict_data)

    def load_multimodal_weights(self, formatter: Any, input_path: str) -> Dict[str, torch.Tensor]:
        """Load and bind multimodal weight space from encrypted DNA-AI file using formatter."""
        weights = formatter.parse_weights(input_path)
        self.bind_multimodal_weights(weights)
        return weights

    def forward(
        self,
        token_ids: Tensor,
        mask: Optional[Tensor] = None,
        states: Optional[List[dict]] = None,
        return_mtp: bool = False,
        use_latent_reasoning: bool = True,
        adapter_name: Optional[str] = None,
    ) -> Union[Tuple[Tensor, Optional[List[dict]]], Tuple[Tuple[Tensor, Tensor], Optional[List[dict]]]]:
        x = self.embed(token_ids)
        adapter_name = adapter_name or self.active_category
        if adapter_name is not None and adapter_name not in self.category_layers:
            raise KeyError(f"Adapter category {adapter_name!r} is not installed")
        new_states = [] if states is not None else None

        for i, layer in enumerate(self.layers):
            layer_state = states[i] if states is not None else None
            x, new_layer_state = layer(x, mask=mask, state=layer_state)
            if new_states is not None:
                new_states.append(new_layer_state)

        # Dedicated specialist layer(s): the routed category runs its stack of
        # fixed layers past the shared base stack. Each layer's transform is
        # gated: out = h + gate * (block(h) - h), with the gate zero-initialised,
        # so an untrained category contributes nothing (identical to base).
        # Only once a category's dataset trains it does the gate open and the
        # specialist block's behaviour become visible. This restores the
        # "untrained category must not perturb the base" guarantee while keeping
        # the cloned-block architecture (base compute is untouched).
        #
        # IMPORTANT: state=None here used to be hardcoded, so every token
        # generated through a routed category layer ran ALRAAttention's
        # stateless parallel_forward path in isolation -- no memory of any
        # prior token, even within the same generation, unlike the base
        # layers just above. Confirmed live: identical prompts produce
        # different next-token logits when run one-token-at-a-time (as
        # generate()/generate_stream() do) vs. all-at-once, specifically at
        # the category-layer stage. States for category layers are appended
        # after the base layers' states, so the list only grows when a
        # category is actually routed (base-only generation is unaffected).
        if adapter_name is not None and adapter_name in self.category_layers:
            stack = self.category_layers[adapter_name]
            gates = self.category_gates[adapter_name] if adapter_name in self.category_gates else None
            base_len = len(self.layers)
            for j, block in enumerate(stack):
                cat_state = None
                if states is not None and len(states) > base_len + j:
                    cat_state = states[base_len + j]
                h, new_cat_state = block(x, mask=mask, state=cat_state)
                if gates is not None and j < len(gates):
                    gate = gates[j]
                    x = x + gate * (h - x)
                else:
                    x = h
                if new_states is not None:
                    new_states.append(new_cat_state)

        x = self.final_norm(x)
        if use_latent_reasoning:
            x = self.latent_header(x)

        logits_main = self.output_proj(x)

        if return_mtp and self.use_mtp:
            logits_mtp = self.mtp_head(x)
            return (logits_main, logits_mtp), new_states

        return logits_main, new_states

    @torch.no_grad()  # NOT inference_mode — that poisons RoPE cache for subsequent training
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 150,
        temperature: float = 0.35,          # Lower temp = confident, coherent, non-hallucinating
        top_p: float = 0.85,               # Narrow nucleus = high quality vocab
        repetition_penalty: float = 1.25,  # Strong anti-loop
        use_mtp_speculation: bool = True,
        use_latent_reasoning: bool = True,
        eos_token_id: Optional[int] = 2,
        min_new_tokens: int = 1,
        adapter_name: Optional[str] = None,
        banned_token_ids: Optional[List[int]] = None,
    ) -> Tensor:
        """Generate text using Multi-Token Prediction (MTP) and Latent CoT reasoning.

        `adapter_name`: explicit per-call category routing. Preferred over
        setting self.active_category before calling -- that's shared
        mutable state on the model instance, which races if two requests
        generate concurrently (e.g. an async webui) and one sets a
        different category mid-flight. Defaults to None, which falls back
        to self.active_category via forward()'s own
        fallback, for compatibility with existing single-request callers.
        """
        self.eval()
        if prompt_ids.numel() == 0 or prompt_ids.size(1) == 0:
            prompt_ids = torch.tensor([[1]], device=prompt_ids.device, dtype=torch.long)
        B, T = prompt_ids.shape
        states = [{} for _ in range(len(self.layers))]

        for t in range(T):
            token = prompt_ids[:, t:t+1]
            # return_mtp=False: skip computing MTP head during prefill/decode
            # (logits_mtp is never used in generation, only in training loss)
            logits_t, states = self.forward(token, states=states, return_mtp=False, use_latent_reasoning=use_latent_reasoning, adapter_name=adapter_name)

        if isinstance(logits_t, tuple):
            logits_t = logits_t[0]

        next_token_logits = logits_t[:, -1, :]
        generated_ids = []
        all_seen_tokens = [row.tolist() for row in prompt_ids]

        if banned_token_ids is None:
            # Mask known pretrain DNA artifact token IDs
            banned_token_ids = [28344, 23214, 12932, 13142, 19409]

        for _ in range(max_new_tokens):
            next_token_logits = torch.nan_to_num(next_token_logits, nan=-1e9, posinf=1e4, neginf=-1e9)

            if banned_token_ids:
                for b_id in banned_token_ids:
                    if 0 <= b_id < next_token_logits.size(-1):
                        next_token_logits[:, b_id] = -1e9

            # Apply repetition penalty to seen tokens
            if repetition_penalty != 1.0:
                for batch_idx, seen_tokens in enumerate(all_seen_tokens):
                    for tok_id in set(seen_tokens):
                        if 0 <= tok_id < next_token_logits.size(-1):
                            val = next_token_logits[batch_idx, tok_id].item()
                            if val < 0:
                                next_token_logits[batch_idx, tok_id] = val * repetition_penalty
                            else:
                                next_token_logits[batch_idx, tok_id] = val / repetition_penalty
            if temperature > 0:
                scaled_logits = next_token_logits / max(temperature, 1e-5)
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                sorted_logits[sorted_indices_to_remove] = float('-inf')
                probs = torch.nan_to_num(torch.softmax(sorted_logits, dim=-1), nan=0.0)

                zero_rows = probs.sum(dim=-1) == 0
                if zero_rows.any():
                    probs[zero_rows, 0] = 1.0
                next_sorted_idx = torch.multinomial(probs, num_samples=1)
                next_token = sorted_indices.gather(1, next_sorted_idx)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_ids.append(next_token)
            for batch_idx in range(B):
                all_seen_tokens[batch_idx].append(next_token[batch_idx, 0].item())
            # Don't stop on a single EOS token while below the minimum generation
            # length. An early-stage / lightly-trained model emits </s> (id 2) as
            # its next-token majority class almost immediately, which used to
            # truncate every response after 1-2 tokens (the "token count dropped"
            # bug). Require min_new_tokens before an EOS break is honoured.
            n_generated = len(generated_ids)
            if n_generated >= min_new_tokens and eos_token_id is not None:
                if isinstance(eos_token_id, (list, tuple, set)):
                    if any((next_token == eid).all() for eid in eos_token_id):
                        break
                elif (next_token == eos_token_id).all():
                    break
            logits_t, states = self.forward(next_token, states=states, return_mtp=False, use_latent_reasoning=use_latent_reasoning, adapter_name=adapter_name)

            if isinstance(logits_t, tuple):
                logits_t = logits_t[0]

            next_token_logits = logits_t[:, -1, :]

        return torch.cat([prompt_ids] + generated_ids, dim=1)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 150,
        temperature: float = 0.35,
        top_p: float = 0.85,
        repetition_penalty: float = 1.25,
        use_latent_reasoning: bool = True,
        eos_token_id: Optional[int] = 2,
        min_new_tokens: int = 1,
        adapter_name: Optional[str] = None,
        banned_token_ids: Optional[List[int]] = None,
    ):
        """Yield sampled tokens one at a time without buffering a response."""
        self.eval()
        if prompt_ids.numel() == 0 or prompt_ids.size(1) == 0:
            prompt_ids = torch.tensor([[1]], device=prompt_ids.device, dtype=torch.long)
        B, T = prompt_ids.shape
        states = [{} for _ in range(len(self.layers))]

        if banned_token_ids is None:
            banned_token_ids = [28344, 23214, 12932, 13142, 19409]

        for t in range(T):
            logits_t, states = self.forward(
                prompt_ids[:, t:t + 1], states=states,
                return_mtp=False, use_latent_reasoning=use_latent_reasoning,
                adapter_name=adapter_name,
            )

        next_token_logits = logits_t[:, -1, :]
        all_seen_tokens = [row.tolist() for row in prompt_ids]
        eos_ids = set(eos_token_id) if isinstance(eos_token_id, (list, tuple, set)) else {eos_token_id}
        eos_ids.discard(None)
        n_yielded = 0

        for _ in range(max_new_tokens):
            logits = torch.nan_to_num(next_token_logits.clone(), nan=-1e9, posinf=1e4, neginf=-1e9)
            if banned_token_ids:
                for b_id in banned_token_ids:
                    if 0 <= b_id < logits.size(-1):
                        logits[:, b_id] = -1e9
            if repetition_penalty != 1.0:
                for batch_idx, seen_tokens in enumerate(all_seen_tokens):
                    for tok_id in set(seen_tokens):
                        if 0 <= tok_id < logits.size(-1):
                            value = logits[batch_idx, tok_id].item()
                            logits[batch_idx, tok_id] = value * repetition_penalty if value < 0 else value / repetition_penalty

            if temperature > 0:
                scaled_logits = logits / max(temperature, 1e-5)
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                remove_sorted = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
                remove_sorted[..., 0] = False

                sorted_logits[remove_sorted] = float("-inf")
                probs = torch.nan_to_num(torch.softmax(sorted_logits, dim=-1), nan=0.0)
                zero_rows = probs.sum(dim=-1) == 0
                if zero_rows.any():
                    probs[zero_rows, 0] = 1.0
                next_sorted_idx = torch.multinomial(probs, num_samples=1)
                next_token = sorted_indices.gather(1, next_sorted_idx)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # The web endpoint is batch-size one.  Yielding a tensor retains
            # a simple, useful API for callers that need token ids.
            yield next_token[0, 0]
            n_yielded += 1
            for batch_idx in range(B):
                all_seen_tokens[batch_idx].append(next_token[batch_idx, 0].item())
            # Honour EOS only after the minimum tail length (see generate() comment
            # for why an early-stage model emitting </s> id 2 immediately used to
            # truncate answers to 1-4 tokens).
            if eos_ids and n_yielded >= min_new_tokens and all(token.item() in eos_ids for token in next_token):
                return

            logits_t, states = self.forward(
                next_token, states=states, return_mtp=False,
                use_latent_reasoning=use_latent_reasoning,
                adapter_name=adapter_name,
            )
            next_token_logits = logits_t[:, -1, :]

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def cpu_dense_config(vocab_size: int = 32768, attention_kind: str = "alra") -> NeuroCoreConfig:
    """Return the maintained compact CPU configuration."""
    cfg = NeuroCoreConfig.small()
    cfg.model_name = "tantra-cpu-dense-32k"
    cfg.vocab.vocab_size = cfg.vocab.byte_bpe_vocab = vocab_size
    cfg.vocab.text_range_end = vocab_size - 1
    cfg.block.alra.dim, cfg.block.alra.num_heads, cfg.block.alra.head_dim = 512, 8, 64
    cfg.block.alra.attention_kind = attention_kind
    cfg.block.sgp.dim, cfg.block.sgp.expansion, cfg.block.sgp.implementation = 512, 2, "swiglu"
    cfg.block.num_layers, cfg.moe.num_experts, cfg.moe.real_top1 = 8, 1, False
    return cfg


def cpu_top1_moe_config(vocab_size: int = 32768, experts: int = 2, attention_kind: str = "alra") -> NeuroCoreConfig:
    """Return the real top-1 MoE CPU comparison configuration."""
    cfg = cpu_dense_config(vocab_size, attention_kind)
    cfg.model_name = f"tantra-cpu-top1-moe-{experts}e-32k"
    cfg.moe.num_experts, cfg.moe.top_k, cfg.moe.real_top1 = max(2, experts), 1, True
    return cfg


def cpu_10m_config(vocab_size: int = 32768, attention_kind: str = "alra") -> NeuroCoreConfig:
    """Return the compact baseline intended for CPU/distillation experiments."""
    cfg = cpu_dense_config(vocab_size, attention_kind)
    cfg.model_name = "tantra-cpu-10m-32k"
    cfg.block.alra.dim, cfg.block.alra.num_heads, cfg.block.alra.head_dim = 224, 7, 32
    cfg.block.sgp.dim, cfg.block.num_layers = 224, 4
    return cfg


def build_cpu_model(profile: str = "dense", attention_kind: str = "alra", vocab_size: int = 32768) -> "NeuroCoreModel":
    if profile == "dense":
        return NeuroCoreModel(cpu_dense_config(vocab_size, attention_kind), use_mtp=False, use_moe=False)
    if profile == "moe2":
        return NeuroCoreModel(cpu_top1_moe_config(vocab_size, 2, attention_kind), use_mtp=False, use_moe=True)
    if profile == "micro10":
        return NeuroCoreModel(cpu_10m_config(vocab_size, attention_kind), use_mtp=False, use_moe=False)
    raise ValueError(f"Unknown CPU profile: {profile}")
