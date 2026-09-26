"""
Tantra/model.py — The NeuroCore language model.

Block = norm -> attention (ALRA linear, or sliding-window softmax every Nth layer) -> norm -> SwiGLU MLP.
Optional: MTP head (predicts t+2 during training), category specialist layers, Top-1 MoE, BitNet.
"""

from typing import Optional, List, Dict, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from Tantra.config import EOS_ID, NeuroCoreConfig, ALRAConfig, SGPConfig, NeuroCoreBlockConfig, BitNetConfig
from Tantra.bitnet import BitLinear
from Tantra.utils import elu_plus_one, top_k_mask, get_logger

log = get_logger("tantra.model")


# ── DynamicScaleNorm ──

class DynamicScaleNorm(nn.Module):
    """
    DSN: LayerNorm(x) * sigmoid(W*x + b) * gamma + beta
    Learned scale adapts to input magnitude dynamically.
    Uses native C++ LayerNorm kernel for guaranteed gradient stability.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.w_scale = nn.Linear(dim, 1, bias=True)
        nn.init.zeros_(self.w_scale.weight)
        nn.init.constant_(self.w_scale.bias, 2.0)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        x_norm = F.layer_norm(x.float(), (x.shape[-1],), eps=self.eps).to(orig_dtype)
        # Compute scale in FP32 to avoid overflow/underflow with fp16/bf16 inputs
        scale = torch.sigmoid(self.w_scale(x.float())).to(orig_dtype)
        # BUG-19 FIX: Cast gamma/beta to x's dtype so the output stays in the
        # model's working precision (e.g. bf16 under AMP) instead of silently
        # upcasting the whole activation to float32 via gamma's default float32 storage.
        return x_norm * scale * self.gamma.to(orig_dtype) + self.beta.to(orig_dtype)


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
    def __init__(self, config: ALRAConfig, bitnet_config: Optional[BitNetConfig] = None):
        super().__init__()
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.use_forget_gate = config.use_forget_gate
        self.eps = 1e-4  # BUG-02 FIX: 1e-6 caused near-zero denominator div (≈1e6 magnitude) with ELU+1 kernels early in training

        assert self.dim == self.num_heads * self.head_dim, "dim must be num_heads * head_dim"

        linear_cls = BitLinear if bitnet_config and bitnet_config.enabled else nn.Linear

        # Early trained checkpoints include projection biases. Retain them so
        # those weights do not silently load into a partly-random model.
        self.w_q = linear_cls(self.dim, self.dim, bias=True)
        self.w_k = linear_cls(self.dim, self.dim, bias=True)
        self.w_v = linear_cls(self.dim, self.dim, bias=True)
        self.w_o = linear_cls(self.dim, self.dim, bias=True)
        
        if self.use_forget_gate:
            self.w_gate = linear_cls(self.dim, self.num_heads, bias=True)

        self.rope = RotaryPositionalEncoding(self.head_dim)
        
    def _apply_kernel(self, x: Tensor) -> Tensor:
        return elu_plus_one(x.clamp(min=-20.0, max=20.0))

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

        # Scale queries ONCE here so all 3 code paths (fast, chunked, sequential)
        # use identical attention math during both training and inference.
        Q = Q * (1.0 / (self.head_dim ** 0.5))

        gates = None
        if self.use_forget_gate:
            gates = torch.sigmoid(self.w_gate(x)).transpose(1, 2)

        if state is not None and T > 1:
            out, S, z = self._chunked_forward(Q, K, V, gates, state.get("S"), state.get("z"))
            state["S"], state["z"] = S, z
            new_state = state
            new_state["step"] = past_len + T
        elif state is not None or T == 1:
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
        out, _, _ = self._chunked_forward(Q, K, V, gates, None, None)
        return out

    def _chunked_forward(self, Q: Tensor, K: Tensor, V: Tensor, gates: Optional[Tensor],
                         S: Optional[Tensor], z: Optional[Tensor], chunk: int = 256):
        """Chunkwise-parallel gated linear attention.

        Exactly equals the token-by-token recurrence
            S_t = g_t * S_{t-1} + k_t v_t^T,   z_t = g_t * z_{t-1} + k_t,   o_t = q_t S_t / (q_t . z_t)
        but runs as matrix ops: quadratic only inside each chunk, linear across chunks.
        Memory ~ T * chunk instead of T^2, and it returns the final state so a whole
        prompt can be pre-filled in one call.
        """
        B, H, T, Dh = Q.shape
        if S is None:
            S = Q.new_zeros(B, H, Dh, Dh)
        if z is None:
            z = Q.new_zeros(B, H, Dh)
        log_g = (torch.log(gates.clamp(min=1e-4, max=1.0)) if gates is not None
                 else Q.new_zeros(B, H, T))
        outs = []
        for c in range(0, T, chunk):
            e = min(c + chunk, T)
            q, k, v, lg = Q[:, :, c:e], K[:, :, c:e], V[:, :, c:e], log_g[:, :, c:e]
            L = e - c
            cum = torch.cumsum(lg, dim=-1)                                   # [B,H,L]
            diff = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).clamp(max=0.0)    # [B,H,L,L]
            causal = torch.tril(torch.ones(L, L, device=Q.device, dtype=torch.bool))
            decay = torch.where(causal, torch.exp(diff.clamp(min=-30.0)), torch.zeros_like(diff))
            attn = torch.matmul(q, k.transpose(-2, -1)) * decay
            carry = torch.exp(cum.clamp(min=-30.0)).unsqueeze(-1)            # [B,H,L,1]
            num = torch.matmul(attn, v) + carry * torch.matmul(q, S)
            den = attn.sum(dim=-1, keepdim=True) + carry * (q * z.unsqueeze(2)).sum(-1, keepdim=True)
            outs.append(torch.nan_to_num(num / den.clamp(min=self.eps), nan=0.0, posinf=1.0, neginf=-1.0))
            # advance state to the end of this chunk
            tail = torch.exp((cum[..., -1:] - cum).clamp(min=-30.0)).unsqueeze(-1)  # [B,H,L,1]
            g_all = torch.exp(cum[..., -1].clamp(min=-30.0))                         # [B,H]
            S = g_all[..., None, None] * S + torch.matmul((k * tail).transpose(-2, -1), v)
            z = g_all[..., None] * z + (k * tail).sum(dim=2)
        return torch.cat(outs, dim=2), S, z

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
            S = S * gate_t.unsqueeze(-1) + KV_t
            z = z * gate_t + K_t
        else:
            S = S + KV_t
            z = z + K_t
            
        state['S'] = S
        state['z'] = z

        num = torch.matmul(Q_t.unsqueeze(2), S).squeeze(2)
        den = (Q_t * z).sum(dim=-1, keepdim=True) + self.eps
        out = torch.nan_to_num(num / den, nan=0.0, posinf=1.0, neginf=-1.0).unsqueeze(2)
        
        return out, state


class CausalSelfAttention(nn.Module):
    """Standard causal softmax attention, optionally limited to a sliding window.

    window=0 -> full causal attention. window=W -> each token sees at most the
    previous W tokens (itself included). During generation the KV cache is
    trimmed to W entries, so memory and per-token cost stay constant on CPU.
    """
    def __init__(self, config: ALRAConfig, bitnet_config: Optional[BitNetConfig] = None, window: int = 0):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = config.dim, config.num_heads, config.head_dim
        self.window = int(window or 0)
        linear_cls = BitLinear if bitnet_config and bitnet_config.enabled else nn.Linear
        self.w_q = linear_cls(self.dim, self.dim, bias=True)
        self.w_k = linear_cls(self.dim, self.dim, bias=True)
        self.w_v = linear_cls(self.dim, self.dim, bias=True)
        self.w_o = linear_cls(self.dim, self.dim, bias=True)
        self.rope = RotaryPositionalEncoding(self.head_dim)

    @staticmethod
    def _band_mask(q_len: int, kv_len: int, window: int, device) -> Tensor:
        # True = may attend. Query i sits at absolute kv position (kv_len - q_len + i).
        q_pos = torch.arange(kv_len - q_len, kv_len, device=device).unsqueeze(1)
        k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
        mask = k_pos <= q_pos
        if window > 0:
            mask = mask & (k_pos > q_pos - window)
        return mask

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, state: Optional[dict] = None) -> Tuple[Tensor, Optional[dict]]:
        batch, tokens, _ = x.shape
        q = self.w_q(x).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        has_cache = state is not None and state.get("k") is not None
        # Absolute position must be tracked separately: the cache may be trimmed.
        past_pos = int(state.get("pos", state["k"].shape[2])) if has_cache else 0

        q, k = self.rope.apply(q, k, tokens, offset=past_pos)

        if has_cache:
            k = torch.cat([state["k"], k], dim=2)
            v = torch.cat([state["v"], v], dim=2)
        kv_len = k.shape[2]

        if tokens == 1:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        elif not has_cache and mask is None and (self.window <= 0 or tokens <= self.window):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            band = self._band_mask(tokens, kv_len, self.window, q.device)
            if mask is not None:
                band = band & mask.bool() if mask.dtype == torch.bool else band
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=band, dropout_p=0.0)

        if state is not None:
            if self.window > 0 and kv_len > self.window - 1:
                # keep W-1 past entries; the next token itself makes W
                k = k[:, :, -(self.window - 1):] if self.window > 1 else k[:, :, :0]
                v = v[:, :, -(self.window - 1):] if self.window > 1 else v[:, :, :0]
            state["k"] = k
            state["v"] = v
            state["pos"] = past_pos + tokens

        return self.w_o(out.transpose(1, 2).reshape(batch, tokens, self.dim)), state


# ── SparseGatedProjection ──

class SparseGatedProjection(nn.Module):
    """SGP: brain-inspired sparse FFN with top-k% active neurons."""
    def __init__(self, config: SGPConfig, bitnet_config: Optional[BitNetConfig] = None):
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = self.dim * config.expansion
        self.k = max(1, int(self.hidden_dim * config.sparsity))
        
        linear_cls = BitLinear if bitnet_config and bitnet_config.enabled else nn.Linear
        
        self.w_up = linear_cls(self.dim, self.hidden_dim, bias=False)
        self.w_down = linear_cls(self.hidden_dim, self.dim, bias=False)
        self.w_gate = linear_cls(self.dim, self.hidden_dim, bias=True)
        
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

        # BUG-11 FIX: Actually update the monitoring metric.
        # detach() so this stat-tracking doesn't participate in backprop.
        with torch.no_grad():
            self._last_active_ratio = mask.float().mean().item()

        if self.training:
            mask_float = mask.to(x.dtype)
            # BUG-03 FIX: The original STE `mask_float.detach() - gates.detach() + gates`
            # passes gradients through `gates` for all neurons (active and inactive).
            # However, for inactive neurons the hidden state is ~0, so the gradient
            # of w_up and the output projection for those neurons is also ~0 —
            # they receive no useful learning signal from the loss.
            # Add a small soft bypass (0.01 * gates) so inactive neurons still carry
            # a tiny gradient signal from their pre-activation, keeping them "alive"
            # and preventing permanent dead neuron lock-in without meaningfully
            # changing the dominant (active) neuron outputs.
            mask_st = mask_float.detach() - gates.detach() + gates
            up = self.act(self.w_up(x))
            # Dominant active path + tiny soft bypass for inactive neurons
            hidden = up * mask_st + 0.01 * gates * (1.0 - mask_float.detach())
        else:
            mask_float = mask.to(x.dtype)
            up = self.act(self.w_up(x))
            hidden = up * mask_float

        return self.w_down(hidden)

    def get_activation_stats(self) -> dict:
        return {"active_ratio": self._last_active_ratio, "target_ratio": self.k / self.hidden_dim}


class SwiGLUProjection(nn.Module):
    """Dense CPU-friendly gated MLP; avoids top-k sorting and masking overhead."""
    def __init__(self, config: SGPConfig, bitnet_config: Optional[BitNetConfig] = None):
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = self.dim * config.expansion
        linear_cls = BitLinear if bitnet_config and bitnet_config.enabled else nn.Linear
        self.w_up = linear_cls(self.dim, self.hidden_dim, bias=False)
        self.w_gate = linear_cls(self.dim, self.hidden_dim, bias=False)
        self.w_down = linear_cls(self.hidden_dim, self.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Top1MoEProjection(nn.Module):
    """Actual top-1 MoE: selected token groups run through separate MLP experts."""
    def __init__(self, config: SGPConfig, num_experts: int, balance_coeff: float = 0.01, bitnet_config: Optional[BitNetConfig] = None):
        super().__init__()
        self.num_experts = max(2, num_experts)
        self.balance_coeff = balance_coeff
        linear_cls = BitLinear if bitnet_config and bitnet_config.enabled else nn.Linear
        self.router = linear_cls(config.dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLUProjection(config, bitnet_config) for _ in range(self.num_experts))
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
    def __init__(self, config: NeuroCoreBlockConfig, layer_idx: int, moe_config: Optional[Any] = None, use_moe: bool = False, bitnet_config: Optional[BitNetConfig] = None, **_ignored: Any):
        super().__init__()
        self.layer_idx = layer_idx
        self.pre_norm = config.pre_norm
        self.use_moe = use_moe
        dim = config.alra.dim
        
        self.norm1 = DynamicScaleNorm(dim)
        local_every = int(getattr(config.alra, "local_attn_every", 0) or 0)
        self.is_local_softmax = local_every > 0 and (layer_idx + 1) % local_every == 0
        if config.alra.attention_kind == "causal":
            self.attn = CausalSelfAttention(config.alra, bitnet_config)
        elif self.is_local_softmax:
            self.attn = CausalSelfAttention(config.alra, bitnet_config,
                                            window=int(getattr(config.alra, "local_window", 512) or 512))
        else:
            self.attn = ALRAAttention(config.alra, bitnet_config)
        self.norm2 = DynamicScaleNorm(dim)
        if use_moe and moe_config is not None and getattr(moe_config, "real_top1", False):
            self.mlp = Top1MoEProjection(config.sgp, moe_config.num_experts, moe_config.load_balance_coeff, bitnet_config)
        elif config.sgp.implementation == "swiglu":
            self.mlp = SwiGLUProjection(config.sgp, bitnet_config)
        else:
            self.mlp = SparseGatedProjection(config.sgp, bitnet_config)

    def forward(
        self, 
        x: Tensor,
        mask: Optional[Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[Tensor, Optional[dict]]:
        if self.pre_norm:
            attn_out, new_state = self.attn(self.norm1(x), mask=mask, state=state)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
        else:
            attn_out, new_state = self.attn(x, mask=mask, state=state)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.mlp(x))
        return x, new_state


# ── LatentCoTHeader ──

class LatentCoTHeader(nn.Module):
    """
    Latent Chain-of-Thought (CoT) Reasoning Header.
    Applies recurrent depth iterations on model hidden states to allow latent reasoning steps
    prior to final token prediction.
    """
    def __init__(self, dim: int, reasoning_depth: int = 3, bitnet_config: Optional[BitNetConfig] = None):
        super().__init__()
        self.dim = dim
        self.reasoning_depth = reasoning_depth
        self.reasoning_norm = DynamicScaleNorm(dim)
        linear_cls = BitLinear if bitnet_config and bitnet_config.enabled else nn.Linear
        # The released checkpoint was trained with these biases.
        self.reasoning_proj = linear_cls(dim, dim, bias=True)
        self.gate = linear_cls(dim * 2, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        state = x
        for _ in range(self.reasoning_depth):
            normed = self.reasoning_norm(state)
            delta = F.silu(self.reasoning_proj(normed))
            g = torch.sigmoid(self.gate(torch.cat([state, delta], dim=-1)))
            # BUG-04 FIX: Tighten clamp from ±100 → ±10.
            # ±100 was 10× wider than the logit soft-cap (±30), causing final_norm
            # to see wildly-scaled inputs and destabilise its variance estimates.
            # ±10 keeps states well within a normalizable range while still allowing
            # the reasoning header full expressive power.
            state = torch.clamp(state + g * delta, min=-10.0, max=10.0)
        return state


# ── NeuroCoreModel with Multi-Token Prediction (MTP) & Latent CoT ──

class NeuroCoreModel(nn.Module):
    """Full NeuroCore language model with Multi-Token Prediction (MTP) heads and Latent Reasoning Headers."""

    def __init__(self, config: NeuroCoreConfig, use_mtp: bool = True, reasoning_depth: int = 3, use_moe: bool = False, **_ignored: Any):
        super().__init__()
        self.config = config
        self.dim = config.block.alra.dim
        self.vocab_size = getattr(config.vocab, "total_embedding_size", config.vocab.vocab_size)
        self.text_vocab_size = config.vocab.vocab_size  # text-only size, for callers that need it
        self.gradient_checkpointing = getattr(config.training, "gradient_checkpointing", False)
        self.use_mtp = use_mtp
        self.reasoning_depth = reasoning_depth
        # ``num_experts`` alone is metadata for legacy checkpoints. Only the
        # explicit real Top-1 configuration enables token-level conditional
        # compute; otherwise the model stays dense.
        self.use_moe = bool(
            use_moe
            and getattr(config.moe, "real_top1", False)
            and getattr(config.moe, "num_experts", 1) > 1
        )
        self.embed = nn.Embedding(self.vocab_size, self.dim)
        nn.init.normal_(self.embed.weight, std=0.02)

        bitnet_config = config.bitnet if config.bitnet.enabled else None

        self.layers = nn.ModuleList([
            NeuroCoreBlock(
                config.block,
                layer_idx=i,
                moe_config=config.moe if self.use_moe else None,
                use_moe=self.use_moe,
                bitnet_config=bitnet_config,
            )
            for i in range(config.block.num_layers)
        ])

        self.final_norm = DynamicScaleNorm(self.dim)
        self.latent_header = LatentCoTHeader(self.dim, reasoning_depth=reasoning_depth, bitnet_config=bitnet_config)
        
        linear_cls = BitLinear if (bitnet_config and bitnet_config.enabled) else nn.Linear

        # Primary head (predicts t+1) - use nn.Linear because weights are tied to embed.weight (FP32/standard embedding)
        self.output_proj = nn.Linear(self.dim, self.vocab_size, bias=False)
        self.output_proj.weight = self.embed.weight

        # Auxiliary MTP head (predicts t+2 for DeepSeek-style Multi-Token Prediction)
        if self.use_mtp:
            self.mtp_head = linear_cls(self.dim, self.vocab_size, bias=False)

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
        bitnet_config = self.config.bitnet if self.config.bitnet.enabled else None
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
                    bitnet_config=bitnet_config
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

    def grow_category(self, category: str, cap: int, max_params: int = 1_000_000_000) -> bool:
        """Append one more specialist layer to a category (up to ``cap`` total), respecting 1B param ceiling."""
        import copy
        if category not in self.category_layers:
            return False
        stack = self.category_layers[category]
        if len(stack) >= cap:
            return False

        # Guard 1 Billion Parameter Limit
        current_params = sum(p.numel() for p in self.parameters())
        source = stack[-1]
        layer_params = sum(p.numel() for p in source.parameters())
        if (current_params + layer_params) > max_params:
            log.warning(
                f"⛔ [GROWTH BLOCKED] Growing category '{category}' (+{layer_params/1e6:.1f}M params) "
                f"would exceed 1 Billion parameter ceiling ({max_params/1e6:.0f}M). Current: {current_params/1e6:.1f}M."
            )
            return False

        new_layer = copy.deepcopy(source)
        with torch.no_grad():
            for p in new_layer.parameters():
                p.data.add_(torch.randn_like(p.data) * 0.001)
        stack.append(new_layer)
        self.category_gates[category].append(
            nn.Parameter(torch.zeros((), dtype=torch.get_default_dtype()), requires_grad=True)
        )
        log.info(f"🌱 Category '{category}' auto-grown to depth {len(stack)} (Total: {sum(p.numel() for p in self.parameters())/1e6:.1f}M params).")
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

    def get_aux_loss(self) -> Tensor:
        """Aggregate real-MoE router balancing losses for training."""
        moe_layers = [
            layer for layer in self.layers
            if isinstance(getattr(layer, "mlp", None), Top1MoEProjection)
            and layer.mlp.last_aux_loss is not None
        ]
        if not moe_layers:
            return torch.zeros((), device=self.embed.weight.device)
        losses = [layer.mlp.last_aux_loss for layer in moe_layers]
        total = torch.stack(losses).sum()
        # Free computation graphs immediately so they don't accumulate across steps
        for layer in moe_layers:
            layer.mlp.last_aux_loss = None
        return total

    def forward(
        self,
        token_ids: Optional[Tensor] = None,
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
            if self.training and getattr(self, "gradient_checkpointing", False) and layer_state is None:
                from torch.utils.checkpoint import checkpoint
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                x, new_layer_state = checkpoint(create_custom_forward(layer), x, mask, None, use_reentrant=False)
            else:
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
            # FIX #2 (CRITICAL): Re-normalize after latent_header.
            # latent_header adds unnormalized deltas (state += g*delta), destroying
            # the normalization that final_norm applied.  output_proj uses weight-tied
            # embeddings that expect unit-scale input; skipping this re-norm causes
            # logit magnitudes to explode, saturating softmax and killing gradients.
            x = self.final_norm(x)

        logits_main = self.output_proj(x)
        # Logit soft-capping (Gemma-2 style) to [-30.0, 30.0] to permanently prevent softmax overflow / NaNs
        logits_main = 30.0 * torch.tanh(logits_main / 30.0)

        if return_mtp and self.use_mtp:
            logits_mtp = self.mtp_head(x)
            logits_mtp = 30.0 * torch.tanh(logits_mtp / 30.0)
            return (logits_main, logits_mtp), new_states

        return logits_main, new_states

    # ── Generation ────────────────────────────────────────────────────────
    @staticmethod
    def _pick_next(logits: Tensor, history: List[List[int]], temperature: float, top_p: float,
                   repetition_penalty: float, no_repeat_ngram_size: int,
                   banned_token_ids: Optional[List[int]]) -> Tensor:
        logits = torch.nan_to_num(logits.clone(), nan=-1e9, posinf=1e4, neginf=-1e9)
        vocab = logits.size(-1)
        for b, hist in enumerate(history):
            # never repeat the same token 4 times in a row
            if len(hist) >= 3 and hist[-1] == hist[-2] == hist[-3]:
                logits[b, hist[-1]] = -1e9
            # block repeated n-grams
            n = no_repeat_ngram_size
            if n > 0 and len(hist) >= n - 1:
                prefix = tuple(hist[len(hist) - (n - 1):]) if n > 1 else ()
                for k in range(len(hist) - (n - 1)):
                    if tuple(hist[k:k + n - 1]) == prefix and 0 <= hist[k + n - 1] < vocab:
                        logits[b, hist[k + n - 1]] = -1e9
            # repetition penalty on generated tokens only
            if repetition_penalty != 1.0:
                for tok in set(hist):
                    if 0 <= tok < vocab:
                        v = logits[b, tok]
                        logits[b, tok] = v * repetition_penalty if v < 0 else v / repetition_penalty
        for tok in banned_token_ids or []:
            if 0 <= tok < vocab:
                logits[:, tok] = -1e9
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        sorted_logits, sorted_idx = torch.sort(logits / max(temperature, 1e-5), descending=True)
        remove = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits[remove] = float("-inf")
        probs = torch.nan_to_num(torch.softmax(sorted_logits, dim=-1), nan=0.0)
        probs[probs.sum(dim=-1) == 0, 0] = 1.0
        return sorted_idx.gather(1, torch.multinomial(probs, num_samples=1))

    @torch.no_grad()  # NOT inference_mode — that poisons the RoPE cache for later training
    def _generate_iter(self, prompt_ids: Tensor, max_new_tokens: int, temperature: float, top_p: float,
                       repetition_penalty: float, no_repeat_ngram_size: int, use_latent_reasoning: bool,
                       eos_token_id, min_new_tokens: int, adapter_name: Optional[str],
                       banned_token_ids: Optional[List[int]]):
        self.eval()
        if prompt_ids.numel() == 0:
            prompt_ids = torch.tensor([[1]], device=self.device, dtype=torch.long)
        B, T = prompt_ids.shape
        n_states = len(self.layers)
        category = adapter_name or self.active_category
        if category and category in self.category_layers:
            n_states += len(self.category_layers[category])
        states = [{} for _ in range(n_states)]
        # Prefill the whole prompt in one pass (attention caches come back exact).
        logits, states = self.forward(prompt_ids, states=states,
                                      use_latent_reasoning=use_latent_reasoning, adapter_name=adapter_name)
        eos_ids = set(eos_token_id) if isinstance(eos_token_id, (list, tuple, set)) else {eos_token_id}
        eos_ids.discard(None)
        history: List[List[int]] = [[] for _ in range(B)]
        for step in range(max_new_tokens):
            next_token = self._pick_next(logits[:, -1, :], history, temperature, top_p,
                                         repetition_penalty, no_repeat_ngram_size, banned_token_ids)
            yield next_token
            for b in range(B):
                history[b].append(int(next_token[b, 0]))
            if eos_ids and step + 1 >= min_new_tokens and all(int(t) in eos_ids for t in next_token[:, 0]):
                return
            logits, states = self.forward(next_token, states=states,
                                          use_latent_reasoning=use_latent_reasoning, adapter_name=adapter_name)

    def generate(self, prompt_ids: Tensor, max_new_tokens: int = 150, temperature: float = 0.35,
                 top_p: float = 0.85, repetition_penalty: float = 1.15, no_repeat_ngram_size: int = 3,
                 use_mtp_speculation: bool = False, use_latent_reasoning: bool = False,
                 eos_token_id: Optional[int] = EOS_ID, min_new_tokens: int = 1,
                 adapter_name: Optional[str] = None, banned_token_ids: Optional[List[int]] = None) -> Tensor:
        """Return prompt + generated token ids, shape [B, T + n]."""
        out = list(self._generate_iter(prompt_ids, max_new_tokens, temperature, top_p, repetition_penalty,
                                       no_repeat_ngram_size, use_latent_reasoning, eos_token_id,
                                       min_new_tokens, adapter_name, banned_token_ids))
        return torch.cat([prompt_ids] + out, dim=1) if out else prompt_ids

    def generate_stream(self, prompt_ids: Tensor, max_new_tokens: int = 150, temperature: float = 0.35,
                        top_p: float = 0.85, repetition_penalty: float = 1.15, no_repeat_ngram_size: int = 3,
                        use_mtp_speculation: bool = False, use_latent_reasoning: bool = False,
                        eos_token_id: Optional[int] = EOS_ID, min_new_tokens: int = 1,
                        adapter_name: Optional[str] = None, banned_token_ids: Optional[List[int]] = None):
        """Yield one token id (0-d tensor) at a time. Batch size 1."""
        for tok in self._generate_iter(prompt_ids, max_new_tokens, temperature, top_p, repetition_penalty,
                                       no_repeat_ngram_size, use_latent_reasoning, eos_token_id,
                                       min_new_tokens, adapter_name, banned_token_ids):
            yield tok[0, 0]

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Loading a trained model ──────────────────────────────────────────────────

def load_model(path: str, device: str = "cpu", int8: bool = False) -> Tuple["NeuroCoreModel", dict]:
    """Rebuild a NeuroCoreModel from any Tantra checkpoint (training or exported).

    Reads the architecture from the checkpoint itself: layer count (including
    auto-grown layers), MTP head, and category specialist layers.
    int8=True (CPU only): Linear layers run with 8-bit weights — ~4x less RAM
    for those layers and usually faster generation, tiny quality loss.
    """
    import re
    from Tantra.utils import safe_load_checkpoint

    ckpt = safe_load_checkpoint(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    cfg = ckpt.get("config")
    if isinstance(cfg, dict):
        cfg = NeuroCoreConfig._from_dict(cfg)
    if cfg is None:
        raise RuntimeError(f"{path} has no saved config; cannot rebuild the model.")
    layer_ids = [int(m.group(1)) for k in state for m in [re.match(r"layers\.(\d+)\.", k)] if m]
    if layer_ids:
        cfg.block.num_layers = max(layer_ids) + 1
    if "embed.weight" in state:
        rows = state["embed.weight"].shape[0]
        if rows != cfg.vocab.total_embedding_size:
            cfg.vocab.vocab_size = cfg.vocab.byte_bpe_vocab = rows
            cfg.vocab.audio_codebook_size = cfg.vocab.image_codebook_size = cfg.vocab.video_codebook_size = 0
    use_moe = bool(getattr(cfg.moe, "real_top1", False) and getattr(cfg.moe, "num_experts", 1) > 1)
    model = NeuroCoreModel(cfg, use_mtp=any(k.startswith("mtp_head") for k in state), use_moe=use_moe)
    depths: Dict[str, int] = {}
    for k in state:
        m = re.match(r"category_layers\.([^.]+)\.(\d+)\.", k)
        if m:
            depths[m.group(1)] = max(depths.get(m.group(1), 0), int(m.group(2)) + 1)
    for name, depth in depths.items():
        model.add_category_layers([name], depth=depth)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        log.warning(f"load_model: {len(missing)} missing / {len(unexpected)} unexpected tensors in {path}")
    model = model.to(device).eval()
    if int8 and str(device) == "cpu":
        model = _to_int8(model)
    return model, ckpt


def _to_int8(model: nn.Module) -> nn.Module:
    """8-bit weights for Linear layers. Uses torchao when installed, else PyTorch's built-in path."""
    import warnings
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
        return model
    except Exception:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    except Exception as exc:
        log.warning(f"INT8 not available in this PyTorch ({exc}); running in float32.")
        return model
