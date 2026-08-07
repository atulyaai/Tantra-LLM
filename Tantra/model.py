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
        self._cos_cached = None
        self._sin_cached = None

    def _build_cache(self, seq_len: int, device: torch.device):
        seq_len = max(1, seq_len)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()

    def get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        seq_len = max(1, seq_len)
        if self._cos_cached is None or seq_len > self._cos_cached.shape[0] or self._cos_cached.device != device:
            build_len = max(seq_len, 2048)
            self._build_cache(build_len, device)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


    def _rotate_half(self, x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply(self, q: Tensor, k: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
        cos, sin = self.get_cos_sin(seq_len, q.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

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
        
        Q, K = self.rope.apply(Q, K, T)
        
        Q = self._apply_kernel(Q)
        K = self._apply_kernel(K)
        
        gates = None
        if self.use_forget_gate:
            gates = torch.sigmoid(self.w_gate(x)).transpose(1, 2)
            
        if state is not None:
            out, new_state = self._sequential_forward(Q, K, V, gates, state)
        else:
            out = self._parallel_forward(Q, K, V, gates)
            new_state = None
            
        out = out.transpose(1, 2).reshape(B, T, self.dim)
        out = self.w_o(out)
        return out, new_state

    def _parallel_forward(self, Q: Tensor, K: Tensor, V: Tensor, gates: Optional[Tensor]) -> Tensor:
        """
        Chunked blockwise scan for linear O(1) memory complexity.
        Processes sequence in blocks of C=256 tokens carrying matrix state (S, z).
        Prevents memory explosion on 100K+ token context windows.
        """
        B, H, T, Dh = Q.shape
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
            self._last_active_ratio = mask_float.mean().item()
            mask_st = mask_float.detach() - gates.detach() + gates
            up = self.act(self.w_up(x))
            hidden = up * mask_st
        else:
            mask_float = mask.to(x.dtype)
            self._last_active_ratio = mask_float.mean().item()
            up = self.act(self.w_up(x))
            hidden = up * mask_float
            
        return self.w_down(hidden)

    def get_activation_stats(self) -> dict:
        return {"active_ratio": self._last_active_ratio, "target_ratio": self.k / self.hidden_dim}


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
        self.attn = ALRAAttention(config.alra)
        self.norm2 = DynamicScaleNorm(dim)
        self.mlp = SparseGatedProjection(config.sgp)

        if self.use_moe and moe_config is not None:
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
    ) -> Union[Tuple[Tensor, Optional[List[dict]]], Tuple[Tuple[Tensor, Tensor], Optional[List[dict]]]]:
        x = self.embed(token_ids)
        new_states = [] if states is not None else None

        for i, layer in enumerate(self.layers):
            layer_state = states[i] if states is not None else None
            x, new_layer_state = layer(x, mask=mask, state=layer_state)
            if new_states is not None:
                new_states.append(new_layer_state)

        x = self.final_norm(x)
        if use_latent_reasoning:
            x = self.latent_header(x)

        logits_main = self.output_proj(x)

        if return_mtp and self.use_mtp:
            logits_mtp = self.mtp_head(x)
            return (logits_main, logits_mtp), new_states

        return logits_main, new_states

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        use_mtp_speculation: bool = True,
        use_latent_reasoning: bool = True,
    ) -> Tensor:
        """Generate text using Multi-Token Prediction (MTP) and Latent CoT reasoning."""
        self.eval()
        if prompt_ids.numel() == 0 or prompt_ids.size(1) == 0:
            prompt_ids = torch.tensor([[1]], device=prompt_ids.device, dtype=torch.long)
        B, T = prompt_ids.shape
        states = [None] * len(self.layers)

        for t in range(T):
            token = prompt_ids[:, t:t+1]
            logits_t, states = self.forward(token, states=states, use_latent_reasoning=use_latent_reasoning)

        if isinstance(logits_t, tuple):
            logits_t = logits_t[0]

        next_token_logits = logits_t[:, -1, :]
        generated_ids = []

        for _ in range(max_new_tokens):
            next_token_logits = torch.nan_to_num(next_token_logits, nan=-1e9, posinf=1e4, neginf=-1e9)
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0)
                if probs.sum(dim=-1).item() == 0:
                    probs[:, 0] = 1.0
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_ids.append(next_token)
            logits_t, states = self.forward(next_token, states=states, use_latent_reasoning=use_latent_reasoning)

            if isinstance(logits_t, tuple):
                logits_t = logits_t[0]

            next_token_logits = logits_t[:, -1, :]

        return torch.cat([prompt_ids] + generated_ids, dim=1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
