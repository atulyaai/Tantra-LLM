"""
Tantra Sparse Mixture-of-Adapters
=================================

  shared base transformer  (RoPE, GQA, SwiGLU, RMSNorm, tied embeddings)
+ N tiny LoRA experts       (each trained alone on its own data)
+ a cheap router            (picks top-k experts per request)

Only the selected experts run, so adding expert #1000 costs disk space,
not CPU time. Each expert is a separate small file, so you can train /
replace one expert without touching the others.

Growth: `grow_depth()` inserts new layers whose output projections are
zero, so the grown model computes exactly the same function and training
continues from there (tiny -> small -> ... -> 1B on the same laptop).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from Tantra.config import ModelConfig


# ── building blocks ───────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class LoRALinear(nn.Module):
    """Linear layer with any number of named LoRA experts attached.

    Which experts are active (and their weights) is set by the parent model
    via `self.active`, a list of (name, weight) pairs. Inactive experts cost
    nothing.
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = False):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=bias)
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.scale: Dict[str, float] = {}
        self.active: List[Tuple[str, float]] = []

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    def _device(self) -> torch.device:
        # After dynamic INT8 quantisation `base` has no parameters (weight is packed); experts stay on CPU.
        p = next(self.base.parameters(), None)
        return p.device if p is not None else torch.device("cpu")

    def add_expert(self, name: str, rank: int, alpha: float) -> None:
        d_in, d_out = self.base.in_features, self.base.out_features
        A = torch.empty(rank, d_in, device=self._device())
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        self.lora_A[name] = nn.Parameter(A)
        self.lora_B[name] = nn.Parameter(torch.zeros(d_out, rank, device=self._device()))
        self.scale[name] = alpha / rank

    def remove_expert(self, name: str) -> None:
        for d in (self.lora_A, self.lora_B):
            if name in d:
                del d[name]
        self.scale.pop(name, None)

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        for name, w in self.active:
            if name in self.lora_A:
                y = y + (x @ self.lora_A[name].t() @ self.lora_B[name].t()) * (self.scale[name] * w)
        return y


def rope_cache(head_dim: int, max_len: int, theta: float) -> Tuple[Tensor, Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [B, H, T, Dh]; cos/sin: [T, Dh/2]
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv = cfg.n_heads, cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        kv_dim = self.head_dim * self.n_kv
        self.q = LoRALinear(cfg.d_model, cfg.d_model)
        self.k = LoRALinear(cfg.d_model, kv_dim)
        self.v = LoRALinear(cfg.d_model, kv_dim)
        self.o = LoRALinear(cfg.d_model, cfg.d_model)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: Optional[Dict] = None) -> Tensor:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # causal only needed when processing more than one new token
        if T > 1 and k.shape[2] != T:  # prefill with existing cache: explicit mask
            S = k.shape[2]
            mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(diagonal=S - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = LoRALinear(cfg.d_model, cfg.mlp_hidden)
        self.up = LoRALinear(cfg.d_model, cfg.mlp_hidden)
        self.down = LoRALinear(cfg.mlp_hidden, cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        return x + self.mlp(self.norm2(x))

    def lora_modules(self) -> Dict[str, LoRALinear]:
        a, m = self.attn, self.mlp
        return {"q": a.q, "k": a.k, "v": a.v, "o": a.o, "gate": m.gate, "up": m.up, "down": m.down}


# ── router ────────────────────────────────────────────────────────────────────

class Router(nn.Module):
    """Bag-of-embeddings classifier over the prompt -> expert scores.

    Uses the (frozen) token embeddings, so routing costs one tiny matmul and
    runs before the transformer. Growable: adding an expert adds one row.
    """

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)  # starts with 0 outputs; grows one row per expert
        self.head.weight = nn.Parameter(torch.empty(0, hidden))
        self.head.bias = nn.Parameter(torch.empty(0))
        self.head.out_features = 0
        self.names: List[str] = []

    def add_expert(self, name: str) -> None:
        old = self.head
        new = nn.Linear(old.in_features, old.out_features + 1).to(old.weight.device)
        with torch.no_grad():
            new.weight[:-1] = old.weight
            new.bias[:-1] = old.bias
            new.weight[-1].normal_(0, 0.02)
            new.bias[-1] = 0.0
        self.head = new
        self.names.append(name)

    def remove_expert(self, name: str) -> None:
        i = self.names.index(name)
        keep = [j for j in range(len(self.names)) if j != i]
        old = self.head
        new = nn.Linear(old.in_features, len(keep)).to(old.weight.device)
        with torch.no_grad():
            new.weight.copy_(old.weight[keep])
            new.bias.copy_(old.bias[keep])
        self.head = new
        self.names.pop(i)

    def forward(self, emb: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # emb: [B, T, D] token embeddings; mask: [B, T] 1 for real tokens
        if mask is None:
            pooled = emb.mean(1)
        else:
            m = mask.unsqueeze(-1).to(emb.dtype)
            pooled = (emb * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.proj(pooled))


# ── full model ────────────────────────────────────────────────────────────────

class TantraModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.router = Router(cfg.d_model)
        self.expert_ranks: Dict[str, int] = {}
        cos, sin = rope_cache(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for layer in self.layers:  # scaled residual init
            nn.init.normal_(layer.attn.o.base.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(layer.mlp.down.base.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ── parameters ──
    def base_parameters(self) -> Iterable[nn.Parameter]:
        for n, p in self.named_parameters():
            if "lora_" not in n and not n.startswith("router."):
                yield p

    def expert_parameters(self, name: str) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if n.endswith(f"lora_A.{name}") or n.endswith(f"lora_B.{name}")]

    def num_params(self, base_only: bool = True) -> int:
        return sum(p.numel() for p in (self.base_parameters() if base_only else self.parameters()))

    # ── experts ──
    @property
    def expert_names(self) -> List[str]:
        return list(self.router.names)

    def _lora_targets(self):
        for layer in self.layers:
            mods = layer.lora_modules()
            for t in self.cfg.lora_targets:
                yield mods[t]

    def _all_lora(self):
        for layer in self.layers:
            yield from layer.lora_modules().values()

    def add_expert(self, name: str, rank: Optional[int] = None) -> None:
        if name in self.expert_ranks:
            return
        rank = rank or self.cfg.lora_rank
        for mod in self._lora_targets():
            mod.add_expert(name, rank, self.cfg.lora_alpha)
        self.router.add_expert(name)
        self.expert_ranks[name] = rank

    def remove_expert(self, name: str) -> None:
        for mod in self._all_lora():
            mod.remove_expert(name)
        self.router.remove_expert(name)
        self.expert_ranks.pop(name, None)

    def set_active(self, experts: Sequence[Tuple[str, float]]) -> None:
        experts = list(experts)
        for mod in self._all_lora():
            mod.active = experts

    def expert_state(self, name: str) -> Dict[str, Tensor]:
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters()
                if n.endswith(f"lora_A.{name}") or n.endswith(f"lora_B.{name}")}

    def lora_shapes(self, name: str, rank: int) -> Dict[str, Tuple[int, ...]]:
        """Parameter names and shapes an adapter `name` of `rank` must provide for this base."""
        where = {id(m): n for n, m in self.named_modules()}
        shapes = {}
        for mod in self._lora_targets():
            prefix = where[id(mod)]
            shapes[f"{prefix}.lora_A.{name}"] = (rank, mod.base.in_features)
            shapes[f"{prefix}.lora_B.{name}"] = (mod.base.out_features, rank)
        return shapes

    def attach_lora(self, name: str, state: Dict[str, Tensor], rank: int) -> None:
        """Attach only the LoRA weights of an expert already known to the router (lazy loading).
        Refuses any missing, extra or mis-shaped tensor instead of skipping it."""
        expected = self.lora_shapes(name, rank)
        given = {k: tuple(v.shape) for k, v in state.items()}
        if given != expected:
            missing = sorted(set(expected) - set(given)); extra = sorted(set(given) - set(expected))
            wrong = sorted(k for k in set(given) & set(expected) if given[k] != expected[k])
            raise ValueError(f"Adapter '{name}' does not fit this base: missing={missing[:3]} extra={extra[:3]} wrong_shape={wrong[:3]}")
        for mod in self._lora_targets():
            mod.add_expert(name, rank, self.cfg.lora_alpha)
        params = dict(self.named_parameters())
        with torch.no_grad():
            for k, v in state.items():
                params[k].copy_(v)
        self.expert_ranks[name] = rank

    def detach_lora(self, name: str) -> None:
        """Free an expert's LoRA weights but keep its router entry."""
        for mod in self._all_lora():
            mod.remove_expert(name)
            mod.active = [(n, w) for n, w in mod.active if n != name]
        self.expert_ranks.pop(name, None)

    def load_expert_state(self, name: str, state: Dict[str, Tensor], rank: int) -> None:
        self.add_expert(name, rank)
        params = dict(self.named_parameters())
        with torch.no_grad():
            for n, t in state.items():
                if n in params and params[n].shape == t.shape:
                    params[n].copy_(t)

    # ── routing ──
    @torch.no_grad()
    def route(self, input_ids: Tensor, max_experts: int = 4, cum_prob: float = 0.8,
              min_prob: float = 0.08) -> List[Tuple[str, float]]:
        """Adaptive routing: the router decides HOW MANY experts a request needs.

        Experts are added in order of confidence until their combined probability
        reaches `cum_prob` (or `max_experts`). A clear request ("hi") uses 1 expert;
        a mixed one ("hi, who are you?") uses 2-3. Experts below `min_prob` are
        dropped. Returned weights are renormalised to sum to 1.
        """
        if not self.router.names:
            return []
        probs = self.router(self.embed(input_ids)).softmax(-1)[0]
        vals, idx = probs.sort(descending=True)
        chosen, total = [], 0.0
        for v, i in zip(vals.tolist(), idx.tolist()):
            if len(chosen) >= max_experts or total >= cum_prob or (chosen and v < min_prob):
                break
            chosen.append((self.router.names[i], v))
            total += v
        return [(n, w / total) for n, w in chosen]

    # ── forward ──
    def forward(self, input_ids: Tensor, caches: Optional[List[Dict]] = None, pos: int = 0) -> Tensor:
        T = input_ids.shape[1]
        needed = pos + T
        if needed > self.rope_cos.size(0):
            new_max = max(needed, self.rope_cos.size(0) * 2)
            c, s = rope_cache(self.cfg.d_model // self.cfg.n_heads, new_max, self.cfg.rope_theta)
            self.rope_cos = c.to(self.rope_cos.device)
            self.rope_sin = s.to(self.rope_sin.device)
        cos, sin = self.rope_cos[pos:needed], self.rope_sin[pos:needed]
        x = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, None if caches is None else caches[i])
        return F.linear(self.norm(x), self.embed.weight)  # tied head

    def loss(self, input_ids: Tensor, labels: Tensor) -> Tensor:
        logits = self(input_ids)
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                               labels[:, 1:].reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 64, temperature: float = 0.7,
                 top_k: int = 40, top_p: float = 0.9, stop_ids: Sequence[int] = (),
                 repetition_penalty: float = 1.1):
        """Yields token ids one at a time (KV-cached)."""
        caches = [{} for _ in self.layers]
        ids = input_ids[:, -(self.cfg.max_seq_len - max_new_tokens):]
        logits = self(ids, caches, 0)[:, -1]
        pos = ids.shape[1]
        seen = ids[0].tolist()
        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0 and seen:
                s = torch.tensor(sorted(set(seen[-64:])), device=logits.device)
                lg = logits[0, s]
                logits[0, s] = torch.where(lg > 0, lg / repetition_penalty, lg * repetition_penalty)
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = logits.topk(min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = -float("inf")
                probs = logits.softmax(-1)
                if top_p < 1.0:
                    sp, si = probs.sort(descending=True)
                    cut = sp.cumsum(-1) - sp > top_p
                    sp[cut] = 0
                    probs = torch.zeros_like(probs).scatter(-1, si, sp)
                nxt = torch.multinomial(probs / probs.sum(-1, keepdim=True), 1)
            tok = int(nxt)
            if tok in stop_ids or pos >= self.cfg.max_seq_len:
                return
            yield tok
            seen.append(tok)
            logits = self(nxt, caches, pos)[:, -1]
            pos += 1

    # ── growth ──
    def grow_depth(self, n_new: int, where: str = "interleave") -> None:
        """Add `n_new` layers without changing the model's output.

        New layers are copies of existing ones with zeroed output projections
        (attn.o, mlp.down) and zeroed expert LoRA-B, so each starts as the
        identity on the residual stream. Train afterwards to use them.
        """
        old = list(self.layers)
        n = len(old)
        src = [round(i * n / n_new) for i in range(n_new)] if where == "interleave" else [n - 1] * n_new
        new_layers: List[nn.Module] = []
        for i, layer in enumerate(old):
            new_layers.append(layer)
            for _ in range(src.count(i)):
                blk = copy.deepcopy(layer)
                with torch.no_grad():
                    blk.attn.o.base.weight.zero_()
                    blk.mlp.down.base.weight.zero_()
                    for mod in blk.lora_modules().values():
                        for p in mod.lora_B.values():
                            p.zero_()
                new_layers.append(blk)
        self.layers = nn.ModuleList(new_layers)
        self.cfg.n_layers = len(new_layers)

    # ── io ──
    def save(self, path: str | Path, experts_dir: Optional[str | Path] = None) -> None:
        """Save base (+ router) to `path`; each expert to experts_dir/<name>.pt."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        base_sd = {k: v for k, v in self.state_dict().items() if "lora_" not in k}
        torch.save({"config": self.cfg.to_dict(), "state": base_sd,
                    "router_names": self.router.names, "expert_ranks": self.expert_ranks}, path)
        if experts_dir is not None:
            for name in self.expert_names:
                save_expert(self, name, experts_dir)

    @classmethod
    def load(cls, path: str | Path, experts_dir: Optional[str | Path] = None,
             map_location: str = "cpu", lazy: bool = False) -> "TantraModel":
        """lazy=True registers experts with the router only; Tantra.inference.expert_cache
        attaches each adapter's weights when a request is routed to it."""
        # weights_only: model files hold tensors and plain metadata; never unpickle arbitrary objects.
        ck = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(ModelConfig.from_dict(ck["config"]))
        ranks = ck.get("expert_ranks", {})
        for name in ck.get("router_names", []):
            if lazy:
                model.router.add_expert(name)
            else:
                model.add_expert(name, ranks.get(name))
        if lazy:
            model.load_state_dict({k: v for k, v in ck["state"].items() if "lora_" not in k}, strict=False)
            return model
        model.load_state_dict(ck["state"], strict=False)
        if experts_dir is not None:
            for name in model.expert_names:
                f = Path(experts_dir) / f"{name}.pt"
                if f.exists():
                    e = torch.load(f, map_location=map_location, weights_only=True)
                    model.load_expert_state(name, e["state"], e["rank"])
        return model


def save_expert(model: TantraModel, name: str, experts_dir: str | Path, base_sha256: Optional[str] = None,
                tokenizer_sha256: Optional[str] = None) -> Path:
    d = Path(experts_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{name}.pt"
    payload = {"rank": model.expert_ranks[name], "state": model.expert_state(name)}
    if base_sha256:
        payload["base_sha256"] = base_sha256
    if tokenizer_sha256:
        payload["tokenizer_sha256"] = tokenizer_sha256
    torch.save(payload, f)
    return f


def build_model(preset_or_cfg="tiny", vocab_size: Optional[int] = None) -> TantraModel:
    from Tantra.config import get_preset
    cfg = get_preset(preset_or_cfg) if isinstance(preset_or_cfg, str) else preset_or_cfg
    if vocab_size:
        cfg.vocab_size = vocab_size
    return TantraModel(cfg)


def quantize_for_cpu(model: TantraModel) -> TantraModel:
    """INT8 dynamic quantization of the big base matmuls (~4x smaller, faster on CPU).
    LoRA experts stay fp32 (they are tiny)."""
    model.eval()
    for layer in model.layers:
        for mod in layer.lora_modules().values():
            mod.base = torch.ao.quantization.quantize_dynamic(
                nn.Sequential(mod.base), {nn.Linear}, dtype=torch.qint8)[0]
    return model
