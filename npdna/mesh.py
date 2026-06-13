"""Neural Mesh — sparse routing fabric for NP-DNA.

Routes each token to the top-k most relevant Strands out of N total.
Only k Strands compute per token — the rest are skipped.
This is TRUE sparse execution (Switch Transformer style), not dense masking.

Includes load-balancing loss to prevent dead Strands.
"""

from __future__ import annotations

import logging
from collections import defaultdict, Counter

import torch
from torch import Tensor, nn, jit

from .config import MeshConfig, StrandConfig
from .genome import Genome

logger = logging.getLogger(__name__)


class AttentionStrand(nn.Module):
    """Local causal attention strand used when ``strand_type='attention'``."""

    def __init__(self, genome: Genome, strand_id: int, config: StrandConfig,
                 category: str | None = None, device: torch.device | None = None):
        super().__init__()
        self.genome = genome
        self.strand_id = strand_id
        self.config = config
        self.category: str | None = category
        self.norm = nn.LayerNorm(config.hidden_size, device=device)
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, device=device)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, device=device)
        self.usage_count: int = 0

    def forward(
        self,
        x: Tensor,
        weights: dict[str, tuple[Tensor, Tensor]] | None = None,
        init_state: Tensor | None = None,
        return_final_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        B, T, H = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        scale = H ** -0.5
        scores = q @ k.transpose(-2, -1) * scale
        if T > 1:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = self.out(attn @ v)
        self.usage_count += B * T
        if return_final_state:
            state_size = self.config.state_size
            final = out[:, -1, :state_size]
            if final.shape[-1] < state_size:
                final = torch.nn.functional.pad(final, (0, state_size - final.shape[-1]))
            return out, final
        return out


def _make_strand(genome, strand_id, config, category=None, device=None):
    strand_type = getattr(config, "strand_type", "ssm")
    if strand_type == "attention":
        return AttentionStrand(genome, strand_id, config, category=category, device=device)
    
    return Strand(genome, strand_id, config, category=category, device=device)


def _sparse_route(x: Tensor, router: nn.Linear, top_k: int):
    """Route tokens to top-k strands. Returns (indices, weights, assignment).

    assignment maps each selected strand to the tokens that chose it:
        {strand_idx: (token_indices, weights_for_those_tokens)}
    """
    B, T, H = x.shape
    scores = router(x)  # (B, T, N)
    N = scores.shape[-1]
    K = min(top_k, N)

    top_weights, top_indices = torch.topk(scores, K, dim=-1)
    top_weights = top_weights.softmax(dim=-1)

    flat_indices = top_indices.reshape(-1)       # (B*T*K,)
    flat_weights = top_weights.reshape(-1)       # (B*T*K,)
    flat_pos = torch.arange(B * T, device=x.device).repeat_interleave(K)

    assignment: dict[int, tuple[Tensor, Tensor]] = {}
    unique_strands = torch.unique(flat_indices)
    for s in unique_strands.tolist():
        mask = flat_indices == s
        if mask.any():
            assignment[s] = (flat_pos[mask], flat_weights[mask])

    return top_weights, top_indices, assignment, scores


@jit.script
def _strand_step(
    state: Tensor,
    gate_input_t: Tensor,
    state_input_t: Tensor,
    W_rec: Tensor,
    b_rec: Tensor,
) -> Tensor:
    gate = torch.sigmoid(gate_input_t + state @ W_rec + b_rec)
    candidate = torch.tanh(state_input_t)
    return gate * state + (1.0 - gate) * candidate


def _strand_scan(
    gate_input: Tensor,
    state_input: Tensor,
    W_rec: Tensor,
    b_rec: Tensor,
    init_state: Tensor,
) -> Tensor:
    B, T, S = gate_input.shape
    state = init_state
    outputs = torch.empty(B, T, S, dtype=gate_input.dtype, device=gate_input.device)
    for t in range(T):
        state = _strand_step(state, gate_input[:, t, :], state_input[:, t, :], W_rec, b_rec)
        outputs[:, t, :] = state
    return outputs


class Strand(nn.Module):
    def __init__(self, genome: Genome, strand_id: int, config: StrandConfig,
                 category: str | None = None, device: torch.device | None = None):
        super().__init__()
        self.genome = genome
        self.strand_id = strand_id
        self.config = config
        self.category: str | None = category
        self.norm = nn.LayerNorm(config.hidden_size, device=device)
        self.usage_count: int = 0

    def forward(
        self,
        x: Tensor,
        weights: dict[str, tuple[Tensor, Tensor]] | None = None,
        init_state: Tensor | None = None,
        return_final_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        B, T, H = x.shape
        S = self.config.state_size
        x = self.norm(x)

        if weights is None:
            weights = self.genome.generate_all(self.strand_id)
        W_gate, b_gate = weights["gate"]
        W_state, b_state = weights["state"]
        W_rec, b_rec = weights["recurrent"]
        W_out, b_out = weights["output"]

        gate_input = x @ W_gate + b_gate
        state_input = x @ W_state + b_state

        init_state = init_state if init_state is not None else torch.zeros(B, S, device=x.device, dtype=x.dtype)
        all_states = _strand_scan(gate_input, state_input, W_rec, b_rec, init_state)

        self.usage_count += B * T
        out = all_states @ W_out + b_out

        if return_final_state:
            return out, all_states[:, -1, :]
        return out


class NeuralMesh(nn.Module):
    """Sparse routing mesh. Each token processed by only top_k Strands.

    True sparse execution: only k strands compute per token, not all N.
    This gives N/k × compute savings (e.g., 100 strands, top-3 = 33×).
    """

    def __init__(self, genome: Genome, config: MeshConfig, layer_offset: int = 0):
        super().__init__()
        self.config = config
        H = config.strand.hidden_size
        self.router = nn.Linear(H, config.num_strands, bias=False)

        self.strands = nn.ModuleList([
            _make_strand(genome, strand_id=layer_offset + i, config=config.strand)
            for i in range(config.num_strands)
        ])

        self.register_buffer("_usage_counts", torch.zeros(config.num_strands), persistent=False)
        self.register_buffer("_last_balance_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("_last_router_entropy", torch.tensor(0.0), persistent=False)
        self._last_top_indices = None
        self._last_top_weights = None
        self._cached_states: list[Tensor | None] = [None] * config.num_strands

    @property
    def strand_ids(self) -> list[int]:
        return [s.strand_id for s in self.strands]

    @property
    def num_strands(self) -> int:
        return len(self.strands)

    def add_strand(self, strand_id: int) -> None:
        """Add one routed Strand and expand router in place."""
        H = self.config.strand.hidden_size
        old_router = self.router
        old_n = len(self.strands)
        new_n = old_n + 1

        new_router = nn.Linear(H, new_n, bias=False, device=old_router.weight.device, dtype=old_router.weight.dtype)
        with torch.no_grad():
            new_router.weight[:old_n].copy_(old_router.weight)
            scale = old_router.weight.detach().std().clamp_min(1e-5)
            new_router.weight[old_n].normal_(mean=0.0, std=float(scale))

        ref_device = self.strands[0].norm.weight.device if self.strands else old_router.weight.device
        self.router = new_router
        self.strands.append(_make_strand(self.strands[0].genome, strand_id=strand_id, config=self.config.strand, device=ref_device))

        old_counts = self._usage_counts
        new_counts = torch.zeros(new_n, device=old_counts.device, dtype=old_counts.dtype)
        new_counts[:old_n].copy_(old_counts)
        self.register_buffer("_usage_counts", new_counts, persistent=False)
        self.config.num_strands = new_n
        self._cached_states.append(None)
        logger.info("NeuralMesh: added strand %d (%d -> %d)", strand_id, old_n, new_n)

    def forward(self, x: Tensor, strand_states: list[Tensor | None] | None = None) -> tuple[Tensor, Tensor]:
        B, T, H = x.shape
        N = self.num_strands
        K = max(1, min(self.config.top_k, N))

        top_weights, top_indices, assignment, scores = _sparse_route(x, self.router, K)
        self._last_top_indices = top_indices.detach()
        self._last_top_weights = top_weights.detach()

        output = torch.zeros_like(x)
        flat_x = x.reshape(B * T, H)

        # Pre-generate all strand weights once per forward
        weight_cache = {}
        for s_id in assignment:
            strand = self.strands[s_id]
            weight_cache[s_id] = strand.genome.generate_all(strand.strand_id)

        # True sparse: only compute strands that were selected
        for s_id, (positions, w) in assignment.items():
            strand = self.strands[s_id]
            init_state = strand_states[s_id] if strand_states is not None and strand_states[s_id] is not None else None
            x_s = flat_x.index_select(0, positions.long()).unsqueeze(1)
            out_s, final_state = strand(x_s, weights=weight_cache[s_id], init_state=init_state, return_final_state=True)
            if strand_states is not None:
                # Reduce (M, S) to (1, S): take state of the last-assigned token
                strand_states[s_id] = final_state[positions.argmax():positions.argmax()+1]
            out_s = out_s.squeeze(1)
            w_b = w.view(-1, 1)
            output.view(B * T, H).index_add_(0, positions.long(), (out_s * w_b).to(output.dtype))

        with torch.no_grad():
            usage = top_indices.flatten().bincount(minlength=N)
            self._usage_counts.copy_(self._usage_counts + usage.float())

        f_i = top_indices.flatten().bincount(minlength=N).float() / max(1, B * T * K)
        router_probs = scores.softmax(dim=-1).mean(dim=(0, 1))
        balance_loss = N * (f_i * router_probs).sum()
        entropy = -(router_probs * router_probs.clamp_min(1e-9).log()).sum()
        entropy = entropy / torch.log(torch.tensor(max(1.0, float(N)), device=x.device))
        self._last_balance_loss.copy_(balance_loss.detach())
        self._last_router_entropy.copy_(entropy.detach())

        return output, balance_loss * self.config.balance_weight

    def reset_cache(self) -> None:
        self._cached_states = [None] * self.num_strands

    def record_activation_topic(self, topic: str) -> None:
        """Record which topic is currently active for strand specialization tracking."""
        pass

    @property
    def usage_stats(self) -> dict[int, float]:
        total = self._usage_counts.sum().item()
        if total == 0:
            return {i: 0.0 for i in range(len(self.strands))}
        return {i: float(c.item() / total) for i, c in enumerate(self._usage_counts)}

    @property
    def last_balance_loss(self) -> float:
        return float(self._last_balance_loss.item())

    @property
    def last_router_entropy(self) -> float:
        return float(self._last_router_entropy.item())

    def reset_usage(self) -> None:
        self._usage_counts.zero_()

    def specialization_report(self) -> dict:
        return {}


class CategoryMesh(nn.Module):
    """Category-fixed expert routing with sparse execution."""

    def __init__(self, genome: Genome, config: MeshConfig, categories: list[tuple[str, int]], layer_offset: int = 0):
        super().__init__()
        self.config = config
        self.categories = categories
        H = config.strand.hidden_size

        self.category_names = [c[0] for c in categories]
        self.category_counts = [c[1] for c in categories]
        num_categories = len(categories)
        num_strands = sum(c[1] for c in categories)

        self.router = nn.Linear(H, num_categories, bias=False)

        self.category_to_strand_id: dict[str, list[int]] = {}
        self.strand_to_category: dict[int, str] = {}
        strand_offset = layer_offset
        for cat_name, cat_count in categories:
            strand_ids = list(range(strand_offset, strand_offset + cat_count))
            self.category_to_strand_id[cat_name] = strand_ids
            for sid in strand_ids:
                self.strand_to_category[sid] = cat_name
            strand_offset += cat_count

        self.strands = nn.ModuleList()
        for cat_name, cat_count in categories:
            for i in range(cat_count):
                sid = self.category_to_strand_id[cat_name][i]
                self.strands.append(_make_strand(genome, strand_id=sid, config=config.strand, category=cat_name))

        self._num_strands = num_strands
        self.register_buffer("_usage_counts", torch.zeros(num_strands), persistent=False)
        self.register_buffer("_last_balance_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("_last_router_entropy", torch.tensor(0.0), persistent=False)
        self._last_top_indices = None
        self._last_top_weights = None

    @property
    def strand_ids(self) -> list[int]:
        return [s.strand_id for s in self.strands]

    @property
    def num_strands(self) -> int:
        return len(self.strands)

    def forward(self, x: Tensor, category_filter: str | None = None) -> tuple[Tensor, Tensor]:
        B, T, H = x.shape
        num_cats = len(self.categories)
        K = min(self.config.top_k, num_cats)
        N = self.num_strands

        if category_filter and category_filter in self.category_to_strand_id:
            strand_ids = self.category_to_strand_id[category_filter]
            scores = self.router(x)
            cat_idx = self.category_names.index(category_filter)
            num_cat_strands = len(strand_ids)

            weight_cache = {sid: self.strands[self._strand_idx_by_id(sid)].genome.generate_all(sid) for sid in strand_ids}

            output = torch.zeros_like(x)
            for sid in strand_ids:
                strand_idx = self._strand_idx_by_id(sid)
                if strand_idx is None:
                    continue
                out_s = self.strands[strand_idx](x, weights=weight_cache.get(sid))
                self._usage_counts[sid] += B * T
                output = output + out_s / num_cat_strands

            f_i = torch.zeros(N, device=x.device)
            f_i[strand_ids] = 1.0 / len(strand_ids)
            probs_mean = torch.softmax(scores, dim=-1).mean(dim=(0, 1))
            strand_probs = torch.zeros(N, device=x.device)
            for j, (cat_name, _) in enumerate(self.categories):
                for sid in self.category_to_strand_id[cat_name]:
                    strand_probs[sid] = probs_mean[j] / len(self.category_to_strand_id[cat_name])
            balance_loss = N * (f_i * strand_probs).sum()
            self._last_balance_loss.copy_(balance_loss.detach())
            return output, balance_loss * self.config.balance_weight

        scores = self.router(x)
        top_weights, top_indices = torch.topk(scores, K, dim=-1)
        top_weights = torch.softmax(top_weights, dim=-1)

        flat_indices = top_indices.reshape(B * T * K)
        flat_weights = top_weights.reshape(B * T * K, 1)
        flat_x = x.unsqueeze(2).expand(-1, -1, K, -1).reshape(B * T * K, H)

        unique_cats = torch.unique(flat_indices)
        unique_cat_names = [self.category_names[int(c)] for c in unique_cats]

        weight_cache = {}
        for cat_name in unique_cat_names:
            for sid in self.category_to_strand_id[cat_name]:
                strand_idx = self._strand_idx_by_id(sid)
                if strand_idx is not None:
                    weight_cache[sid] = self.strands[strand_idx].genome.generate_all(self.strands[strand_idx].strand_id)

        output = torch.zeros_like(x)
        for cat_idx in unique_cats:
            cat_name = self.category_names[int(cat_idx)]
            strand_ids = self.category_to_strand_id[cat_name]
            num_cat_strands = len(strand_ids)

            cat_mask = flat_indices == cat_idx
            if cat_mask.sum() == 0:
                continue

            for i, sid in enumerate(strand_ids):
                strand_idx = self._strand_idx_by_id(sid)
                if strand_idx is None:
                    continue

                indices_s = torch.where(cat_mask)[0]
                w_s = flat_weights.index_select(0, indices_s) / num_cat_strands
                x_s = flat_x.index_select(0, indices_s).unsqueeze(1)

                out_s = self.strands[strand_idx](x_s, weights=weight_cache.get(sid)).squeeze(1)
                self._usage_counts[sid] += indices_s.shape[0]

                remainder = indices_s % (T * K)
                batch_i = indices_s // (T * K)
                time_i = remainder // K
                output = output.index_put((batch_i, time_i), out_s * w_s, accumulate=True)

        routing_probs_all = torch.softmax(scores, dim=-1)
        probs_mean = routing_probs_all.mean(dim=(0, 1))

        f_i = torch.zeros(N, device=x.device)
        cat_counts = flat_indices.bincount(minlength=num_cats).float()
        cat_total = cat_counts.sum().clamp_min(1.0)
        for j, (cat_name, _) in enumerate(self.categories):
            cat_strand_ids = self.category_to_strand_id[cat_name]
            cat_weight = cat_counts[j] / cat_total
            for sid in cat_strand_ids:
                f_i[sid] = cat_weight / len(cat_strand_ids)

        strand_probs = torch.zeros(N, device=x.device)
        for j, (cat_name, _) in enumerate(self.categories):
            for sid in self.category_to_strand_id[cat_name]:
                strand_probs[sid] = probs_mean[j] / len(self.category_to_strand_id[cat_name])

        balance_loss = N * (f_i * strand_probs).sum()
        entropy = -(probs_mean * probs_mean.clamp_min(1e-9).log()).sum()
        entropy = entropy / torch.log(torch.tensor(max(1.0, float(num_cats)), device=x.device))
        self._last_balance_loss.copy_(balance_loss.detach())
        self._last_router_entropy.copy_(entropy.detach())

        return output, balance_loss * self.config.balance_weight

    def reset_cache(self) -> None:
        pass

    def record_activation_topic(self, topic: str) -> None:
        pass

    def _strand_idx_by_id(self, strand_id: int) -> int | None:
        for i, s in enumerate(self.strands):
            if s.strand_id == strand_id:
                return i
        return None

    @property
    def usage_stats(self) -> dict[int, float]:
        total = self._usage_counts.sum().item()
        if total == 0:
            return {i: 0.0 for i in range(self.num_strands)}
        return {i: self._usage_counts[i].item() / total for i in range(self.num_strands)}

    @property
    def last_balance_loss(self) -> float:
        return float(self._last_balance_loss.item())

    @property
    def last_router_entropy(self) -> float:
        return float(self._last_router_entropy.item())

    def reset_usage(self) -> None:
        self._usage_counts.zero_()

    def specialization_report(self) -> dict:
        report = {}
        for cat_name, count in self.categories:
            strand_ids = self.category_to_strand_id[cat_name]
            usages = {sid: float(self._usage_counts[sid].item()) for sid in strand_ids}
            total = sum(usages.values()) or 1.0
            report[cat_name] = {
                "strand_ids": strand_ids,
                "usage_share": {sid: u/total for sid, u in usages.items()},
                "total_usage": total,
            }
        return report

    def add_strand(self, strand_id: int) -> None:
        H = self.config.strand.hidden_size
        old_router = self.router
        n_cats = len(self.categories)
        new_router = nn.Linear(H, n_cats, bias=False, device=old_router.weight.device)
        with torch.no_grad():
            new_router.weight.copy_(old_router.weight)
        self.router = new_router

        ref_device = self.strands[0].norm.weight.device if self.strands else None
        old_n = len(self.strands)
        self.strands.append(Strand(self.strands[0].genome, strand_id=strand_id, config=self.config.strand, device=ref_device))
        new_n = len(self.strands)

        old_counts = self._usage_counts
        new_counts = torch.zeros(new_n, device=old_counts.device)
        new_counts[:old_n].copy_(old_counts)
        self.register_buffer("_usage_counts", new_counts, persistent=False)
        logger.info("CategoryMesh: added strand %d (%d -> %d)", strand_id, old_n, new_n)
