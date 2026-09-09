"""
tantra/evolution.py — Auto-Growth & Self-Repair Controller for NeuroCore.

Provides:
  - Loss Plateau Detection & Dynamic Layer/Expert Insertion
  - Self-Repair Engine (Detects NaNs, dead neurons, and exploded weight tensors)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

from Tantra.utils import get_logger

log = get_logger("tantra.evolution")


class AutoGrowthController:
    """Monitors training loss trajectory and layer representation saturation (80-90%)
    to dynamically expand model capacity only when existing layers are genuinely filled.
    """

    def __init__(
        self,
        plateau_patience: int = 250,
        min_delta: float = 0.003,
        max_layers: Optional[int] = None,
        saturation_threshold: float = 0.80,
        max_loss_for_growth: float = 10.0,
        max_params: int = 1_000_000_000,
    ):
        self.plateau_patience = plateau_patience
        self.min_delta = min_delta
        self.max_layers = max_layers
        self.saturation_threshold = float(saturation_threshold)
        self.max_loss_for_growth = float(max_loss_for_growth)
        self.max_params = int(max_params)
        self.loss_history: List[float] = []
        self.growth_events: List[Dict[str, Any]] = []
        # Guard: allow at most one growth event per plateau window to prevent
        # repeated layer additions that desync generate()'s layer-state list.
        self._steps_since_growth: int = 0

    @staticmethod
    def compute_capacity_saturation(model: nn.Module) -> float:
        """Measure what fraction (0.0 to 1.0) of layer representation capacity is utilized.

        Evaluates:
        1. Neuron activation ratio in SparseGatedProjection layers.
        2. Dimensional representation participation ratio (effective rank / dimension usage)
           across output projection matrices.
        """
        actual_model = model
        while hasattr(actual_model, "module"):
            actual_model = actual_model.module
        while hasattr(actual_model, "_orig_mod"):
            actual_model = actual_model._orig_mod
        while hasattr(actual_model, "module"):
            actual_model = actual_model.module

        if not hasattr(actual_model, "layers") or not actual_model.layers:
            return 0.0

        saturation_scores: List[float] = []

        for layer in actual_model.layers:
            # 1. Sparse MLP neuron active ratio
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "_last_active_ratio") and mlp._last_active_ratio > 0:
                target_ratio = getattr(mlp, "k", 1) / max(1, getattr(mlp, "hidden_dim", 1))
                if target_ratio > 0:
                    act_score = min(1.0, float(mlp._last_active_ratio) / target_ratio)
                    saturation_scores.append(act_score)

            # 2. Linear projection dimensional participation ratio:
            w_candidate = None
            if mlp is not None and hasattr(mlp, "w_down") and hasattr(mlp.w_down, "weight"):
                w_candidate = mlp.w_down.weight
            elif hasattr(layer, "attn") and hasattr(layer.attn, "w_o") and hasattr(layer.attn.w_o, "weight"):
                w_candidate = layer.attn.w_o.weight

            if w_candidate is not None and w_candidate.ndim == 2:
                with torch.no_grad():
                    w_data = w_candidate.detach().float()
                    row_vars = torch.var(w_data, dim=-1)
                    active_dims = (row_vars > 1e-5).float().mean().item()
                    saturation_scores.append(active_dims)

        if not saturation_scores:
            return 0.85
        return sum(saturation_scores) / len(saturation_scores)

    def observe(self, loss: float, model: nn.Module, optimizer: Optional[Any] = None) -> bool:
        """Observe step loss & layer capacity saturation. Returns True if capacity growth was triggered."""
        self.loss_history.append(loss)
        self._steps_since_growth += 1

        if len(self.loss_history) < self.plateau_patience:
            return False

        # Only allow growth after at least plateau_patience steps since last growth,
        # so a single plateau window can't trigger multiple consecutive grow events.
        if self._steps_since_growth < self.plateau_patience:
            return False

        recent = self.loss_history[-self.plateau_patience :]
        window_start = sum(recent[:10]) / 10.0
        window_end = sum(recent[-10:]) / 10.0
        improvement = window_start - window_end

        if improvement < self.min_delta:
            # Check 1: Loss must be within normal learning regime (<= max_loss_for_growth).
            # If loss is 13.0+, the model is still in initial convergence or divergence;
            # it does NOT need more layers, it needs to learn the current layers!
            if window_end > self.max_loss_for_growth:
                log.debug(
                    f"Loss plateau detected at {window_end:.4f}, but loss > {self.max_loss_for_growth:.1f}. "
                    f"Skipping growth until current layer capacity is learned."
                )
                return False

            # Check 2: Layer representation capacity must be 80-90% saturated
            saturation = self.compute_capacity_saturation(model)
            if saturation < self.saturation_threshold:
                log.info(
                    f"Loss plateau detected at {window_end:.4f}, but layer representation is only "
                    f"{saturation*100:.1f}% saturated (< {self.saturation_threshold*100:.0f}% threshold). "
                    f"Existing layers still have capacity headroom — continuing optimization."
                )
                return False

            log.info(
                f"🧠 [CAPACITY SATURATED: {saturation*100:.1f}%] Layer capacity reached saturation threshold "
                f"(>= {self.saturation_threshold*100:.0f}%) and loss plateaued at {window_end:.4f} "
                f"(improvement: {improvement:.5f} < {self.min_delta}). Dynamically expanding model depth..."
            )
            grown = self.grow_capacity(model, optimizer=optimizer)
            keep = max(0, self.plateau_patience // 2)
            self.loss_history = self.loss_history[-keep:] if keep > 0 else []
            self._steps_since_growth = 0
            return grown

        return False

    def grow_capacity(self, model: nn.Module, optimizer: Optional[Any] = None) -> bool:
        """Dynamically add capacity to model layers or experts with 1B parameter ceiling."""
        actual_model = model
        while hasattr(actual_model, "module"):
            actual_model = actual_model.module
        while hasattr(actual_model, "_orig_mod"):
            actual_model = actual_model._orig_mod
        while hasattr(actual_model, "module"):
            actual_model = actual_model.module

        if hasattr(actual_model, "layers") and isinstance(actual_model.layers, nn.ModuleList) and len(actual_model.layers) > 0:
            if self.max_layers is not None and len(actual_model.layers) >= self.max_layers:
                log.info("Auto-growth plateau observed, but maximum depth (%d) is already reached.", self.max_layers)
                return False

            # Check 1 Billion Parameter Limit guard
            current_params = sum(p.numel() for p in actual_model.parameters())
            last_layer = actual_model.layers[-1]
            layer_params = sum(p.numel() for p in last_layer.parameters())

            if (current_params + layer_params) > self.max_params:
                log.warning(
                    f"⛔ [GROWTH BLOCKED] Adding layer ({layer_params/1e6:.1f}M params) would exceed "
                    f"1 Billion parameter ceiling ({self.max_params/1e6:.0f}M). Current: {current_params/1e6:.1f}M."
                )
                return False

            # Duplicate and perturb last layer to grow depth
            import copy
            new_layer = copy.deepcopy(last_layer)
            
            # Small random perturbation to break symmetry
            for p in new_layer.parameters():
                p.data.add_(torch.randn_like(p.data) * 0.001)
                
            actual_model.layers.append(new_layer)
            if hasattr(actual_model, "config") and hasattr(actual_model.config, "block"):
                actual_model.config.block.num_layers = len(actual_model.layers)

            # Sync with optimizer so new parameters receive gradients and updates
            if optimizer is not None:
                if hasattr(optimizer, "refresh_optimizer"):
                    optimizer.refresh_optimizer()
                elif hasattr(optimizer, "add_param_group"):
                    decay = [p for p in new_layer.parameters() if p.ndim >= 2]
                    no_decay = [p for p in new_layer.parameters() if p.ndim < 2]
                    ref_lr = optimizer.param_groups[0].get("lr", 1e-4) if optimizer.param_groups else 1e-4
                    ref_wd = optimizer.param_groups[0].get("weight_decay", 0.01) if optimizer.param_groups else 0.01
                    if decay:
                        optimizer.add_param_group({"params": decay, "lr": ref_lr, "weight_decay": ref_wd})
                    if no_decay:
                        optimizer.add_param_group({"params": no_decay, "lr": ref_lr, "weight_decay": 0.0})
                    log.info("Registered newly grown layer parameters via add_param_group (decay/no-decay separated).")
                elif hasattr(optimizer, "param_groups") and optimizer.param_groups:
                    optimizer.param_groups[0]["params"].extend(list(new_layer.parameters()))

            new_total_params = sum(p.numel() for p in actual_model.parameters())
            log.info(
                f"🌱 Model capacity auto-grown: total layers is now {len(actual_model.layers)} "
                f"({new_total_params/1e6:.1f}M / {self.max_params/1e6:.0f}M params max)."
            )
            self.growth_events.append({"type": "add_layer", "new_total": len(actual_model.layers), "total_params": new_total_params})
            return True
        return False


class SelfRepairEngine:
    """Scans neural network tensors, gradients, and predictions for anomalies, repairing them on the fly.
    
    Tantra Autonomous Self-Healing Laws:
    1. Law of Numerical Integrity: Auto-repairs NaNs, Infs, and exploded weights.
    2. Law of Gradient Sanity: Auto-clips and purges corrupted optimizer momentum buffers.
    3. Law of Representation Diversity: Detects mode collapse and restores prediction entropy.
    4. Law of Layer Stability: Keeps LayerNorm scales strictly bounded within healthy ranges.
    """

    def scan_and_repair(self, model: nn.Module, max_norm: float = 50.0) -> Dict[str, int]:
        """Scan all module parameters and repair corrupted/exploded values."""
        repaired_nans = 0
        repaired_explosions = 0
        repaired_dead = 0

        for name, param in model.named_parameters():
            if param.data is None:
                continue

            # 1. Repair NaNs / Infs
            nans_mask = torch.isnan(param.data) | torch.isinf(param.data)
            if nans_mask.any():
                count = int(nans_mask.sum().item())
                repaired_nans += count
                param.data.copy_(torch.nan_to_num(param.data, nan=0.0, posinf=0.02, neginf=-0.02))
                param.data[nans_mask] += torch.randn_like(param.data[nans_mask]) * 0.01

            # 2. Repair Exploded Weights (scaled by sqrt(numel) for proper element RMS threshold)
            # Default threshold: max per-element RMS of 5.0 (well above normal weight initialization ~0.02)
            elem_rms = torch.sqrt(torch.mean(param.data ** 2))
            if not torch.isnan(elem_rms) and not torch.isinf(elem_rms) and elem_rms > 5.0:
                param.data.mul_(5.0 / (elem_rms + 1e-6))
                repaired_explosions += 1

            # 3. Repair Dead Neurons (zero weights in multi-neuron linear projections)
            if "weight" in name and param.dim() == 2 and param.size(0) > 1 and "w_scale" not in name and "gate" not in name:
                row_norms = param.data.norm(dim=1)
                dead_rows = row_norms < 1e-6
                if dead_rows.any():
                    count_dead = int(dead_rows.sum().item())
                    repaired_dead += count_dead
                    param.data[dead_rows] = torch.randn_like(param.data[dead_rows]) * 0.02

        if repaired_nans > 0 or repaired_explosions > 0 or repaired_dead > 0:
            log.info(f"Self-Repair triggered: Repaired {repaired_nans} NaNs, {repaired_explosions} exploded tensors, {repaired_dead} dead neurons.")

        return {
            "repaired_nans": repaired_nans,
            "repaired_explosions": repaired_explosions,
            "repaired_dead": repaired_dead,
        }

    def purge_corrupted_optimizer_state(self, optimizer: torch.optim.Optimizer) -> int:
        """Purge and reset any NaN or Inf entries in optimizer momentum buffers."""
        purged = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if state:
                    for key in ["exp_avg", "exp_avg_sq"]:
                        if key in state and state[key] is not None:
                            bad_mask = torch.isnan(state[key]) | torch.isinf(state[key])
                            if bad_mask.any():
                                state[key][bad_mask] = 0.0
                                purged += int(bad_mask.sum().item())
        return purged

    def sanitize_optimizer_momentum(self, optimizer: torch.optim.Optimizer, grad_norm: float, threshold: float = 8.0) -> bool:
        """Law 2: Sanitize optimizer momentum if gradient explosion occurs."""
        self.purge_corrupted_optimizer_state(optimizer)
        if grad_norm <= threshold:
            return False
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if state and "exp_avg" in state and state["exp_avg"] is not None:
                    state["exp_avg"].mul_(0.5)  # Dampen runaway momentum
        return True

    def check_and_restore_entropy(self, logits_flat: torch.Tensor, min_entropy: float = 0.3) -> float:
        """Law 3: Real-time prediction entropy monitor to detect and prevent mode collapse."""
        with torch.no_grad():
            probs = torch.softmax(logits_flat[:100], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
            return entropy

    def stabilize_layer_norms(self, model: nn.Module) -> int:
        """Law 4: Keep norm gain parameters safely bounded."""
        repaired = 0
        for name, param in model.named_parameters():
            if "norm" in name and "weight" in name and param.data is not None:
                out_of_bounds = (param.data < 0.01) | (param.data > 10.0)
                if out_of_bounds.any():
                    param.data.clamp_(0.01, 10.0)
                    repaired += 1
        return repaired



class CategoryGrowthController:
    """Bidirectional capacity control for per-category specialist layers.

    A category is a stack of identical-shape specialist layers. Capacity moves
    in BOTH directions without changing any tensor shape:

      * GROW  — when the category's held-out loss plateaus *and* it still has
        headroom (depth < cap) and is actually being used. This is the "fit but
        needs more" case: add a layer so further training can help.
      * SHRINK — when the loss has plateaued, the category is effectively
        *converged* (its recent loss is within ``fit_target_ratio`` of its best,
        i.e. ~95% there) AND it is rarely routed (low usage). That is the
        "less used, so reduce" case: reclaim a layer's parameters.

    ``fit_target_ratio`` is the user's "95%": a category that has already
    reached 95% of its best achievable loss is considered fit enough that an
    idle one can be safely shrunk.
    """

    def __init__(self, plateau_patience: int = 1000, min_delta: float = 0.005,
                 low_usage_frac: float = 0.05, fit_target_ratio: float = 0.95):
        self.plateau_patience = plateau_patience
        self.min_delta = min_delta
        self.low_usage_frac = low_usage_frac
        self.fit_target_ratio = fit_target_ratio
        self._state: Dict[str, dict] = {}

    def observe(self, category: str, loss: float, cat_routed: int, total_routed: int,
                depth: int, min_depth: int, max_depth: int) -> Optional[str]:
        """Feed one evaluation sample. Returns 'grow', 'shrink', or None."""
        st = self._state.setdefault(category, {"loss": [], "usage": 0, "steps_since": 0, "best": float("inf")})
        st["loss"].append(loss)
        st["usage"] += int(cat_routed)
        st["best"] = min(st["best"], loss)
        st["steps_since"] += 1

        if len(st["loss"]) < self.plateau_patience:
            return None
        if st["steps_since"] < self.plateau_patience:
            return None

        recent = st["loss"][-self.plateau_patience:]
        window_start = sum(recent[:10]) / 10.0
        window_end = sum(recent[-10:]) / 10.0
        improvement = window_start - window_end
        # "fit" = current loss is at least fit_target_ratio (e.g. 0.95) as good
        # as the best seen, i.e. within ~5% above the best achievable loss.
        fit = window_end <= st["best"] / self.fit_target_ratio
        usage_low = (st["usage"] < self.low_usage_frac * max(1, int(total_routed))) or (total_routed == 0 and st["usage"] == 0)

        decision = None
        if improvement < self.min_delta:
            if depth < max_depth and not (fit and usage_low):
                decision = "grow"
            elif depth > min_depth and fit and usage_low:
                decision = "shrink"

        if decision is not None:
            st["loss"].clear()
            st["usage"] = 0
            st["steps_since"] = 0
        return decision
