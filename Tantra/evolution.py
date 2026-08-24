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
    """Monitors training loss trajectory and dynamically grows model capacity."""

    def __init__(self, plateau_patience: int = 1000, min_delta: float = 0.005, max_layers: Optional[int] = None):
        self.plateau_patience = plateau_patience
        self.min_delta = min_delta
        self.loss_history: List[float] = []
        self.growth_events: List[Dict[str, Any]] = []
        # Guard: allow at most one growth event per plateau window to prevent
        # repeated layer additions that desync generate()'s layer-state list.
        self._steps_since_growth: int = 0
        self.max_layers = max_layers

    def observe(self, loss: float, model: nn.Module) -> bool:
        """Observe step loss. Returns True if capacity growth was triggered."""
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
            log.info(f"Loss plateau detected (improvement: {improvement:.5f} < {self.min_delta}). Triggering auto-growth...")
            if not self.grow_capacity(model):
                return False
            self.loss_history.clear()
            self._steps_since_growth = 0
            return True

        return False

    def grow_capacity(self, model: nn.Module) -> bool:
        """Dynamically add capacity to model layers or experts."""
        if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList) and len(model.layers) > 0:
            if self.max_layers is not None and len(model.layers) >= self.max_layers:
                log.info("Auto-growth plateau observed, but maximum depth (%d) is already reached.", self.max_layers)
                return False
            # Duplicate and perturb last layer to grow depth
            import copy
            last_layer = model.layers[-1]
            new_layer = copy.deepcopy(last_layer)
            
            # Small random perturbation to break symmetry
            for p in new_layer.parameters():
                p.data.add_(torch.randn_like(p.data) * 0.001)
                
            model.layers.append(new_layer)
            log.info(f"Model capacity auto-grown: total layers is now {len(model.layers)}")
            self.growth_events.append({"type": "add_layer", "new_total": len(model.layers)})
            return True
        return False


class SelfRepairEngine:
    """Scans neural network tensors for NaNs, numerical explosions, or dead neurons, repairing them on the fly."""

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
                # Reset corrupted entries with small normal noise
                param.data[nans_mask] = torch.randn_like(param.data[nans_mask]) * 0.01

            # 2. Repair Exploded Weights (scaled by sqrt(numel) for proper element RMS threshold)
            # Default threshold: max per-element RMS of 5.0 (well above normal weight initialization ~0.02)
            elem_rms = torch.sqrt(torch.mean(param.data ** 2))
            if not torch.isnan(elem_rms) and not torch.isinf(elem_rms) and elem_rms > 5.0:
                param.data.mul_(5.0 / (elem_rms + 1e-6))
                repaired_explosions += 1

            # 3. Repair Dead Neurons (zero weights in linear layers)
            if "weight" in name and param.dim() == 2:
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
