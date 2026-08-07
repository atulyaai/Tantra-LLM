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

    def __init__(self, plateau_patience: int = 50, min_delta: float = 0.005):
        self.plateau_patience = plateau_patience
        self.min_delta = min_delta
        self.loss_history: List[float] = []
        self.growth_events: List[Dict[str, Any]] = []

    def observe(self, loss: float, model: nn.Module) -> bool:
        """Observe step loss. Returns True if capacity growth was triggered."""
        self.loss_history.append(loss)
        
        if len(self.loss_history) < self.plateau_patience:
            return False

        recent = self.loss_history[-self.plateau_patience :]
        window_start = sum(recent[:10]) / 10.0
        window_end = sum(recent[-10:]) / 10.0
        improvement = window_start - window_end

        if improvement < self.min_delta:
            log.info(f"Loss plateau detected (improvement: {improvement:.5f} < {self.min_delta}). Triggering auto-growth...")
            self.grow_capacity(model)
            self.loss_history.clear()
            return True

        return False

    def grow_capacity(self, model: nn.Module) -> None:
        """Dynamically add capacity to model layers or experts."""
        if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList) and len(model.layers) > 0:
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

            # 2. Repair Exploded Weights
            norms = torch.norm(param.data, p=2)
            if not torch.isnan(norms) and not torch.isinf(norms) and norms > max_norm:
                param.data.mul_(max_norm / (norms + 1e-6))
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
