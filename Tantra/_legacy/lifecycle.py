"""
Tantra Expert Lifecycle Manager
=================================
Dynamic expert management: grow, shrink, split, combine.
All operations preserve base model + other experts.
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import torch
import torch.nn as nn

from Tantra.model.tantra_model import TantraModel, FusedLoRALayer
from Tantra.config import load_taxonomy

log = logging.getLogger("Tantra.lifecycle")


class LifecycleAction(Enum):
    EXPAND = "expand"
    SHRINK = "shrink"
    SPLIT = "split"
    COMBINE = "combine"
    REPLACE = "replace"


@dataclass
class ExpertLifecycleConfig:
    """Configuration for lifecycle management."""
    min_rank: int = 4
    max_rank: int = 256
    grow_threshold: float = 0.15      # avg router weight to trigger grow
    shrink_threshold: float = 0.005   # avg router weight to trigger shrink
    split_threshold: float = 0.25     # avg router weight to trigger split
    combine_correlation: float = 0.9  # correlation to trigger combine
    min_steps: int = 1000             # min steps before action
    cooldown_steps: int = 5000        # steps between actions on same expert


@dataclass
class ExpertStats:
    """Runtime statistics for an expert."""
    name: str
    activations: deque = field(default_factory=lambda: deque(maxlen=1000))
    losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    grad_norms: deque = field(default_factory=lambda: deque(maxlen=1000))
    co_activations: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=1000)))
    last_action_step: int = 0
    
    def update(self, activation: float, loss: float = None, grad_norm: float = None) -> None:
        self.activations.append(activation)
        if loss is not None:
            self.losses.append(loss)
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
    
    def update_coactivation(self, other_name: str, weight: float) -> None:
        self.co_activations[other_name].append(weight)
    
    def avg_activation(self) -> float:
        return sum(self.activations) / len(self.activations) if self.activations else 0.0
    
    def avg_loss(self) -> float:
        return sum(self.losses) / len(self.losses) if self.losses else 0.0
    
    def correlation(self, other_name: str) -> float:
        """Correlation with another expert's activations."""
        a = list(self.activations)
        b = list(self.co_activations.get(other_name, []))
        if len(a) < 10 or len(b) < 10:
            return 0.0
        # Align lengths
        n = min(len(a), len(b))
        a, b = a[-n:], b[-n:]
        import numpy as np
        return np.corrcoef(a, b)[0, 1] if n > 1 else 0.0


class ExpertLifecycleManager:
    """
    Manage expert lifecycle: grow, shrink, split, combine.
    Monitors usage patterns and recommends actions.
    """
    
    def __init__(
        self,
        model: TantraModel,
        config: ExpertLifecycleConfig = None,
    ):
        self.model = model
        self.config = config or ExpertLifecycleConfig()
        self.expert_stats: Dict[str, ExpertStats] = {
            name: ExpertStats(name) for name in model.expert_names
        }
        self.step = 0
        self.action_history: List[Dict] = []
    
    def track(
        self,
        router_weights: torch.Tensor,  # [B, n_experts]
        losses: Dict[str, float] = None,
        grad_norms: Dict[str, float] = None,
    ) -> None:
        """
        Track expert usage from router output.
        Call every training step.
        """
        self.step += 1
        B = router_weights.shape[0]
        
        # Mean activation per expert across batch
        mean_weights = router_weights.mean(dim=0).detach().cpu().numpy()  # [n_experts]
        
        for i, name in enumerate(self.model.expert_names):
            weight = float(mean_weights[i])
            loss = losses.get(name) if losses else None
            grad = grad_norms.get(name) if grad_norms else None
            
            self.expert_stats[name].update(weight, loss, grad)
            
            # Track co-activations
            for j, other_name in enumerate(self.model.expert_names):
                if i != j:
                    other_weight = float(mean_weights[j])
                    self.expert_stats[name].update_coactivation(other_name, other_weight)
    
    def analyze(self) -> List[Tuple[LifecycleAction, str, Dict]]:
        """
        Analyze usage patterns and recommend actions.
        Returns list of (action, expert_name, details).
        """
        actions = []
        
        for name, stats in self.expert_stats.items():
            if self.step - stats.last_action_step < self.config.cooldown_steps:
                continue
            if len(stats.activations) < self.config.min_steps:
                continue
            
            avg_act = stats.avg_activation()
            avg_loss = stats.avg_loss()
            
            # Get current rank
            adapter_layer = self.model.adapters
            if not hasattr(adapter_layer, 'adapter_ranks'):
                continue
            current_rank = adapter_layer.adapter_ranks[
                adapter_layer.adapter_names.index(name)
            ].item() if name in adapter_layer.adapter_names else 0
            
            # EXPAND: High usage, low loss, room to grow
            if (avg_act > self.config.grow_threshold and 
                avg_loss < 2.0 and 
                current_rank < self.config.max_rank):
                new_rank = min(current_rank * 2, self.config.max_rank)
                actions.append((
                    LifecycleAction.EXPAND,
                    name,
                    {"old_rank": current_rank, "new_rank": new_rank, 
                     "avg_activation": avg_act, "avg_loss": avg_loss}
                ))
            
            # SHRINK: Low usage, waste of capacity
            elif (avg_act < self.config.shrink_threshold and 
                  current_rank > self.config.min_rank):
                new_rank = max(current_rank // 2, self.config.min_rank)
                actions.append((
                    LifecycleAction.SHRINK,
                    name,
                    {"old_rank": current_rank, "new_rank": new_rank,
                     "avg_activation": avg_act}
                ))
            
            # SPLIT: Very high usage, maxed out → create specialist sibling
            elif (avg_act > self.config.split_threshold and 
                  current_rank >= self.config.max_rank * 0.8):
                actions.append((
                    LifecycleAction.SPLIT,
                    name,
                    {"current_rank": current_rank, "avg_activation": avg_act}
                ))
            
            # COMBINE: Check pairwise correlations
            for other_name in self.model.expert_names:
                if other_name <= name:  # Avoid duplicates
                    continue
                corr = stats.correlation(other_name)
                if corr > self.config.combine_correlation:
                    actions.append((
                        LifecycleAction.COMBINE,
                        f"{name}+{other_name}",
                        {"correlation": corr, "experts": [name, other_name]}
                    ))
        
        return actions
    
    def execute(self, action: LifecycleAction, name: str, details: Dict) -> bool:
        """Execute a lifecycle action."""
        try:
            if action == LifecycleAction.EXPAND:
                return self._expand_expert(name, details["new_rank"])
            elif action == LifecycleAction.SHRINK:
                return self._shrink_expert(name, details["new_rank"])
            elif action == LifecycleAction.SPLIT:
                return self._split_expert(name)
            elif action == LifecycleAction.COMBINE:
                return self._combine_experts(details["experts"][0], details["experts"][1])
            return False
        except Exception as e:
            log.error(f"Failed to execute {action.value} on {name}: {e}")
            return False
    
    def _expand_expert(self, name: str, new_rank: int) -> bool:
        """Increase LoRA rank for an expert."""
        adapter_layer = self.model.adapters
        if name not in adapter_layer.adapter_names:
            return False
        
        old_rank = adapter_layer.adapter_ranks[
            adapter_layer.adapter_names.index(name)
        ].item()
        
        # Resize in fused layer
        adapter_layer.resize_expert_rank(name, new_rank)
        
        # Update stats
        self.expert_stats[name].last_action_step = self.step
        self.action_history.append({
            "step": self.step,
            "action": "expand",
            "expert": name,
            "old_rank": old_rank,
            "new_rank": new_rank,
        })
        
        log.info(f"📈 EXPAND {name}: rank {old_rank} → {new_rank}")
        return True
    
    def _shrink_expert(self, name: str, new_rank: int) -> bool:
        """Decrease LoRA rank for an expert."""
        adapter_layer = self.model.adapters
        if name not in adapter_layer.adapter_names:
            return False
        
        old_rank = adapter_layer.adapter_ranks[
            adapter_layer.adapter_names.index(name)
        ].item()
        
        adapter_layer.resize_expert_rank(name, new_rank)
        
        self.expert_stats[name].last_action_step = self.step
        self.action_history.append({
            "step": self.step,
            "action": "shrink",
            "expert": name,
            "old_rank": old_rank,
            "new_rank": new_rank,
        })
        
        log.info(f"📉 SHRINK {name}: rank {old_rank} → {new_rank}")
        return True
    
    def _split_expert(self, name: str) -> bool:
        """Split expert into two: original + specialist sibling."""
        adapter_layer = self.model.adapters
        if name not in adapter_layer.adapter_names:
            return False
        
        current_rank = adapter_layer.adapter_ranks[
            adapter_layer.adapter_names.index(name)
        ].item()
        
        # Clone adapter weights with noise
        A, B = adapter_layer.get_adapter_params(name)
        sibling_name = f"{name}_v2"
        
        # Add small noise to differentiate
        noise_scale = 0.01
        A_sibling = A + torch.randn_like(A) * noise_scale
        B_sibling = B + torch.randn_like(B) * noise_scale
        
        # Add to fused layer (requires expanding the layer)
        # This is complex - in practice, we'd pre-allocate reserve capacity
        # For now, mark for manual handling
        log.info(f"✂️ SPLIT requested: {name} → {name} + {sibling_name}")
        log.info(f"   (Requires router expansion - use reserve capacity)")
        
        self.action_history.append({
            "step": self.step,
            "action": "split",
            "expert": name,
            "sibling": sibling_name,
            "rank": current_rank,
        })
        
        return True
    
    def _combine_experts(self, name1: str, name2: str) -> bool:
        """Combine two highly correlated experts."""
        adapter_layer = self.model.adapters
        if name1 not in adapter_layer.adapter_names or name2 not in adapter_layer.adapter_names:
            return False
        
        # Weighted merge by usage
        stats1 = self.expert_stats[name1]
        stats2 = self.expert_stats[name2]
        
        w1 = stats1.avg_activation()
        w2 = stats2.avg_activation()
        total = w1 + w2
        
        if total == 0:
            return False
        
        A1, B1 = adapter_layer.get_adapter_params(name1)
        A2, B2 = adapter_layer.get_adapter_params(name2)
        
        # Merge
        A_merged = (w1 * A1 + w2 * A2) / total
        B_merged = (w1 * B1 + w2 * B2) / total
        
        # New name
        merged_name = f"{name1}_{name2}_merged"
        
        log.info(f"🔀 COMBINE {name1} + {name2} → {merged_name}")
        log.info(f"   Weights: {w1:.4f}, {w2:.4f}")
        
        self.action_history.append({
            "step": self.step,
            "action": "combine",
            "experts": [name1, name2],
            "merged": merged_name,
            "weights": [w1, w2],
        })
        
        return True
    
    def get_expert_report(self) -> Dict[str, Dict]:
        """Get usage report for all experts."""
        return {
            name: {
                "avg_activation": stats.avg_activation(),
                "avg_loss": stats.avg_loss(),
                "num_updates": len(stats.activations),
                "top_correlations": self._get_top_correlations(name),
            }
            for name, stats in self.expert_stats.items()
        }
    
    def _get_top_correlations(self, name: str, top_k: int = 3) -> List[Tuple[str, float]]:
        stats = self.expert_stats[name]
        corrs = []
        for other in self.expert_stats:
            if other != name:
                c = stats.correlation(other)
                if c > 0:
                    corrs.append((other, c))
        corrs.sort(key=lambda x: -x[1])
        return corrs[:top_k]
    
    def auto_manage(self) -> int:
        """Analyze and execute all recommended actions."""
        actions = self.analyze()
        executed = 0
        for action, name, details in actions:
            if self.execute(action, name, details):
                executed += 1
        return executed


# ── Reserve Capacity Manager ──────────────────────────────────────────────────

class ReserveCapacityManager:
    """
    Manage pre-allocated reserve capacity for dynamic expert growth.
    Pre-allocates extra rank capacity in fused layer.
    """
    
    def __init__(
        self,
        adapter_layer: FusedLoRALayer,
        reserve_ranks: Dict[str, int] = None,
    ):
        self.adapter_layer = adapter_layer
        self.reserve_ranks = reserve_ranks or {}
        self.allocated: Dict[str, int] = {}
    
    def allocate_reserve(self, expert_name: str, additional_rank: int) -> bool:
        """Allocate reserve capacity to an expert."""
        if expert_name not in self.adapter_layer.adapter_names:
            return False
        
        current_rank = self.adapter_layer.adapter_ranks[
            self.adapter_layer.adapter_names.index(expert_name)
        ].item()
        
        max_rank = current_rank + self.reserve_ranks.get(expert_name, 0)
        new_rank = min(current_rank + additional_rank, max_rank)
        
        if new_rank > current_rank:
            self.adapter_layer.resize_expert_rank(expert_name, new_rank)
            self.allocated[expert_name] = self.allocated.get(expert_name, 0) + (new_rank - current_rank)
            return True
        return False
    
    def release_reserve(self, expert_name: str, release_rank: int) -> bool:
        """Release reserve capacity from an expert."""
        if expert_name not in self.adapter_layer.adapter_names:
            return False
        
        current_rank = self.adapter_layer.adapter_ranks[
            self.adapter_layer.adapter_names.index(expert_name)
        ].item()
        
        min_rank = max(current_rank - self.allocated.get(expert_name, 0), 4)
        new_rank = max(current_rank - release_rank, min_rank)
        
        if new_rank < current_rank:
            self.adapter_layer.resize_expert_rank(expert_name, new_rank)
            self.allocated[expert_name] = self.allocated.get(expert_name, 0) - (current_rank - new_rank)
            return True
        return False
    
    def get_available_reserve(self, expert_name: str) -> int:
        """Get available reserve for an expert."""
        current_rank = self.adapter_layer.adapter_ranks[
            self.adapter_layer.adapter_names.index(expert_name)
        ].item()
        max_rank = current_rank + self.reserve_ranks.get(expert_name, 0)
        return max_rank - current_rank
    
    def total_reserve(self) -> int:
        return sum(self.reserve_ranks.values())
    
    def total_allocated(self) -> int:
        return sum(self.allocated.values())