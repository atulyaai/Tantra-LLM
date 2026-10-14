"""
Tantra Expert Health Monitor
==============================
Monitor per-expert training health:
- Loss spikes (wrong data detection)
- Gradient explosion
- Activation distribution shifts
- Usage patterns
"""

from __future__ import annotations

import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

log = logging.getLogger("Tantra.health")


class HealthStatus(Enum):
    OK = "OK"
    WARMING_UP = "WARMING_UP"
    LOSS_SPIKE = "LOSS_SPIKE"
    GRADIENT_EXPLOSION = "GRADIENT_EXPLOSION"
    DISTRIBUTION_SHIFT = "DISTRIBUTION_SHIFT"
    LOW_USAGE = "LOW_USAGE"
    DEAD_EXPERT = "DEAD_EXPERT"


@dataclass
class ExpertHealth:
    """Health metrics for a single expert."""
    name: str
    window_size: int = 500
    
    losses: deque = field(default_factory=lambda: deque(maxlen=500))
    grad_norms: deque = field(default_factory=lambda: deque(maxlen=500))
    activation_means: deque = field(default_factory=lambda: deque(maxlen=500))
    activation_stds: deque = field(default_factory=lambda: deque(maxlen=500))
    router_weights: deque = field(default_factory=lambda: deque(maxlen=500))
    
    # Thresholds
    loss_spike_threshold: float = 3.0
    grad_explosion_threshold: float = 5.0
    distribution_shift_sigma: float = 2.0
    min_activation_std: float = 1e-4
    
    def update(
        self,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        activation_mean: Optional[float] = None,
        activation_std: Optional[float] = None,
        router_weight: Optional[float] = None,
    ) -> None:
        """Update metrics with new values."""
        if loss is not None:
            self.losses.append(loss)
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
        if activation_mean is not None:
            self.activation_means.append(activation_mean)
        if activation_std is not None:
            self.activation_stds.append(activation_std)
        if router_weight is not None:
            self.router_weights.append(router_weight)
    
    def check(self) -> HealthStatus:
        """Check health status."""
        if len(self.losses) < 50:
            return HealthStatus.WARMING_UP
        
        # Check loss spike
        if len(self.losses) > 100:
            recent = np.mean(list(self.losses)[-20:])
            historical = np.mean(list(self.losses)[:-20])
            if recent > historical * self.loss_spike_threshold:
                return HealthStatus.LOSS_SPIKE
        
        # Check gradient explosion
        if len(self.grad_norms) > 50:
            recent_grad = np.mean(list(self.grad_norms)[-10:])
            historical_grad = np.percentile(list(self.grad_norms)[:-10], 99)
            if recent_grad > historical_grad * self.grad_explosion_threshold:
                return HealthStatus.GRADIENT_EXPLOSION
        
        # Check activation distribution shift
        if len(self.activation_means) > 50:
            recent_mean = np.mean(list(self.activation_means)[-20:])
            hist_means = list(self.activation_means)[:-20]
            hist_mean = np.mean(hist_means)
            hist_std = np.std(hist_means)
            if hist_std > 0 and abs(recent_mean - hist_mean) > self.distribution_shift_sigma * hist_std:
                return HealthStatus.DISTRIBUTION_SHIFT
        
        # Check dead expert (near-zero activation std)
        if len(self.activation_stds) > 20:
            recent_std = np.mean(list(self.activation_stds)[-10:])
            if recent_std < self.min_activation_std:
                return HealthStatus.DEAD_EXPERT
        
        # Check low usage
        if len(self.router_weights) > 100:
            avg_usage = np.mean(self.router_weights)
            if avg_usage < 0.001:
                return HealthStatus.LOW_USAGE
        
        return HealthStatus.OK
    
    def get_stats(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "avg_loss": float(np.mean(self.losses)) if self.losses else 0.0,
            "avg_grad_norm": float(np.mean(self.grad_norms)) if self.grad_norms else 0.0,
            "avg_activation_mean": float(np.mean(self.activation_means)) if self.activation_means else 0.0,
            "avg_activation_std": float(np.mean(self.activation_stds)) if self.activation_stds else 0.0,
            "avg_router_weight": float(np.mean(self.router_weights)) if self.router_weights else 0.0,
            "num_updates": len(self.losses),
        }


class ExpertHealthMonitor:
    """
    Monitor health of all experts during training.
    Detects wrong data, gradient issues, distribution shifts.
    """
    
    def __init__(
        self,
        expert_names: List[str],
        window_size: int = 500,
        alert_callback: Optional[callable] = None,
    ):
        self.experts = {
            name: ExpertHealth(name, window_size)
            for name in expert_names
        }
        self.alert_callback = alert_callback
        self.alert_history: List[Dict] = []
    
    def update(
        self,
        expert_name: str,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        activation: Optional[Any] = None,
        router_weight: Optional[float] = None,
    ) -> None:
        """Update metrics for an expert."""
        if expert_name not in self.experts:
            return
        
        health = self.experts[expert_name]
        
        # Extract activation stats if tensor provided
        act_mean = act_std = None
        if activation is not None and hasattr(activation, 'mean'):
            act_mean = float(activation.mean().item())
            act_std = float(activation.std().item())
        
        health.update(
            loss=loss,
            grad_norm=grad_norm,
            activation_mean=act_mean,
            activation_std=act_std,
            router_weight=router_weight,
        )
        
        # Check health
        status = health.check()
        if status != HealthStatus.OK and status != HealthStatus.WARMING_UP:
            self._alert(expert_name, status, health.get_stats())
    
    def _alert(self, expert_name: str, status: HealthStatus, stats: Dict) -> None:
        """Handle health alert."""
        alert = {
            "expert": expert_name,
            "status": status.value,
            "stats": stats,
        }
        self.alert_history.append(alert)
        
        log.warning(f"⚠️ EXPERT HEALTH ALERT: {expert_name} - {status.value}")
        log.warning(f"   Stats: {stats}")
        
        if self.alert_callback:
            self.alert_callback(expert_name, status, stats)
    
    def check_all(self) -> Dict[str, HealthStatus]:
        """Check health of all experts."""
        return {name: health.check() for name, health in self.experts.items()}
    
    def get_report(self) -> Dict[str, Any]:
        """Get full health report."""
        return {
            "experts": {
                name: {
                    "status": health.check().value,
                    "stats": health.get_stats(),
                }
                for name, health in self.experts.items()
            },
            "alerts": self.alert_history[-50:],  # Last 50 alerts
        }
    
    def get_anomalies(self) -> Dict[str, HealthStatus]:
        """Get only experts with anomalies."""
        return {
            name: status
            for name, status in self.check_all().items()
            if status not in (HealthStatus.OK, HealthStatus.WARMING_UP)
        }
    
    def should_quarantine_batch(self, expert_name: str) -> bool:
        """Check if current batch should be quarantined for this expert."""
        status = self.experts[expert_name].check()
        return status in (HealthStatus.LOSS_SPIKE, HealthStatus.GRADIENT_EXPLOSION)
    
    def reset_expert(self, expert_name: str) -> None:
        """Reset health tracking for an expert (e.g., after data fix)."""
        if expert_name in self.experts:
            self.experts[expert_name] = ExpertHealth(expert_name, self.experts[expert_name].window_size)
            log.info(f"Health monitor reset for {expert_name}")


# ── Automated Data Quality Check ─────────────────────────────────────────────

def check_batch_quality(
    batch: Dict,
    tokenizer,
    expert_name: str,
) -> Dict[str, Any]:
    """
    Quick quality checks on a batch before training.
    Returns dict with quality flags.
    """
    texts = batch.get("text", [])
    input_ids = batch.get("input_ids")
    
    issues = []
    
    # Check for empty/short texts
    short_count = sum(1 for t in texts if len(t) < 20)
    if short_count > len(texts) * 0.1:
        issues.append(f"SHORT_TEXTS: {short_count}/{len(texts)}")
    
    # Check for repeated tokens (degenerate)
    if input_ids is not None:
        for i, ids in enumerate(input_ids):
            unique_ratio = len(set(ids.tolist())) / len(ids)
            if unique_ratio < 0.1:
                issues.append(f"REPEATED_TOKENS sample {i}: unique_ratio={unique_ratio:.2f}")
    
    # Check for wrong language
    # (would need language detection)
    
    return {
        "expert": expert_name,
        "batch_size": len(texts),
        "issues": issues,
        "pass": len(issues) == 0,
    }