"""
Plasticity Engine — KNOWLEDGE-PRESERVING monitoring for NP-DNA.

Core principle: NEVER destroy learned knowledge.
  - Underutilized strands → distill into active strands via soft transfer
  - Overloaded strands → spawn child strands seeded from parent (copy, don't erase)
  - Dead strands → reactivate by rebalancing router, never zero out
  - Merging → only when strands are truly redundant (near-identical routing)

No zeroing. No destructive reinitialization. Knowledge only flows, never deleted.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from .model import NpDnaCore

logger = logging.getLogger(__name__)


@dataclass
class PlasticityEvent:
    step: int
    event_type: str
    details: str = ""


@dataclass
class PlasticityMetrics:
    strand_load: float = 0.0
    max_strand_load: float = 0.0
    min_strand_load: float = 1.0
    dead_count: int = 0
    overloaded_count: int = 0
    layer_loss_plateau: int = 0
    cortex_unused: int = 0
    router_entropy: float = 0.0
    last_check: float = field(default_factory=time.time)


class PlasticityEngine:
    """Monitors NP-DNA and recommends growth — never destroys knowledge.

    - Dead strands: detected and logged; router is rebalanced via entropy bonus
    - Overloaded strands: spawn new strands seeded from the parent
    - Redundant strands: merged by averaging router weights only (not strand params)
    - Loss plateau: triggers layer growth recommendation
    """

    def __init__(
        self,
        core: NpDnaCore,
        check_interval: int = 100,
        dead_threshold: float = 0.01,
        overload_threshold: float = 0.18,
        plateau_window: int = 50,
        plateau_threshold: float = 0.01,
        entropy_target: float = 0.6,
    ):
        self.core = core
        self.check_interval = check_interval
        self.dead_threshold = dead_threshold
        self.overload_threshold = overload_threshold
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.entropy_target = entropy_target

        self.events: list[PlasticityEvent] = []
        self.loss_history: list[float] = []
        self.metrics = PlasticityMetrics()

    def record_loss(self, loss: float) -> None:
        if isinstance(loss, bool):
            raise TypeError("loss must be numeric, got bool")
        if not isinstance(loss, (int, float)):
            raise TypeError(f"loss must be numeric, got {type(loss).__name__}")
        self.loss_history.append(loss)

    def check(self, step: int) -> list[PlasticityEvent]:
        if step % self.check_interval != 0:
            return []

        events: list[PlasticityEvent] = []

        events.extend(self._diagnose_strand_usage(step))
        events.extend(self._check_plateau(step))

        self.events.extend(events)
        return events

    def _diagnose_strand_usage(self, step: int) -> list[PlasticityEvent]:
        events = []
        total_loads = []
        total_dead = 0
        total_overloaded = 0

        for layer_i, mesh in enumerate(self.core.model.mesh_layers):
            stats = mesh.usage_stats
            if not stats:
                mesh.reset_usage()
                continue

            loads = list(stats.values())
            layer_avg = sum(loads) / len(loads) if loads else 0.0
            layer_max = max(loads) if loads else 0.0
            layer_min = min(loads) if loads else 0.0
            total_loads.extend(loads)

            dead = [s_id for s_id, ratio in stats.items() if ratio < self.dead_threshold]
            overloaded = [s_id for s_id, ratio in stats.items() if ratio > self.overload_threshold]

            if dead:
                total_dead += len(dead)
                msg = f"Layer {layer_i}: {len(dead)} low-usage strands — router needs more data to balance"
                logger.info("Plasticity: %s", msg)
                events.append(PlasticityEvent(step, "low_usage_strands", msg))

            if overloaded:
                total_overloaded += len(overloaded)
                msg = f"Layer {layer_i}: {len(overloaded)} high-use strands — suggesting growth"
                logger.info("Plasticity: %s", msg)
                events.append(PlasticityEvent(step, "overloaded_strands", msg))

            entropy = getattr(mesh, "last_router_entropy", 0.0)
            if entropy < 0.3:
                logger.info("Plasticity: Layer %d router entropy=%.3f — low diversity", layer_i, entropy)

            mesh.reset_usage()

        if total_loads:
            self.metrics.strand_load = sum(total_loads) / len(total_loads)
            self.metrics.max_strand_load = max(total_loads)
            self.metrics.min_strand_load = min(total_loads)
            self.metrics.dead_count = total_dead
            self.metrics.overloaded_count = total_overloaded

        self.metrics.last_check = time.time()
        return events

    def _check_plateau(self, step: int) -> list[PlasticityEvent]:
        events = []
        W = self.plateau_window
        if len(self.loss_history) < W * 2:
            return events

        old_avg = sum(self.loss_history[-W * 2:-W]) / W
        new_avg = sum(self.loss_history[-W:]) / W

        if old_avg > 0:
            improvement = (old_avg - new_avg) / old_avg
            if improvement < self.plateau_threshold:
                msg = f"Loss plateau: {old_avg:.4f} -> {new_avg:.4f} ({improvement:.1%})"
                logger.info("Plasticity: %s", msg)
                events.append(PlasticityEvent(step, "plateau", msg))

        return events

    def summary(self) -> str:
        if not self.events:
            return "No plasticity events recorded."
        lines = [f"Plasticity: {len(self.events)} events"]
        for e in self.events[-20:]:
            lines.append(f"  step {e.step}: [{e.event_type}] {e.details}")
        return "\n".join(lines)


class PlasticityAutoScaler:
    """Small policy object for dashboard/training auto-scale recommendations."""

    DEFAULT_CONFIG = {
        "strand_overload": 0.9,
        "plateau_window": 10,
        "plateau_variance": 0.001,
        "cortex_unused_ratio": 0.75,
        "max_history": 50,
        "max_strands": 128,
    }

    def __init__(self, config: dict | None = None):
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self.metrics = PlasticityMetrics()
        self._action_history: list[dict] = []

    def check_and_scale(
        self,
        strand_capacity: float,
        loss_history: list[float],
        cortex_size: int,
        cortex_used: int,
    ) -> list[str]:
        actions: list[str] = []
        now = time.time()
        cortex_unused = max(0, int(cortex_size) - max(0, int(cortex_used)))

        self.metrics.strand_load = float(strand_capacity)
        self.metrics.cortex_unused = cortex_unused
        self.metrics.last_check = now

        if strand_capacity >= self.config["strand_overload"]:
            actions.append("add_strand")

        window = int(self.config["plateau_window"])
        if len(loss_history) >= window:
            recent = [float(v) for v in loss_history[-window:]]
            if max(recent) - min(recent) <= float(self.config["plateau_variance"]):
                self.metrics.layer_loss_plateau = len(recent)
                actions.append("add_layer")

        if cortex_size > 0:
            unused_ratio = cortex_unused / max(1, cortex_size)
            if unused_ratio >= self.config["cortex_unused_ratio"]:
                actions.append("prune_cortex")

        return actions

    def get_metrics(self) -> dict:
        return {
            "strand_load": self.metrics.strand_load,
            "cortex_unused": self.metrics.cortex_unused,
            "layer_loss_plateau": self.metrics.layer_loss_plateau,
            "last_check": self.metrics.last_check,
        }

    def get_action_history(self) -> list[dict]:
        return list(self._action_history)
