"""Metadata-only model-family registry and controlled-growth planning.

This module does not resize tensors or train models. It validates lineage and
produces an auditable plan; execution belongs in a separately tested growth
backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FamilyMember:
    id: str
    target_parameters: int
    role: str
    parent: str | None = None
    status: str = "planned"


@dataclass(frozen=True)
class GrowthPlan:
    parent: str
    child: str
    method: str
    required_gates: tuple[str, ...]
    execution_status: str = "planning_only"


class ModelFamily:
    """Validated directed acyclic family of model-size targets."""

    METHODS = {
        "distilled_sibling",
        "function_preserving_expansion",
        "depth_width_morphism",
        "sparse_expert_growth",
        "fresh_sibling_control",
    }

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.members = {
            row["id"]: FamilyMember(
                id=row["id"],
                target_parameters=int(row["target_parameters"]),
                role=row["role"],
                parent=row.get("parent"),
                status=row.get("status", "planned"),
            )
            for row in config.get("family", [])
        }
        self.validate()

    @classmethod
    def load(cls, path: str | Path) -> "ModelFamily":
        with Path(path).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("Family configuration must be a mapping")
        return cls(config)

    def validate(self) -> None:
        if not self.members:
            raise ValueError("Family must contain at least one member")
        for member in self.members.values():
            if member.target_parameters <= 0:
                raise ValueError(f"{member.id}: target_parameters must be positive")
            if member.parent is not None and member.parent not in self.members:
                raise ValueError(f"{member.id}: unknown parent {member.parent}")
            if member.parent == member.id:
                raise ValueError(f"{member.id}: cannot parent itself")

        for member_id in self.members:
            visited: set[str] = set()
            current: str | None = member_id
            while current is not None:
                if current in visited:
                    raise ValueError(f"Family lineage cycle detected at {current}")
                visited.add(current)
                current = self.members[current].parent

        methods = set(self.config.get("growth_methods", []))
        unknown = methods - self.METHODS
        if unknown:
            raise ValueError("Unknown growth methods: " + ", ".join(sorted(unknown)))

    def plan(self, parent: str, child: str, method: str) -> GrowthPlan:
        if parent not in self.members or child not in self.members:
            raise KeyError("Unknown family member")
        if self.members[child].parent != parent:
            raise ValueError(f"{child} is not declared as a child of {parent}")
        if method not in self.METHODS or method not in self.config.get("growth_methods", []):
            raise ValueError(f"Growth method is not enabled: {method}")
        gates = (
            "freeze_parent_and_record_hashes",
            "validate_tensor_mapping_and_tokenizer_compatibility",
            "run_function_preservation_probes_before_training",
            "compare_parent_and_same_budget_control",
            "evaluate_frozen_heldout_by_language_and_domain",
            "check_safety_latency_and_peak_memory",
            "retain_rollback_artifact",
        )
        return GrowthPlan(parent, child, method, gates)
