"""Auditable pretrained-model intake, adaptation and expert-inheritance contracts.

These are metadata/validation contracts, not a model downloader or trainer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

INTEGRATION_MODES = {"model_gateway", "neural_adapter"}
ADAPTATION_METHODS = {"lora", "adapter", "full_finetune", "distillation"}
INHERITANCE_METHODS = {
    "adapter_transfer", "compatible_weight_transfer",
    "behavioural_distillation", "feature_alignment",
}
GROWTH_METHODS = {"width", "depth", "residual_branch"}
LICENSE_STATUSES = {"approved", "review_required", "restricted", "unknown"}


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value.lower()
    ):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")


@dataclass(frozen=True)
class PretrainedSource:
    source_id: str
    provider: str
    model_name: str
    revision: str
    architecture: str
    parameter_count: int
    tokenizer_id: str
    tokenizer_sha256: str
    weights_sha256: str
    license_id: str
    license_status: str
    integration_mode: str = "model_gateway"

    def validate(self) -> None:
        for field_name in (
            "source_id", "provider", "model_name", "revision",
            "architecture", "tokenizer_id", "license_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} is required")
        if isinstance(self.parameter_count, bool) or not isinstance(self.parameter_count, int) or self.parameter_count < 1:
            raise ValueError("parameter_count must be a positive integer")
        _require_sha256(self.tokenizer_sha256, "tokenizer_sha256")
        _require_sha256(self.weights_sha256, "weights_sha256")
        if self.license_status not in LICENSE_STATUSES:
            raise ValueError("unknown license_status")
        if self.license_status != "approved":
            raise ValueError("source license must be reviewed before integration")
        if self.integration_mode not in INTEGRATION_MODES:
            raise ValueError("unsupported integration_mode")


@dataclass(frozen=True)
class ExpertInheritance:
    child_id: str
    parent_ids: tuple[str, ...]
    method: str
    capabilities: tuple[str, ...]
    dataset_manifest_sha256: str
    preserve_parent_checkpoints: bool = True

    def validate(self, known_source_ids: Iterable[str]) -> None:
        if not isinstance(self.child_id, str) or not self.child_id.strip():
            raise ValueError("child_id is required")
        if not self.parent_ids or any(not isinstance(p, str) or not p.strip() for p in self.parent_ids):
            raise ValueError("at least one valid parent_id is required")
        known = set(known_source_ids)
        if self.child_id in self.parent_ids:
            raise ValueError("child cannot inherit from itself")
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError("duplicate inheritance parent")
        if not set(self.parent_ids).issubset(known):
            raise ValueError("unknown inheritance parent")
        if self.method not in INHERITANCE_METHODS:
            raise ValueError("unsupported inheritance method")
        if not self.capabilities or any(not isinstance(c, str) or not c.strip() for c in self.capabilities):
            raise ValueError("at least one named capability is required")
        _require_sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        if self.preserve_parent_checkpoints is not True:
            raise ValueError("parent checkpoints must remain immutable")


@dataclass(frozen=True)
class ExpansionPlan:
    source_id: str
    target_id: str
    target_parameter_count: int
    method: str
    adaptation: str
    use_distillation: bool = True
    use_replay: bool = True
    preserve_source: bool = True

    def validate(self, source: PretrainedSource) -> None:
        source.validate()
        if self.source_id != source.source_id:
            raise ValueError("plan source does not match pretrained source")
        if not isinstance(self.target_id, str) or not self.target_id.strip() or self.target_id == self.source_id:
            raise ValueError("target_id must be distinct and non-empty")
        if isinstance(self.target_parameter_count, bool) or not isinstance(self.target_parameter_count, int):
            raise ValueError("target_parameter_count must be a positive integer")
        if self.target_parameter_count <= source.parameter_count:
            raise ValueError("target parameter count must exceed source")
        if self.method not in GROWTH_METHODS:
            raise ValueError("unsupported growth method")
        if self.adaptation not in ADAPTATION_METHODS:
            raise ValueError("unsupported adaptation method")
        if self.preserve_source is not True:
            raise ValueError("source checkpoint must remain immutable")
        if self.use_distillation is not True or self.use_replay is not True:
            raise ValueError("distillation and replay are required for the default upgrade path")

    def to_record(self, source: PretrainedSource) -> dict[str, Any]:
        self.validate(source)
        return {
            "schema_version": 1,
            "source": asdict(source),
            "candidate": {
                "model_id": self.target_id,
                "parameter_count_target": self.target_parameter_count,
                "growth_method": self.method,
                "adaptation": self.adaptation,
                "status": "planned_not_trained",
            },
            "training": {
                "teacher_distillation": self.use_distillation,
                "old_capability_replay": self.use_replay,
                "source_checkpoint_immutable": True,
            },
            "promotion_gates": [
                "license_and_provenance_review",
                "source_checkpoint_hash_unchanged",
                "held_out_capability_gain",
                "old_capability_regression_limits",
                "router_and_integration_tests",
                "latency_and_memory_budget",
                "safety_and_identity_regression",
            ],
        }
