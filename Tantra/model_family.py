"""Auditable model-family metadata and safe expansion proposals.

This module does not resize weights or claim a model has grown. It validates
lineage and creates plans that a later, separately tested morphism trainer can
consume. Checkpoints remain immutable; candidates belong under runs/.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


GROWTH_STRATEGIES = {
    "function_preserving_depth",
    "function_preserving_width",
    "distill_sibling",
    "add_domain_experts",
}
INITIALIZATION_POLICIES = {
    "copy_with_residual_gate",
    "function_preserving_projection",
    "teacher_distillation",
    "independent_expert_init",
}


@dataclass(frozen=True)
class FamilyMember:
    model_id: str
    parameter_count: int
    tokenizer_vocab_size: int
    tokenizer_sha256: str
    checkpoint_sha256: str
    backend: str
    parent_id: str | None = None
    lineage_strategy: str = "root"
    status: str = "candidate"

    def validate(self) -> None:
        if not self.model_id or any(c.isspace() for c in self.model_id):
            raise ValueError("model_id must be a non-empty token")
        if self.parameter_count < 1:
            raise ValueError("parameter_count must be positive")
        if self.tokenizer_vocab_size < 1:
            raise ValueError("tokenizer_vocab_size must be positive")
        for name, value in (("tokenizer_sha256", self.tokenizer_sha256),
                            ("checkpoint_sha256", self.checkpoint_sha256)):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
                raise ValueError(f"{name} must be a SHA-256 hex digest")
        if not self.backend:
            raise ValueError("backend is required")
        if self.parent_id is None and self.lineage_strategy != "root":
            raise ValueError("root members must use lineage_strategy='root'")
        if self.parent_id is not None and self.lineage_strategy not in GROWTH_STRATEGIES:
            raise ValueError("unknown child lineage strategy")


@dataclass(frozen=True)
class ExpansionProposal:
    source_id: str
    target_id: str
    target_parameter_count: int
    strategy: str
    initialization: str
    preserve_tokenizer: bool = True
    source_immutable: bool = True

    def validate(self, source: FamilyMember) -> None:
        source.validate()
        if self.source_id != source.model_id:
            raise ValueError("proposal source does not match source member")
        if not self.target_id or self.target_id == self.source_id:
            raise ValueError("target_id must be distinct and non-empty")
        if self.target_parameter_count <= source.parameter_count:
            raise ValueError("family growth target must exceed source parameter count")
        if self.strategy not in GROWTH_STRATEGIES:
            raise ValueError("unsupported growth strategy")
        if self.initialization not in INITIALIZATION_POLICIES:
            raise ValueError("unsupported initialization policy; random-only growth is forbidden")
        if not self.preserve_tokenizer:
            raise ValueError("tokenizer migration is a separate, gated operation")
        if not self.source_immutable:
            raise ValueError("source checkpoint must remain immutable")

    def candidate_record(self, source: FamilyMember) -> dict[str, Any]:
        self.validate(source)
        return {
            "schema_version": 1,
            "model": {
                "model_id": self.target_id,
                "parameter_count_target": self.target_parameter_count,
                "tokenizer_vocab_size": source.tokenizer_vocab_size,
                "tokenizer_sha256": source.tokenizer_sha256,
                "backend": source.backend,
                "parent_id": source.model_id,
                "lineage_strategy": self.strategy,
                "initialization": self.initialization,
                "status": "planned_not_trained",
            },
            "source": {
                "checkpoint_sha256": source.checkpoint_sha256,
                "immutable": True,
            },
            "required_gates": [
                "shape_and_tensor_integrity",
                "finite_weights_and_gradients",
                "function_preservation_before_training",
                "held_out_regression_suite",
                "language_split_evaluation_hi_en_hinglish",
                "latency_ram_vram_benchmark",
                "human_review_for_claimed_capabilities",
            ],
        }


def validate_family_manifest(payload: Mapping[str, Any]) -> list[FamilyMember]:
    """Validate a JSON-compatible family manifest and return its members."""
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported family manifest schema_version")
    rows = payload.get("members")
    if not isinstance(rows, list) or not rows:
        raise ValueError("members must be a non-empty list")
    members = [FamilyMember(**row) for row in rows]
    by_id: dict[str, FamilyMember] = {}
    for member in members:
        member.validate()
        if member.model_id in by_id:
            raise ValueError("duplicate model_id: " + member.model_id)
        by_id[member.model_id] = member
    for member in members:
        if member.parent_id is not None and member.parent_id not in by_id:
            raise ValueError("unknown parent_id: " + member.parent_id)
        if member.parent_id == member.model_id:
            raise ValueError("model cannot be its own parent")
    # Detect cycles, not just missing parent references.
    for member in members:
        seen: set[str] = set()
        current: FamilyMember | None = member
        while current is not None:
            if current.model_id in seen:
                raise ValueError("cycle in model-family lineage")
            seen.add(current.model_id)
            current = by_id.get(current.parent_id) if current.parent_id else None
    return members


def member_record(member: FamilyMember) -> dict[str, Any]:
    member.validate()
    return asdict(member)
