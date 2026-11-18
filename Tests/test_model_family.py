import pytest

from Tantra.model_family import (
    ExpansionProposal,
    FamilyMember,
    validate_family_manifest,
)


def member(**overrides):
    values = dict(
        model_id="tantra-0p8b",
        parameter_count=800_000_000,
        tokenizer_vocab_size=10277,
        tokenizer_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        backend="native",
    )
    values.update(overrides)
    return FamilyMember(**values)


def test_valid_root_and_family_child():
    root = member()
    child = member(
        model_id="tantra-2b",
        parameter_count=2_000_000_000,
        parent_id=root.model_id,
        lineage_strategy="function_preserving_depth",
    )
    result = validate_family_manifest({
        "schema_version": 1,
        "members": [root.__dict__, child.__dict__],
    })
    assert [row.model_id for row in result] == ["tantra-0p8b", "tantra-2b"]


def test_legacy_tokenizer_is_valid_and_expansion_keeps_its_identity():
    root = member(tokenizer_vocab_size=10277)
    proposal = ExpansionProposal(
        root.model_id, "tantra-2b", 2_000_000_000,
        "function_preserving_depth", "copy_with_residual_gate",
    )
    assert proposal.candidate_record(root)["model"]["tokenizer_vocab_size"] == 10277


def test_rejects_missing_tokenizer():
    with pytest.raises(ValueError, match="positive"):
        member(tokenizer_vocab_size=0).validate()


def test_rejects_random_only_or_unknown_initialization():
    root = member()
    proposal = ExpansionProposal(
        root.model_id, "tantra-2b", 2_000_000_000,
        "function_preserving_depth", "random",
    )
    with pytest.raises(ValueError, match="random-only"):
        proposal.validate(root)


def test_rejects_shrinking_or_equal_parameter_target():
    root = member()
    proposal = ExpansionProposal(
        root.model_id, "tantra-same", root.parameter_count,
        "distill_sibling", "teacher_distillation",
    )
    with pytest.raises(ValueError, match="exceed"):
        proposal.validate(root)


def test_candidate_record_pins_parent_and_tokenizer_without_claiming_training():
    root = member()
    proposal = ExpansionProposal(
        root.model_id, "tantra-2b", 2_000_000_000,
        "function_preserving_depth", "copy_with_residual_gate",
    )
    record = proposal.candidate_record(root)
    assert record["source"]["immutable"] is True
    assert record["model"]["tokenizer_sha256"] == root.tokenizer_sha256
    assert record["model"]["status"] == "planned_not_trained"
    assert "held_out_regression_suite" in record["required_gates"]


def test_manifest_rejects_lineage_cycle():
    a = member(parent_id="tantra-b", lineage_strategy="distill_sibling")
    b = member(model_id="tantra-b", parent_id=a.model_id, lineage_strategy="distill_sibling")
    with pytest.raises(ValueError, match="cycle"):
        validate_family_manifest({"schema_version": 1, "members": [a.__dict__, b.__dict__]})
