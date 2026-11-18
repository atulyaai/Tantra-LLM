import pytest

from Tantra.pretrained import (
    ExpansionPlan,
    ExpertInheritance,
    PretrainedSource,
)


def source(**overrides):
    values = dict(
        source_id="foundation-a",
        provider="community",
        model_name="example-foundation",
        revision="abc123",
        architecture="decoder-transformer",
        parameter_count=800_000_000,
        tokenizer_id="tok-a",
        tokenizer_sha256="a" * 64,
        license_id="permissive-example",
        license_status="approved",
        integration_mode="model_gateway",
    )
    values.update(overrides)
    return PretrainedSource(**values)


def test_source_requires_reviewed_license_and_valid_hash():
    source().validate()
    with pytest.raises(ValueError, match="reviewed"):
        source(license_status="unknown").validate()
    with pytest.raises(ValueError, match="reviewed"):
        source(license_status="review_required").validate()
    with pytest.raises(ValueError, match="SHA-256"):
        source(tokenizer_sha256="bad").validate()


@pytest.mark.parametrize(
    "method",
    ["adapter_transfer", "compatible_weight_transfer", "behavioural_distillation", "feature_alignment"],
)
def test_all_inheritance_methods_accept_known_parents(method):
    spec = ExpertInheritance(
        child_id="tantra-coding-child",
        parent_ids=("foundation-a", "foundation-b"),
        method=method,
        capabilities=("reasoning", "coding"),
        dataset_manifest_sha256="c" * 64,
    )
    spec.validate({"foundation-a", "foundation-b"})


def test_inheritance_rejects_missing_parent_and_mutable_parent():
    spec = ExpertInheritance(
        child_id="child",
        parent_ids=("missing",),
        method="adapter_transfer",
        capabilities=("coding",),
        dataset_manifest_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="unknown"):
        spec.validate({"foundation-a"})


def test_expansion_plan_records_distillation_replay_and_no_training_claim():
    src = source()
    plan = ExpansionPlan(
        source_id=src.source_id,
        target_id="foundation-a-2b",
        target_parameter_count=2_000_000_000,
        method="residual_branch",
        adaptation="lora",
    )
    record = plan.to_record(src)
    assert record["candidate"]["status"] == "planned_not_trained"
    assert record["training"]["teacher_distillation"] is True
    assert record["training"]["old_capability_replay"] is True
    assert record["training"]["source_checkpoint_immutable"] is True


@pytest.mark.parametrize("method", ["width", "depth", "residual_branch"])
def test_all_three_growth_methods_are_supported(method):
    src = source()
    ExpansionPlan(
        src.source_id, "child", 1_000_000_000, method, "adapter"
    ).validate(src)


def test_expansion_rejects_upgrade_without_replay():
    src = source()
    plan = ExpansionPlan(
        src.source_id, "child", 1_000_000_000,
        "residual_branch", "lora", use_replay=False
    )
    with pytest.raises(ValueError, match="required"):
        plan.validate(src)
