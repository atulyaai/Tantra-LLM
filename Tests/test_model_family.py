from pathlib import Path

import pytest

from Tantra.model_family import ModelFamily

ROOT = Path(__file__).resolve().parents[1]


def test_repository_family_manifest_loads_and_has_64k_balanced_policy():
    family = ModelFamily.load(ROOT / "configs" / "model_family.yaml")
    assert family.config["growth_policy"] == "family"
    assert family.config["tokenizer_target"] == 65536
    assert family.config["runtime_policy"] == "balanced"
    assert family.members["sparse_frontier"].target_parameters == 1_000_000_000_000


def test_growth_plan_is_explicitly_planning_only_and_has_gates():
    family = ModelFamily.load(ROOT / "configs" / "model_family.yaml")
    plan = family.plan("edge", "balanced", "function_preserving_expansion")
    assert plan.execution_status == "planning_only"
    assert "run_function_preservation_probes_before_training" in plan.required_gates
    assert "retain_rollback_artifact" in plan.required_gates


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["family"][0].update(target_parameters=0), "positive"),
        (lambda c: c["family"][0].update(parent="missing"), "unknown parent"),
        (lambda c: c["family"][0].update(parent="balanced"), "cycle"),
        (lambda c: c["growth_methods"].append("random_growth"), "Unknown growth"),
    ],
)
def test_invalid_family_contract_is_rejected(mutate, message):
    import copy

    original = ModelFamily.load(ROOT / "configs" / "model_family.yaml").config
    config = copy.deepcopy(original)
    mutate(config)
    with pytest.raises(ValueError, match=message):
        ModelFamily(config)


def test_plan_rejects_non_parent_and_unknown_method():
    family = ModelFamily.load(ROOT / "configs" / "model_family.yaml")
    with pytest.raises(ValueError, match="not declared"):
        family.plan("edge", "quality", "distilled_sibling")
    with pytest.raises(ValueError, match="not enabled"):
        family.plan("edge", "balanced", "random_growth")
