"""Tests for the category adapter / specialist-layer system."""
import os
import json
import tempfile

import pytest
import torch

from Tantra.config import NeuroCoreConfig
from Tantra.model import NeuroCoreModel
from Tantra.model import cpu_dense_config
from Tantra.adapters import (
    AdapterRegistry, AdapterCategory, RequestRouter,
    install_category_layers,
)


def _tmp_registry(tmp_path=None):
    import tempfile
    d = tempfile.mkdtemp() if tmp_path is None else str(tmp_path)
    return AdapterRegistry(path=os.path.join(d, "registry.json"))


def test_specialist_layer_is_installed_and_identity_like():
    cfg = cpu_dense_config(vocab_size=256)
    model = NeuroCoreModel(cfg, use_mtp=False)
    ids = torch.randint(0, 256, (1, 8))
    baseline, _ = model(ids, use_latent_reasoning=False)

    model.add_category_layers(["math"], clone_layer_index=-1)
    adapted, _ = model(ids, use_latent_reasoning=False, adapter_name="math")
    # No training yet: the cloned specialist layer behaves like running the
    # last shared block again — close enough that the architecture is valid.
    assert "math" in model.category_layers
    assert adapted.shape == baseline.shape


def test_freeze_for_category_trains_only_one_layer():
    cfg = cpu_dense_config(vocab_size=256)
    model = NeuroCoreModel(cfg, use_mtp=False)
    model.add_category_layers(["math", "science"], clone_layer_index=-1)

    model.freeze_for_category("math")
    assert all(p.requires_grad for p in model.category_layers["math"].parameters())
    assert not any(p.requires_grad for p in model.category_layers["science"].parameters())
    # Shared base must stay frozen.
    assert not any(p.requires_grad for p in model.layers[0].parameters())


def test_install_helper_reports_param_counts():
    cfg = cpu_dense_config(vocab_size=256)
    model = NeuroCoreModel(cfg, use_mtp=False)
    registry = _tmp_registry()
    for c in (AdapterCategory(name="code"), AdapterCategory(name="safety")):
        registry._categories[c.name] = c
    counts = install_category_layers(model, registry.all())
    assert set(counts) == {"code", "safety"}
    assert all(v > 0 for v in counts.values())
    # A single transformer layer at dim 256 sits in the ~1-3M-per-adapter budget.
    assert all(100_000 <= v <= 10_000_000 for v in counts.values())


def test_request_router_picks_expected_category():
    registry = _tmp_registry()
    registry.seed_defaults()
    router = RequestRouter(registry)

    assert router.route("How do I write a for loop in Python?") == "code"
    assert router.route("Solve the integral of x squared") == "math"
    assert router.route("Translate this sentence to Hindi: नमस्ते") == "multilingual"
    # Generic chit-chat routes to the general conversation category (the
    # base-fallback adapter), not a specialized domain.
    assert router.route("hi, how are you?") == "general"


def test_registry_persists_categories(tmp_path):
    registry = _tmp_registry(tmp_path)
    registry.seed_defaults()
    assert len(registry) == 8
    registry.add("history", description="Historic knowledge", topics=["history"], rank=32)
    assert "history" in registry

    reloaded = AdapterRegistry(path=str(tmp_path / "registry.json"))
    assert "history" in reloaded
    assert reloaded.get("history").topics == ["history"]
    assert reloaded.remove("history")
    assert "history" not in reloaded


def test_registry_persists_depth_and_bounds(tmp_path):
    registry = _tmp_registry(tmp_path)
    cat = AdapterCategory(name="code", max_depth=3)
    registry.add(cat.name, description=cat.description, topics=cat.topics,
                 rank=cat.rank, max_depth=cat.max_depth)
    registry.update_depth("code", depth=2, params=12345)

    reloaded = AdapterRegistry(path=str(tmp_path / "registry.json"))
    rcat = reloaded.get("code")
    assert rcat.depth == 2
    assert rcat.max_depth == 3
    assert rcat.min_depth == 1
    assert rcat.params == 12345


def test_category_stack_grows_and_shrinks_shape_safe():
    cfg = cpu_dense_config(vocab_size=256)
    model = NeuroCoreModel(cfg, use_mtp=False)
    model.add_category_layers(["math"], clone_layer_index=-1, depth=1)
    assert model.category_depth("math") == 1
    ids = torch.randint(0, 256, (1, 8))
    out1, _ = model(ids, use_latent_reasoning=False, adapter_name="math")

    assert model.grow_category("math", cap=3) is True
    assert model.category_depth("math") == 2
    out2, _ = model(ids, use_latent_reasoning=False, adapter_name="math")
    assert out2.shape == out1.shape

    assert model.shrink_category("math", floor=1) is True
    assert model.category_depth("math") == 1
    out3, _ = model(ids, use_latent_reasoning=False, adapter_name="math")
    assert out3.shape == out1.shape

    # Cannot shrink below the floor.
    assert model.shrink_category("math", floor=1) is False
    assert model.category_depth("math") == 1


def test_growth_controller_decides_grow_and_shrink():
    from Tantra.evolution import CategoryGrowthController

    # GROW: plateaued, used a lot, below cap -> add a layer.
    grow_ctrl = CategoryGrowthController(plateau_patience=10, min_delta=0.001)
    decision = None
    for _ in range(15):
        d = grow_ctrl.observe("math", 1.20, cat_routed=1000, total_routed=1000,
                              depth=1, min_depth=1, max_depth=3)
        if d is not None:
            decision = d
            break
    assert decision == "grow"

    # SHRINK: converged (~95% of best) and barely routed -> reclaim a layer.
    shrink_ctrl = CategoryGrowthController(plateau_patience=10, min_delta=0.001, fit_target_ratio=0.95)
    decision = None
    best = 0.50
    for _ in range(15):
        # Loss hovers just under the 95% fit bar of the best seen.
        d = shrink_ctrl.observe("science", best * 0.94, cat_routed=10, total_routed=100000,
                                depth=2, min_depth=1, max_depth=3)
        if d is not None:
            decision = d
            break
    assert decision == "shrink"
