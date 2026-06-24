import torch

from npdna import CONFIGS, PREFERRED_CONFIG_NAMES, AtulyaTokenizer, NpDnaCore
from npdna.config import BIG_LAYER_NAMES, LayerSpec, NpDnaConfig


def test_seed_is_only_named_config():
    assert tuple(CONFIGS) == ("seed",)
    assert PREFERRED_CONFIG_NAMES == ("seed",)


def test_seed_core_starts_small_and_can_expand():
    core = NpDnaCore.from_config("seed")
    old_vocab = core.model.vocab_size
    old_mean = core.model.embedding.weight.mean(dim=0).detach().clone()
    core.model.resize_embeddings(old_vocab + 8)
    assert core.model.vocab_size == old_vocab + 8
    new_rows = core.model.embedding.weight[old_vocab:].detach()
    assert new_rows.sub(old_mean).abs().max().item() > 0
    assert torch.pdist(new_rows).min().item() > 0


def test_seed_uses_small_expert_strand_budget():
    cfg = NpDnaConfig(complexity=1.0)
    strands_by_name = {spec.name: spec.num_strands for spec in cfg.mesh_specs}

    assert cfg.total_strands == 65
    assert all(
        2 <= count <= 4
        for name, count in strands_by_name.items()
        if name not in BIG_LAYER_NAMES
    )
    assert all(strands_by_name[name] == 8 for name in BIG_LAYER_NAMES)
    assert all(spec.top_k <= spec.num_strands for spec in cfg.mesh_specs)
    assert cfg.genome.max_strands == 76


def test_explicit_checkpoint_specs_are_preserved():
    cfg = NpDnaConfig(num_layers=1, mesh_specs=[LayerSpec(name="legacy", num_strands=9, top_k=3)])

    assert cfg.mesh_specs[0].num_strands == 9
    assert cfg.total_strands == 9
    assert cfg.genome.max_strands == 9


def test_model_strand_growth_respects_layer_caps():
    core = NpDnaCore.from_config("seed")

    core.model.grow_strands(count=10)

    strands_by_name = {spec.name: spec.num_strands for spec in core.config.mesh_specs}
    assert sum(strands_by_name.values()) == 76
    assert all(
        count <= 4
        for name, count in strands_by_name.items()
        if name not in BIG_LAYER_NAMES
    )
    assert all(strands_by_name[name] == 8 for name in BIG_LAYER_NAMES)


def test_tokenizer_round_trip():
    tokenizer = AtulyaTokenizer(initial_capacity=4096, max_capacity=256_000)
    text = "Hello NP-DNA"
    assert tokenizer.decode(tokenizer.encode(text)) == text
