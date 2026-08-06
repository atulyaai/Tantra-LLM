"""Combined tests for Tantra-LLM model, tokenizer, config, and growth mechanics.

Covers:
- Config and tokeniser basics (seed config, vocab resizing, tokenizer round-trip, dynamic vocab growth)
- Dynamic strand growth with and without unbounded mode
- Frozen genome cache behaviour (persistence, direct write, trainable rejection)
- SSM strand forward pass
"""

import pytest
import torch
from pathlib import Path

from npdna import CONFIGS, PREFERRED_CONFIG_NAMES, AtulyaTokenizer, NpDnaCore
from npdna.architecture import BIG_LAYER_NAMES, LayerSpec, NpDnaConfig, StrandConfig
from npdna.model import NpDnaModel


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


def test_dynamic_vocab_growth_can_force_frequent_chunk_tokens():
    tokenizer = AtulyaTokenizer(initial_capacity=2048)
    added, stats = tokenizer.dynamic_vocab_growth(
        ["alpha beta gamma alpha beta gamma"],
        sample_size=10,
        merge_rounds=1,
        min_pair_freq=1,
        target_vocab_size=tokenizer.size + 4,
        return_stats=True,
    )

    assert added >= 4
    assert stats["forced_tokens"] >= 1
    assert "alpha" in tokenizer.token_to_id
    assert tokenizer.encode("alpha", allow_growth=False) == [tokenizer.token_to_id["alpha"]]


def test_forced_chunk_tokens_invalidate_cached_bpe_encoding():
    tokenizer = AtulyaTokenizer(initial_capacity=2048)
    before = tokenizer.encode("example", allow_growth=False)

    added, stats = tokenizer.dynamic_vocab_growth(
        ["example example example"],
        sample_size=10,
        merge_rounds=1,
        min_pair_freq=1,
        target_vocab_size=tokenizer.size + 8,
        return_stats=True,
    )

    assert before != [tokenizer.token_to_id["example"]]
    assert added >= 1
    assert stats["forced_tokens"] >= 1
    assert tokenizer.encode("example", allow_growth=False) == [tokenizer.token_to_id["example"]]


def test_forced_chunk_tokens_respect_min_pair_frequency():
    tokenizer = AtulyaTokenizer(initial_capacity=2048)
    added, stats = tokenizer.dynamic_vocab_growth(
        [
            "common common unique_a",
            "common common unique_b",
        ],
        sample_size=10,
        merge_rounds=1,
        min_pair_freq=2,
        target_vocab_size=tokenizer.size + 20,
        return_stats=True,
    )

    assert added >= 1
    assert stats["forced_tokens"] < 20
    assert "common" in tokenizer.token_to_id
    assert "unique_a" not in tokenizer.token_to_id
    assert "unique_b" not in tokenizer.token_to_id


def test_dynamic_vocab_growth_returns_stats_when_vocab_is_full():
    tokenizer = AtulyaTokenizer(initial_capacity=1, max_capacity=1)

    added, stats = tokenizer.dynamic_vocab_growth(["unused"], return_stats=True)

    assert added == 0
    assert stats == {
        "sampled_texts": 0,
        "target_merges": 0,
        "forced_tokens": 0,
    }


def test_unbounded_growth_removes_the_strand_software_cap():
    config = NpDnaConfig(complexity=1.0)
    config.growth_unbounded = True
    model = NpDnaModel(config)
    before = sum(len(mesh.strands) for mesh in model.mesh_layers)
    model.grow_strands(1)
    after = sum(len(mesh.strands) for mesh in model.mesh_layers)
    assert after == before + len(model.mesh_layers)


def test_frozen_weight_cache_persists_in_training_mode():
    genome = NpDnaCore.from_config("seed").model.genome
    for parameter in genome.parameters():
        parameter.requires_grad_(False)
    genome.enable_frozen_weight_cache()
    genome.train()

    first = genome.generate_all(0)
    second = genome.generate_all(0)

    assert first is second
    assert not first["gate"][0].requires_grad


def test_direct_weight_write_binds_and_invalidates_strand_weights():
    model = NpDnaCore.from_config("seed").model
    for parameter in model.genome.parameters():
        parameter.requires_grad_(False)
    model.genome.enable_frozen_weight_cache(direct_write=True)
    strand = model.mesh_layers[0].strands[0]

    first = strand.direct_weights()
    assert strand.direct_weights() is first
    model.genome.disable_inference_cache()
    model.genome.enable_frozen_weight_cache(direct_write=True)
    assert strand.direct_weights() is not first


def test_frozen_weight_cache_rejects_trainable_genome():
    genome = NpDnaCore.from_config("seed").model.genome
    with pytest.raises(RuntimeError, match="all genome parameters"):
        genome.enable_frozen_weight_cache()


def test_ssm_strand_model_forward_runs():
    strand = StrandConfig(hidden_size=64, state_size=64, strand_type="ssm", use_swiglu=True)
    config = NpDnaConfig(
        complexity=0.5,
        initial_vocab=128,
        num_layers=1,
        mesh_specs=[LayerSpec(name="test", num_strands=2, top_k=1, strand=strand)],
    )
    model = NpDnaModel(config)
    logits, balance = model(torch.randint(0, model.vocab_size, (2, 8)))
    assert logits.shape == (2, 8, model.vocab_size)
    assert balance.ndim == 0


def _write_fake_checkpoint(tmp: Path, core: NpDnaCore, *, corrupt_key: str | None = None) -> Path:
    ckpt = tmp / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    core.save(str(ckpt))
    if corrupt_key is not None:
        state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
        target = state[corrupt_key]
        bad = target.new_zeros(list(reversed(target.shape)))  # swap dims
        state[corrupt_key] = bad
        torch.save(state, ckpt / "model.pt")
    return ckpt


# Corrupt a non-embedding weight so the load routine reaches the shape-mismatch
# guard (embedding.weight is validated by an earlier hidden-size check).
_CORRUPT_KEY = "lm_head.weight"


def test_load_rejects_shape_mismatch(tmp_path, monkeypatch):
    monkeypatch.delenv("NPDNA_REPAIR", raising=False)
    core = NpDnaCore.from_config("seed")
    ckpt = _write_fake_checkpoint(tmp_path, core, corrupt_key=_CORRUPT_KEY)
    with pytest.raises(RuntimeError, match=_CORRUPT_KEY):
        NpDnaCore.load(ckpt)


def test_load_repair_strips_mismatched(tmp_path, monkeypatch):
    monkeypatch.setenv("NPDNA_REPAIR", "1")
    core = NpDnaCore.from_config("seed")
    ckpt = _write_fake_checkpoint(tmp_path, core, corrupt_key=_CORRUPT_KEY)
    loaded = NpDnaCore.load(ckpt)
    assert loaded.config.hidden_size == core.config.hidden_size
