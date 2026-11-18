"""Regression tests for training integrity and strict checkpoint loading."""
import json

import pytest
import torch
from torch import nn

from Tantra import ModelConfig
from Tantra.train import (
    TantraModel,
    TrainArgs,
    _loop,
    split_expert_rows,
)


def _key(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def test_feedback_oversampling_happens_only_after_unique_split():
    curated = [
        {"messages": [{"role": "user", "content": f"question {i}"},
                      {"role": "assistant", "content": f"answer {i}"}]}
        for i in range(20)
    ]
    feedback = [curated[0], {"messages": [{"role": "user", "content": "new feedback"},
                                          {"role": "assistant", "content": "new answer"}]}]
    train, held = split_expert_rows(curated, feedback, seed=3, feedback_weight=3)
    assert not ({_key(row) for row in train} & {_key(row) for row in held})
    assert len(held) == 2
    # Repeated feedback examples may be weighted in train, but never appear in held too.


class _ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def loss(self, ids, labels):
        return self.weight.square()


class _TwoBatchDataset:
    def __len__(self):
        return 2

    def batches(self, batch_size, rng=None):
        yield torch.ones((1, 2), dtype=torch.long), torch.ones((1, 2), dtype=torch.long), None
        yield torch.ones((1, 2), dtype=torch.long), torch.ones((1, 2), dtype=torch.long), None


def test_gradient_accumulation_flushes_short_epoch_tail():
    model = _ScalarModel()
    before = model.weight.detach().clone()
    loss = _loop(
        model, [model.weight], _TwoBatchDataset(),
        TrainArgs(steps=2, batch_size=1, lr=0.1, warmup=0, log_every=10,
                  gradient_accumulation=3),
        "accumulation-regression",
    )
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, model.weight.detach())


def test_incomplete_adapter_checkpoint_is_rejected():
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2,
                      n_kv_heads=1, mlp_mult=2, max_seq_len=32, lora_rank=2)
    model = TantraModel(cfg)
    model.add_expert("greeting", rank=2)
    state = model.expert_state("greeting")
    first_key = next(iter(state))
    incomplete = dict(state)
    incomplete.pop(first_key)
    with pytest.raises(ValueError, match="incomplete or incompatible"):
        model.load_expert_state("greeting", incomplete, rank=2)


def test_base_loader_rejects_missing_registered_adapter(tmp_path):
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2,
                      n_kv_heads=1, mlp_mult=2, max_seq_len=32, lora_rank=2)
    model = TantraModel(cfg)
    model.add_expert("greeting", rank=2)
    base = tmp_path / "base.pt"
    model.save(base, tmp_path / "experts")
    (tmp_path / "experts" / "greeting.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Missing adapter checkpoint"):
        TantraModel.load(base, tmp_path / "experts")



def test_tiny_model_config_keeps_nonzero_feed_forward_width():
    cfg = ModelConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=2,
                      n_kv_heads=1, mlp_mult=2, max_seq_len=32)
    assert cfg.mlp_hidden >= 64


def test_model_config_rejects_invalid_rope_head_dimension():
    with pytest.raises(ValueError, match="head dimension must be even"):
        ModelConfig(vocab_size=64, d_model=12, n_layers=1, n_heads=3,
                    n_kv_heads=1, mlp_mult=2, max_seq_len=32)


def test_tokenizer_rejects_missing_control_tokens():
    from tokenizers import Tokenizer, models
    from Tantra.tokenizer import TantraTokenizer
    with pytest.raises(ValueError, match="missing required special tokens"):
        TantraTokenizer(Tokenizer(models.BPE()))



def test_kv_cached_logits_match_full_causal_forward():
    torch.manual_seed(7)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2,
                      n_kv_heads=1, mlp_mult=2, max_seq_len=32)
    model = TantraModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 9))
    with torch.no_grad():
        full = model(tokens)
        caches = [{} for _ in model.layers]
        pieces = []
        for pos in range(tokens.shape[1]):
            pieces.append(model(tokens[:, pos:pos + 1], caches=caches, pos=pos))
        cached = torch.cat(pieces, dim=1)
    assert torch.allclose(full, cached, atol=1e-5, rtol=1e-5)
