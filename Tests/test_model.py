"""Tests for tantra.model + tantra.bitnet + tantra.moe"""
import pytest
import torch

from Tantra.config import NeuroCoreConfig, MoEConfig
from Tantra.model import DynamicScaleNorm, RotaryPositionalEncoding, ALRAAttention, SparseGatedProjection, NeuroCoreBlock, NeuroCoreModel


def test_dsn_output_shape():
    dsn = DynamicScaleNorm(256)
    x = torch.randn(2, 10, 256)
    out = dsn(x)
    assert out.shape == (2, 10, 256)

def test_rope_applies():
    rope = RotaryPositionalEncoding(64)
    q = torch.randn(1, 4, 10, 64)
    k = torch.randn(1, 4, 10, 64)
    q_r, k_r = rope.apply(q, k, 10)
    assert q_r.shape == q.shape
    assert not torch.allclose(q, q_r)

def test_model_forward():
    cfg = NeuroCoreConfig.small()
    model = NeuroCoreModel(cfg)
    ids = torch.randint(0, cfg.vocab.vocab_size, (1, 16))
    logits, states = model(ids)
    assert logits.shape == (1, 16, cfg.vocab.vocab_size)
    assert states is None

def test_model_generate():
    cfg = NeuroCoreConfig.small()
    model = NeuroCoreModel(cfg)
    prompt = torch.randint(0, cfg.vocab.vocab_size, (1, 4))
    out = model.generate(prompt, max_new_tokens=3, temperature=0.0)
    assert out.shape == (1, 7)  # 4 prompt + 3 generated

def test_model_param_count():
    cfg = NeuroCoreConfig.small()
    model = NeuroCoreModel(cfg)
    n = model.num_parameters
    assert n > 0

def test_latent_cot_header():
    cfg = NeuroCoreConfig.small()
    model = NeuroCoreModel(cfg, reasoning_depth=2)
    ids = torch.randint(0, cfg.vocab.vocab_size, (1, 8))
    logits_with_cot, _ = model(ids, use_latent_reasoning=True)
    logits_no_cot, _ = model(ids, use_latent_reasoning=False)
    assert logits_with_cot.shape == (1, 8, cfg.vocab.vocab_size)
    assert logits_no_cot.shape == (1, 8, cfg.vocab.vocab_size)
    assert not torch.allclose(logits_with_cot, logits_no_cot)

