"""
tests/test_acceleration.py — Tests for MTP loss, bfloat16 autocast, and speculative decoding.
"""
import pytest
import torch

from Tantra.config import NeuroCoreConfig, ALRAConfig, SGPConfig, VocabConfig
from Tantra.model import NeuroCoreModel
from Tantra.train import NeuroTrainer

@pytest.fixture
def micro_config():
    cfg = NeuroCoreConfig()
    cfg.block.num_layers = 1
    cfg.block.alra.dim = 32
    cfg.block.alra.num_heads = 4
    cfg.block.alra.head_dim = 8
    cfg.block.sgp.dim = 32
    cfg.vocab = VocabConfig(vocab_size=1000)
    cfg.dim = 32
    cfg.use_mtp = True
    cfg.reasoning_depth = 1
    return cfg

def test_mtp_loss_and_bfloat16_autocast(micro_config):
    model = NeuroCoreModel(micro_config)
    trainer = NeuroTrainer(model, lr=1e-3)
    
    x = torch.randint(0, 1000, (2, 16))
    y = torch.randint(0, 1000, (2, 16))
    
    loss, acc, ppl, grad_norm, at_boundary = trainer.train_step(x, y)
    assert isinstance(loss, float)
    assert loss >= 0.0
    assert at_boundary is True
    assert trainer.step_count == 1

def test_speculative_mtp_generation(micro_config):
    model = NeuroCoreModel(micro_config)
    prompt = torch.tensor([[1, 2, 3]])
    
    out = model.generate(prompt, max_new_tokens=8, use_mtp_speculation=True)
    assert out.shape == (1, 11)


def test_token_stream_generation(micro_config):
    model = NeuroCoreModel(micro_config)
    prompt = torch.tensor([[1, 2, 3]])

    tokens = list(model.generate_stream(prompt, max_new_tokens=4, temperature=0))
    assert 1 <= len(tokens) <= 4
    assert all(token.ndim == 0 for token in tokens)
