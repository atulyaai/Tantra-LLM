"""
Tests/test_optimizers.py — Tests for AdamW and native Lion optimizer.
"""
import torch
import pytest
from Tantra.model import build_cpu_model
from Tantra.train import NeuroTrainer, Lion, build_optimizer


def test_build_optimizers():
    p = [torch.nn.Parameter(torch.randn(10, 10))]
    opt_adamw = build_optimizer("adamw", p, lr=1e-4, weight_decay=0.01)
    assert opt_adamw is not None

    opt_lion = build_optimizer("lion", p, lr=5e-5, weight_decay=0.05)
    assert isinstance(opt_lion, (Lion, torch.optim.Optimizer))

    opt_adam = build_optimizer("adam", p, lr=1e-4, weight_decay=0.01)
    assert isinstance(opt_adam, torch.optim.Adam)

    opt_sgd = build_optimizer("sgd", p, lr=1e-3, weight_decay=0.0)
    assert isinstance(opt_sgd, torch.optim.SGD)


def test_lion_step_execution():
    model = build_cpu_model("micro10", attention_kind="causal")
    trainer = NeuroTrainer(model, lr=5e-5, weight_decay=0.05, optimizer_name="lion", total_steps=5)
    
    x = torch.randint(0, 32768, (1, 16))
    y = torch.randint(0, 32768, (1, 16))
    loss, acc, ppl, grad_norm, at_boundary = trainer.train_step(x, y)
    
    assert loss > 0
    assert ppl > 0
    assert at_boundary is True
    assert trainer.step_count == 1
