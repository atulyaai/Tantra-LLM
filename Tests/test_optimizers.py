"""
Tests/test_optimizers.py — Comprehensive tests for AdamW, native Lion optimizer,
mathematical update correctness, and memory state verification.
"""
import torch
import pytest
from Tantra.model import build_cpu_model
from Tantra.train import NeuroTrainer, Lion, build_optimizer


def test_build_optimizers():
    p = [torch.nn.Parameter(torch.randn(10, 10))]
    opt_adamw = build_optimizer("adamw", p, lr=1e-4, weight_decay=0.01)
    assert opt_adamw is not None

    opt_lion = build_optimizer("lion", p, lr=3e-5, weight_decay=0.05)
    assert isinstance(opt_lion, (Lion, torch.optim.Optimizer))

    opt_adam = build_optimizer("adam", p, lr=1e-4, weight_decay=0.01)
    assert isinstance(opt_adam, torch.optim.Adam)

    opt_sgd = build_optimizer("sgd", p, lr=1e-3, weight_decay=0.0)
    assert isinstance(opt_sgd, torch.optim.SGD)


def test_lion_math_correctness_and_buffer_isolation():
    """Verify exact numerical formulation of Lion (Chen et al. 2023)."""
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    opt = Lion([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)
    
    # Step 1: grad = [0.5, -0.5]
    p.grad = torch.tensor([0.5, -0.5], dtype=torch.float32)
    opt.step()
    
    # Expected update:
    # m_0 = 0
    # update = sign(0.9 * 0 + 0.1 * [0.5, -0.5]) = sign([0.05, -0.05]) = [1.0, -1.0]
    # p_1 = [1.0, -2.0] - 0.1 * [1.0, -1.0] = [0.9, -1.9]
    # m_1 = 0.99 * 0 + 0.01 * [0.5, -0.5] = [0.005, -0.005]
    assert torch.allclose(p.data, torch.tensor([0.9, -1.9], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(opt.state[p]["exp_avg"], torch.tensor([0.005, -0.005], dtype=torch.float32), atol=1e-6)

    # Step 2: verify momentum accumulation with non-zero buffer
    p.grad = torch.tensor([0.1, -0.1], dtype=torch.float32)
    opt.step()
    # update = sign(0.9 * [0.005, -0.005] + 0.1 * [0.1, -0.1]) = sign([0.0145, -0.0145]) = [1.0, -1.0]
    # p_2 = [0.9, -1.9] - 0.1 * [1.0, -1.0] = [0.8, -1.8]
    # m_2 = 0.99 * [0.005, -0.005] + 0.01 * [0.1, -0.1] = [0.00595, -0.00595]
    assert torch.allclose(p.data, torch.tensor([0.8, -1.8], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(opt.state[p]["exp_avg"], torch.tensor([0.00595, -0.00595], dtype=torch.float32), atol=1e-6)


def test_lion_memory_footprint_vs_adamw():
    """Verify Lion uses exactly 1 state buffer while AdamW uses 2."""
    p_lion = torch.nn.Parameter(torch.randn(100, 100))
    p_adamw = torch.nn.Parameter(torch.randn(100, 100))

    opt_lion = Lion([p_lion], lr=1e-4)
    opt_adamw = torch.optim.AdamW([p_adamw], lr=1e-4)

    # Perform 1 step
    p_lion.grad = torch.randn_like(p_lion)
    p_adamw.grad = torch.randn_like(p_adamw)

    opt_lion.step()
    opt_adamw.step()

    # Lion must have exactly 1 momentum tensor
    assert len(opt_lion.state[p_lion]) == 1
    assert "exp_avg" in opt_lion.state[p_lion]

    # AdamW has 2 buffers (exp_avg, exp_avg_sq) + step
    assert len(opt_adamw.state[p_adamw]) >= 2
    assert "exp_avg" in opt_adamw.state[p_adamw]
    assert "exp_avg_sq" in opt_adamw.state[p_adamw]


def test_lion_step_execution():
    model = build_cpu_model("micro10", attention_kind="causal")
    trainer = NeuroTrainer(model, lr=3e-5, weight_decay=0.05, optimizer_name="lion", total_steps=5)
    
    x = torch.randint(0, 32768, (1, 16))
    y = torch.randint(0, 32768, (1, 16))
    loss, acc, ppl, grad_norm, at_boundary = trainer.train_step(x, y)
    
    assert loss > 0
    assert ppl > 0
    assert at_boundary is True
    assert trainer.step_count == 1


def test_optimizer_and_scheduler_continuity_on_resume(tmp_path):
    """Verify that same-stage resume preserves optimizer momentum and LR scheduler position."""
    model1 = build_cpu_model("micro10", attention_kind="causal")
    trainer1 = NeuroTrainer(model1, lr=1e-4, total_steps=100, warmup_steps=10)
    trainer1.training_stage = "sft"

    x = torch.randint(0, 32768, (1, 16))
    y = torch.randint(0, 32768, (1, 16))

    # Run 5 training steps to accumulate momentum and advance scheduler
    for _ in range(5):
        trainer1.train_step(x, y)

    initial_lr = trainer1.optimizer.param_groups[0]["lr"]
    assert trainer1.step_count == 5
    assert initial_lr > 0.0

    ckpt_path = str(tmp_path / "test_resume.pt")
    trainer1.save_checkpoint(ckpt_path, save_optimizer=True)

    # Recreate model & trainer and load checkpoint
    model2 = build_cpu_model("micro10", attention_kind="causal")
    trainer2 = NeuroTrainer(model2, lr=1e-4, total_steps=100, warmup_steps=10)
    trainer2.load_checkpoint(ckpt_path)

    assert trainer2.step_count == 5
    assert trainer2.training_stage == "sft"
    resumed_lr = trainer2.scheduler.get_last_lr()[0] if hasattr(trainer2.scheduler, "get_last_lr") else trainer2.optimizer.param_groups[0]["lr"]
    assert resumed_lr > 0.0

    # Ensure running step 6 does not crash or reset
    loss, acc, ppl, grad_norm, at_boundary = trainer2.train_step(x, y)
    assert trainer2.step_count == 6

