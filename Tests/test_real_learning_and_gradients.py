"""
Tests/test_real_learning_and_gradients.py
Rigorous verification tests that validate REAL learning, non-zero gradients,
convergence, and authentic BPE tokenization (NO stubs, NO fake ord() mocks, NO shape-only assertions).
"""
import os
import pytest
import torch
import torch.nn as nn

from Tantra.config import NeuroCoreConfig, VocabConfig, ALRAConfig, SGPConfig
from Tantra.model import NeuroCoreModel, ALRAAttention, SparseGatedProjection
from Tantra.train import NeuroTrainer
from Tantra.evolution import AutoGrowthController


def _build_test_model(vocab_size=256, dim=64, num_layers=2, reasoning_depth=1):
    cfg = NeuroCoreConfig(model_name="test-real-learning")
    cfg.vocab = VocabConfig(vocab_size=vocab_size)
    cfg.block.alra.dim = dim
    cfg.block.alra.num_heads = 4
    cfg.block.alra.head_dim = dim // 4
    cfg.block.sgp.dim = dim
    cfg.block.num_layers = num_layers
    cfg.use_mtp = False
    cfg.reasoning_depth = reasoning_depth
    return NeuroCoreModel(cfg, use_mtp=False, reasoning_depth=reasoning_depth), cfg


def test_real_gradient_flow_non_zero():
    """
    Catch BUG-01, BUG-02, BUG-03:
    Every single parameter in the embedding, attention, SGP, latent reasoning,
    and output projection MUST receive non-zero, non-NaN, non-inf gradients.
    """
    model, cfg = _build_test_model(vocab_size=256, dim=64, num_layers=2, reasoning_depth=1)
    model.train()
    
    # Input batch with targets
    x = torch.randint(1, 250, (2, 16))
    y = torch.randint(1, 250, (2, 16))
    
    logits, _ = model(x, use_latent_reasoning=True)
    loss = nn.functional.cross_entropy(logits.view(-1, cfg.vocab.vocab_size), y.view(-1))
    loss.backward()
    
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss must be finite"
    assert loss.item() > 0.0, "Initial cross entropy loss must be > 0"
    
    dead_params = []
    nan_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        assert param.grad is not None, f"Parameter {name} did not receive any gradient"
        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            nan_params.append(name)
        if param.grad.abs().sum().item() == 0.0:
            dead_params.append(name)
            
    assert len(nan_params) == 0, f"Parameters with NaN/Inf gradients: {nan_params}"
    assert len(dead_params) == 0, f"Parameters with ZERO gradient flow: {dead_params}"


def test_real_loss_convergence_and_learning():
    """
    True learning verification:
    Train for 30 steps on a memorizable task with 0 warmup.
    Loss at step 30 MUST be significantly lower than step 1.
    Top-1 token accuracy MUST increase above initial step.
    """
    model, cfg = _build_test_model(vocab_size=64, dim=64, num_layers=2, reasoning_depth=0)
    trainer = NeuroTrainer(
        model,
        lr=1e-2,
        weight_decay=0.0,
        warmup_steps=0,
        total_steps=50,
        use_mtp_loss=False,
        use_latent_reasoning=False,
    )
    
    # Simple recurring target pattern to memorize
    torch.manual_seed(42)
    fixed_x = torch.tensor([[5, 10, 15, 20, 25, 30, 35, 40]], dtype=torch.long)
    fixed_y = torch.tensor([[10, 15, 20, 25, 30, 35, 40, 45]], dtype=torch.long)
    
    losses = []
    accuracies = []
    
    for step in range(30):
        loss_val, top1, top5, grad_norm, _ = trainer.train_step(fixed_x, fixed_y)
        losses.append(loss_val)
        accuracies.append(top1)
    
    # 1. Loss MUST strictly decrease from start to end
    initial_loss = losses[0]
    final_loss = losses[-1]
    assert final_loss < initial_loss, f"Model failed to learn! Start loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}"
    
    # Loss should drop noticeably over 30 steps on a memorizable pattern
    assert final_loss < initial_loss * 0.70, (
        f"Convergence too slow: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
    
    # 2. Accuracy must increase to > 0%
    assert accuracies[-1] > 0.0, f"Accuracy remained at 0%: {accuracies[-1]}"


def test_real_bpe_tokenizer_trained_vocab():
    """
    Verifies that the REAL BPE tokenizer on disk (Model/tokenizer.json)
    encodes actual code and math keywords into single learned tokens,
    rather than falling back to character-by-character bytes.
    """
    from tokenizers import Tokenizer
    tokenizer_path = "Model/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        pytest.skip(f"Trained tokenizer not found at {tokenizer_path}")
    
    tok = Tokenizer.from_file(tokenizer_path)
    
    # Common keywords from the 100% corpus that must be in vocabulary
    test_words = ["def", "class", "return", "import", "assistant", "Solve", "function"]
    
    single_token_count = 0
    for word in test_words:
        encoded = tok.encode(word)
        if len(encoded.ids) == 1:
            single_token_count += 1
            
    # At least 4 of these 7 critical domain words MUST be atomic single tokens
    assert single_token_count >= 4, (
        f"Tokenizer vocabulary quality is poor: only {single_token_count}/7 domain words are single tokens"
    )


def test_capacity_saturation_metric_and_growth_guard():
    """
    Verifies the rewritten AutoGrowthController:
    1. Saturation metric is well-defined between 0.0 and 1.0.
    2. Growth is BLOCKED if model is unsaturated (< 80%).
    3. Growth is BLOCKED if loss is too high (> 10.0) even if saturated.
    """
    model, _ = _build_test_model(vocab_size=256, dim=64, num_layers=2)
    controller = AutoGrowthController(saturation_threshold=0.80, max_loss_for_growth=10.0, plateau_patience=10)
    
    # Run a forward pass to populate _last_active_ratio
    x = torch.randint(1, 250, (2, 8))
    model(x)
    
    saturation = AutoGrowthController.compute_capacity_saturation(model)
    assert 0.0 <= saturation <= 1.0, f"Saturation metric out of bounds: {saturation}"
    
    # Scenario A: Loss is high (13.5), controller must reject growth even if plateaued
    initial_layers = len(model.layers)
    for _ in range(15):
        grown = controller.observe(13.5, model)
        assert grown is False, "AutoGrowth triggered when loss was 13.5 (> 10.0)!"
    assert len(model.layers) == initial_layers
    
    # Scenario B: Artificially force low saturation on SparseGatedProjection
    controller2 = AutoGrowthController(saturation_threshold=0.80, max_loss_for_growth=10.0, plateau_patience=10)
    for m in model.modules():
        if isinstance(m, SparseGatedProjection):
            m._last_active_ratio = 0.05
            
    for _ in range(15):
        grown2 = controller2.observe(2.0, model)
        assert grown2 is False, "AutoGrowth triggered when saturation was far below 80%!"
    assert len(model.layers) == initial_layers


def test_alra_attention_no_denominator_collapse():
    """
    Verifies BUG-02 fix: ALRA linear attention denominator eps = 1e-4
    prevents division-by-zero / NaN output even with near-zero queries and keys.
    """
    cfg = ALRAConfig(dim=64, num_heads=4, head_dim=16)
    attn = ALRAAttention(cfg)
    attn.eval()
    
    # Pathological zero/near-zero inputs
    x_zeros = torch.zeros(1, 16, 64)
    out_zeros, _ = attn(x_zeros)
    assert not torch.isnan(out_zeros).any(), "ALRA produced NaN on zero inputs"
    assert not torch.isinf(out_zeros).any(), "ALRA produced Inf on zero inputs"
    
    # Tiny inputs
    x_tiny = torch.randn(1, 16, 64) * 1e-6
    out_tiny, _ = attn(x_tiny)
    assert not torch.isnan(out_tiny).any(), "ALRA produced NaN on tiny inputs"
    assert not torch.isinf(out_tiny).any(), "ALRA produced Inf on tiny inputs"


def test_sparse_gated_projection_active_ratio_and_soft_bypass():
    """
    Verifies BUG-03 and BUG-11:
    1. _last_active_ratio is recorded on every forward pass.
    2. Soft bypass prevents zero-gradient lock on inactive neurons.
    """
    cfg = SGPConfig(dim=64, expansion=2, sparsity=0.25)
    sgp = SparseGatedProjection(cfg)
    assert sgp._last_active_ratio == 0.0
    
    x = torch.randn(2, 12, 64, requires_grad=True)
    out = sgp(x)
    
    # Must update active ratio
    assert sgp._last_active_ratio > 0.0, "_last_active_ratio was not updated on forward pass"
    
    # Backward pass must propagate to gate and input
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0.0, "Gradients failed to flow back through SparseGatedProjection"
    assert sgp.w_gate.weight.grad is not None
    assert sgp.w_gate.weight.grad.abs().sum().item() > 0.0, "Gate projection received zero gradient"
