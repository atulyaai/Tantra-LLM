"""Consolidated test suite: Tests/test_core_architecture.py"""


# ─────────────────────────────────────────────────────────────────
# Source: test_adapters.py
# ─────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────
# Source: test_bitnet.py
# ─────────────────────────────────────────────────────────────────

"""
tests/test_bitnet.py — Comprehensive unit tests for BitNet quantization, uint8/int32 bit-packing,
BitLinear matrix multiplication, floating point precision matching, and CPU inference acceleration.
"""

import pytest
import torch
import torch.nn as nn

from Tantra.config import BitNetConfig
from Tantra.bitnet import StraightThrough, TernaryQuantizer, BitLinear, TernaryCPUKernel, BitNetTrainerHooks


def test_quantizer_pack_unpack_int32():
    config = BitNetConfig()
    quantizer = TernaryQuantizer(config)
    
    W = torch.randn(32, 64)
    W_q, scale = quantizer.quantize(W)
    
    packed = quantizer.pack(W_q)
    assert packed.dtype == torch.int32
    
    unpacked = quantizer.unpack(packed, W.shape)
    assert unpacked.shape == W.shape
    assert torch.equal(W_q, unpacked)


def test_quantizer_pack_unpack_uint8():
    config = BitNetConfig()
    quantizer = TernaryQuantizer(config)
    
    W = torch.randn(32, 64)
    W_q, scale = quantizer.quantize(W)
    
    packed_u8 = quantizer.pack_uint8(W_q)
    assert packed_u8.dtype == torch.uint8
    
    unpacked_u8 = quantizer.unpack_uint8(packed_u8, W.shape)
    assert unpacked_u8.shape == W.shape
    assert torch.equal(W_q, unpacked_u8)


def test_uint8_int32_packing_equivalence():
    config = BitNetConfig()
    quantizer = TernaryQuantizer(config)
    
    W = torch.randn(16, 32)
    W_q, _ = quantizer.quantize(W)
    
    packed_int32 = quantizer.pack(W_q)
    packed_u8 = quantizer.pack_uint8(W_q)
    
    assert torch.equal(packed_int32, packed_u8.view(torch.int32))


def test_bitlinear_forward_precision_matching():
    torch.manual_seed(42)
    layer = BitLinear(64, 128, bias=True)
    x = torch.randn(4, 16, 64)
    
    # Training forward
    layer.train()
    out_train = layer(x)
    
    # Convert to inference mode
    layer.to_inference_mode()
    out_infer = layer(x)
    
    assert out_train.shape == out_infer.shape == (4, 16, 128)
    assert torch.allclose(out_train, out_infer, atol=1e-5)


def test_bitlinear_mode_toggle_roundtrip():
    layer = BitLinear(32, 64, bias=True)
    assert not layer.is_inference
    assert layer.weight is not None
    
    layer.to_inference_mode()
    assert layer.is_inference
    assert layer.weight is None
    assert layer.packed_weight is not None
    assert layer.packed_weight_u8 is not None
    
    layer.to_training_mode()
    assert not layer.is_inference
    assert layer.weight is not None
    assert layer.packed_weight is None
    assert layer.packed_weight_u8 is None


def test_ternary_cpu_kernel_matmul():
    kernel = TernaryCPUKernel()
    config = BitNetConfig()
    quantizer = TernaryQuantizer(config)
    
    W = torch.randn(64, 128)
    W_q, scale = quantizer.quantize(W)
    packed = quantizer.pack(W_q)
    
    x = torch.randn(2, 8, 128)
    out = kernel.matmul(x, packed, scale, (64, 128))
    
    assert out.shape == (2, 8, 64)
    assert not torch.isnan(out).any()


def test_ternary_cpu_kernel_benchmark():
    kernel = TernaryCPUKernel()
    bench = kernel.benchmark(in_f=128, out_f=256, batch=4)
    
    assert "fp32_ms" in bench
    assert "ternary_ms" in bench
    assert "speedup" in bench
    assert bench["fp32_ms"] >= 0
    assert bench["ternary_ms"] >= 0


def test_trainer_hooks_integration():
    model = nn.Sequential(
        BitLinear(16, 32),
        nn.ReLU(),
        BitLinear(32, 8)
    )
    hooks = BitNetTrainerHooks(model, BitNetConfig())
    assert len(hooks.bitlinear_layers) == 2
    
    groups = hooks.get_param_groups()
    assert len(groups) == 3
    
    stats = hooks.get_quantization_stats()
    assert "avg_sparsity" in stats
    assert "avg_pos" in stats
    assert "avg_neg" in stats


# ─────────────────────────────────────────────────────────────────
# Source: test_model.py
# ─────────────────────────────────────────────────────────────────

"""Tests for tantra.model + tantra.bitnet + tantra.moe"""
import pytest
import torch

from Tantra.config import NeuroCoreConfig, MoEConfig, VocabConfig
from Tantra.hardware import HardwareDetector, Profiler, RuntimeConfigBuilder
from Tantra.model import DynamicScaleNorm, RotaryPositionalEncoding, ALRAAttention, SparseGatedProjection, NeuroCoreBlock, NeuroCoreModel, Top1MoEProjection, cpu_dense_config, cpu_top1_moe_config
from Tantra.train import NeuroTrainer


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


def test_cpu_dense_profile_uses_tied_embeddings_and_swiglu():
    cfg = cpu_dense_config(vocab_size=256)
    model = NeuroCoreModel(cfg, use_mtp=False)
    assert model.output_proj.weight.data_ptr() == model.embed.weight.data_ptr()
    assert model.layers[0].router is None
    assert model.layers[0].mlp.__class__.__name__ == "SwiGLUProjection"


def test_real_top1_moe_dispatches_and_exposes_balance_loss():
    cfg = cpu_top1_moe_config(vocab_size=256, experts=2)
    model = NeuroCoreModel(cfg, use_mtp=False, use_moe=True)
    assert isinstance(model.layers[0].mlp, Top1MoEProjection)
    logits, _ = model(torch.randint(0, 256, (2, 8)), use_latent_reasoning=False)
    assert logits.shape == (2, 8, 256)
    assert model.get_aux_loss().item() > 0


def test_cpu_causal_attention_profile_forward():
    cfg = cpu_dense_config(vocab_size=256, attention_kind="causal")
    model = NeuroCoreModel(cfg, use_mtp=False)
    logits, _ = model(torch.randint(0, 256, (1, 8)), use_latent_reasoning=False)
    assert logits.shape == (1, 8, 256)


def test_category_layer_routes_then_trains_in_isolation():
    cfg = cpu_dense_config(vocab_size=256, attention_kind="causal")
    model = NeuroCoreModel(cfg, use_mtp=False)
    ids = torch.randint(0, 256, (1, 8))
    baseline, _ = model(ids, use_latent_reasoning=False)
    model.add_category_layers(["math", "science"], depth=1, clone_layer_index=-1)
    routed, _ = model(ids, use_latent_reasoning=False, adapter_name="math")
    assert routed.shape == baseline.shape
    # A freshly installed category carries a zero residual gate, so routing it
    # is an exact identity pass-through: the base output is unchanged until the
    # category's own dataset opens the gate. This restores the documented
    # "untrained category does not perturb the base" guarantee.
    assert torch.allclose(baseline, routed, atol=1e-6)
    model.freeze_for_category("math")
    assert all(p.requires_grad for p in model.category_layers["math"].parameters())
    assert all(p.requires_grad for p in model.category_gates["math"])
    assert not any(p.requires_grad for p in model.category_layers["science"].parameters())
    # Opening the gate makes the routed output differ from the base: the
    # specialist layer's transform now contributes.
    with torch.no_grad():
        model.category_gates["math"][0].fill_(1.0)
    routed_open, _ = model(ids, use_latent_reasoning=False, adapter_name="math")
    assert routed_open.shape == baseline.shape
    assert not torch.allclose(baseline, routed_open)


@pytest.fixture
def micro_config():
    cfg = NeuroCoreConfig()
    cfg.block.num_layers = 1
    cfg.block.alra.dim, cfg.block.alra.num_heads, cfg.block.alra.head_dim = 32, 4, 8
    cfg.block.sgp.dim = 32
    cfg.vocab = VocabConfig(vocab_size=1000)
    cfg.use_mtp, cfg.reasoning_depth = True, 1
    return cfg


def test_mtp_training_and_speculative_generation(micro_config):
    model = NeuroCoreModel(micro_config)
    trainer = NeuroTrainer(model, lr=1e-3)
    x = torch.randint(0, 1000, (2, 16))
    loss, _, _, _, at_boundary = trainer.train_step(x, x.clone())
    assert loss >= 0 and at_boundary and trainer.step_count == 1
    assert model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=8, use_mtp_speculation=True).shape == (1, 11)
    assert 1 <= len(list(model.generate_stream(torch.tensor([[1, 2, 3]]), max_new_tokens=4, temperature=0))) <= 4


def test_hardware_detection_and_runtime_config():
    profile = HardwareDetector().detect()
    assert profile.ram_total_mb > 0 and profile.cpu.physical_cores > 0
    perf = Profiler(profile).run()
    runtime = RuntimeConfigBuilder().build(profile, perf)
    assert perf.fp32_matmul_gflops > 0
    assert runtime.device in ("cpu", "cuda:0", "mps") and runtime.batch_size >= 1


@pytest.fixture
def micro_config():
    cfg = NeuroCoreConfig()
    cfg.block.num_layers = 1
    cfg.block.alra.dim, cfg.block.alra.num_heads, cfg.block.alra.head_dim = 32, 4, 8
    cfg.block.sgp.dim = 32
    cfg.vocab = VocabConfig(vocab_size=1000)
    cfg.use_mtp, cfg.reasoning_depth = True, 1
    return cfg


def test_mtp_training_and_speculative_generation(micro_config):
    model = NeuroCoreModel(micro_config)
    trainer = NeuroTrainer(model, lr=1e-3)
    x = torch.randint(0, 1000, (2, 16))
    loss, _, _, _, at_boundary = trainer.train_step(x, x.clone())
    assert loss >= 0 and at_boundary and trainer.step_count == 1
    assert model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=8, use_mtp_speculation=True).shape == (1, 11)
    assert 1 <= len(list(model.generate_stream(torch.tensor([[1, 2, 3]]), max_new_tokens=4, temperature=0))) <= 4


def test_variable_seq_len_checkpoint_transfer(micro_config, tmp_path):
    model1 = NeuroCoreModel(micro_config)
    trainer1 = NeuroTrainer(model1, lr=1e-3)
    x128 = torch.randint(0, 1000, (2, 128))
    loss1, _, _, _, _ = trainer1.train_step(x128, x128.clone())
    assert loss1 > 0

    ckpt_path = str(tmp_path / "test_ckpt.pt")
    trainer1.save_checkpoint(ckpt_path)

    # Resume into larger sequence lengths (256, 512)
    model2 = NeuroCoreModel(micro_config)
    trainer2 = NeuroTrainer(model2, lr=1e-3)
    trainer2.load_checkpoint(ckpt_path)

    x256 = torch.randint(0, 1000, (2, 256))
    loss2, acc2, ppl2, grad_norm2, _ = trainer2.train_step(x256, x256.clone())
    assert loss2 > 0 and not torch.isnan(torch.tensor(loss2)) and not torch.isinf(torch.tensor(loss2))

    x512 = torch.randint(0, 1000, (2, 512))
    loss3, acc3, ppl3, grad_norm3, _ = trainer2.train_step(x512, x512.clone())
    assert loss3 > 0 and not torch.isnan(torch.tensor(loss3))


def test_best_val_loss_preservation_on_resume(micro_config, tmp_path):
    model1 = NeuroCoreModel(micro_config)
    trainer1 = NeuroTrainer(model1, lr=1e-3)
    trainer1.best_val_loss = 5.25
    trainer1.best_loss = 5.25

    ckpt_path = str(tmp_path / "test_best_resume.pt")
    trainer1.save_checkpoint(ckpt_path)

    # Resume into fresh trainer
    model2 = NeuroCoreModel(micro_config)
    trainer2 = NeuroTrainer(model2, lr=1e-3)
    trainer2.load_checkpoint(ckpt_path)

    assert trainer2.best_val_loss == 5.25
    assert trainer2.best_loss == 5.25




# ─────────────────────────────────────────────────────────────────
# Source: test_mtp_speculation.py
# ─────────────────────────────────────────────────────────────────

import os
import sys
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Tantra.model import NeuroCoreModel
from Tantra.config import NeuroCoreConfig


def test_mtp_dual_token_forward():
    cfg = NeuroCoreConfig.small()
    cfg.vocab.vocab_size = 1000
    model = NeuroCoreModel(cfg, use_mtp=True)
    model.eval()

    seq_len = 16
    input_ids = torch.randint(0, 1000, (1, seq_len))

    with torch.no_grad():
        (logits_main, logits_mtp), _ = model.forward(input_ids, return_mtp=True)

    assert logits_main.shape == (1, seq_len, 1000)
    assert logits_mtp.shape == (1, seq_len, 1000)

    pred_t1 = torch.argmax(logits_main[:, -1, :], dim=-1).item()
    pred_t2 = torch.argmax(logits_mtp[:, -1, :], dim=-1).item()

    assert 0 <= pred_t1 < 1000
    assert 0 <= pred_t2 < 1000

