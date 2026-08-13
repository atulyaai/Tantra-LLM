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


def test_hardware_detection_and_runtime_config():
    profile = HardwareDetector().detect()
    assert profile.ram_total_mb > 0 and profile.cpu.physical_cores > 0
    perf = Profiler(profile).run()
    runtime = RuntimeConfigBuilder().build(profile, perf)
    assert perf.fp32_matmul_gflops > 0
    assert runtime.device in ("cpu", "cuda:0", "mps") and runtime.batch_size >= 1
