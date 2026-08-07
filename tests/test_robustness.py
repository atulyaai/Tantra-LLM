"""
tests/test_robustness.py — Comprehensive edge case & robustness stress test suite for Tantra-LLM.
Tests empty inputs, giant sequence lengths, single token prompts, invalid token IDs, NaNs, shape matching, and boundary conditions.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from tantra.config import NeuroCoreConfig, BitNetConfig, CompressionConfig, MoEConfig, VocabConfig
from tantra.bitnet import StraightThrough, TernaryQuantizer, BitLinear, TernaryCPUKernel
from tantra.codec import DNACodec, ResidualPredictor, ZSTDDictTrainer, AdaptiveHuffmanCoder
from tantra.model import NeuroCoreModel, ALRAAttention, DynamicScaleNorm, RotaryPositionalEncoding, SparseGatedProjection
from tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from tantra.evolution import AutoGrowthController, SelfRepairEngine
from tantra.moe import MoERouter, LoadBalancer, ExpertRegistry, LazyExpertLoader
from tantra.dataset import JSONLDataset, format_jsonl_prompt
from tantra.eval import EvaluationEngine
from tantra.train import NeuroTrainer, generate_synthetic_batch


# ── 1. BitNet Edge Cases & Stress Tests ──────────────────────────────────────

def test_bitnet_empty_tensor():
    cfg = BitNetConfig()
    quantizer = TernaryQuantizer(cfg)
    empty_w = torch.empty((0, 10), dtype=torch.float32)
    
    w_q, scale = quantizer.quantize(empty_w)
    assert w_q.numel() == 0
    assert not torch.isnan(scale)

    packed = quantizer.pack(w_q)
    assert packed.numel() == 0

    unpacked = quantizer.unpack(packed, (0, 10))
    assert unpacked.shape == (0, 10)


def test_bitnet_nan_and_inf_inputs():
    x = torch.tensor([float('nan'), float('inf'), -float('inf'), 0.5, -0.5])
    out = StraightThrough.apply(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_bitlinear_inference_mode_edge_cases():
    layer = BitLinear(16, 32, bias=True)
    layer.to_inference_mode()
    
    # Batch 1, Seq 1
    x_single = torch.randn(1, 1, 16)
    out_single = layer(x_single)
    assert out_single.shape == (1, 1, 32)
    assert not torch.isnan(out_single).any()

    # 1D Input
    x_1d = torch.randn(16)
    out_1d = layer(x_1d)
    assert out_1d.shape == (32,)


# ── 2. Codec & DNA Compression Robustness ───────────────────────────────────

def test_dna_codec_various_dtypes_and_shapes():
    cfg = CompressionConfig()
    codec = DNACodec(cfg)
    
    for dtype in [torch.float32, torch.float16, torch.int32, torch.int8]:
        tensor = torch.randn(64, 64).to(dtype) if dtype.is_floating_point else torch.randint(-50, 50, (64, 64)).to(dtype)
        with tempfile.NamedTemporaryFile(suffix=".dna", delete=False) as tmp:
            path = tmp.name
        try:
            stats = codec.compress(tensor, path)
            assert stats.sha256_match
            decompressed = codec.decompress(path)
            assert decompressed.shape == tensor.shape
            assert decompressed.dtype == tensor.dtype
        finally:
            if os.path.exists(path):
                os.remove(path)


def test_residual_predictor_zero_epochs():
    cfg = CompressionConfig()
    predictor = ResidualPredictor(cfg)
    res = predictor.train_on_tensors([], epochs=0)
    assert "final_loss" in res
    assert not torch.isnan(torch.tensor(res["final_loss"]))


# ── 3. Model Architecture & RoPE Shape Mismatch Tests ─────────────────────────

def test_rope_heads_equals_seq_len_bug_fix():
    """Stress test RoPE when num_heads == seq_len (e.g. 32 heads, seq_len = 32)."""
    rope = RotaryPositionalEncoding(head_dim=64)
    num_heads = 32
    seq_len = 32
    q = torch.randn(2, num_heads, seq_len, 64)
    k = torch.randn(2, num_heads, seq_len, 64)
    
    q_rot, k_rot = rope.apply(q, k, seq_len)
    assert q_rot.shape == (2, num_heads, seq_len, 64)
    assert k_rot.shape == (2, num_heads, seq_len, 64)
    assert not torch.isnan(q_rot).any()


def test_model_giant_sequence_length():
    cfg = NeuroCoreConfig.small()
    cfg.block.num_layers = 1
    cfg.block.alra.dim = 32
    cfg.block.alra.num_heads = 4
    cfg.block.alra.head_dim = 8
    cfg.block.sgp.dim = 32
    model = NeuroCoreModel(cfg)
    model.eval()
    
    # Giant sequence length = 1024 tokens
    giant_input = torch.randint(0, cfg.vocab.vocab_size, (1, 1024))
    with torch.no_grad():
        logits, _ = model(giant_input)
    assert logits.shape == (1, 1024, cfg.vocab.vocab_size)
    assert not torch.isnan(logits).any()


def test_model_single_token_prompt_and_empty_prompt():
    cfg = NeuroCoreConfig.small()
    cfg.block.num_layers = 1
    cfg.block.alra.dim = 32
    cfg.block.alra.num_heads = 4
    cfg.block.alra.head_dim = 8
    cfg.block.sgp.dim = 32
    model = NeuroCoreModel(cfg)
    model.eval()
    
    # Single token prompt
    single_token = torch.tensor([[42]])
    gen_single = model.generate(single_token, max_new_tokens=5)
    assert gen_single.shape == (1, 6)

    # Empty prompt (0 seq length)
    empty_prompt = torch.tensor([[]], dtype=torch.long)
    gen_empty = model.generate(empty_prompt, max_new_tokens=5)
    assert gen_empty.size(1) >= 5
    assert not torch.isnan(gen_empty.float()).any()


# ── 4. MoE & Routing Edge Cases ──────────────────────────────────────────────

def test_moe_router_empty_input():
    cfg = MoEConfig()
    router = MoERouter(cfg, embed_dim=128)
    empty_x = torch.empty((0, 0, 128))
    weights, experts, _ = router(empty_x)
    assert weights.numel() == 0
    
    loss = router.load_balancing_loss(experts)
    assert not torch.isnan(loss)
    assert loss.item() == 0.0


def test_load_balancer_zero_coeff():
    lb = LoadBalancer(num_experts=8, coeff=0.0)
    probs = torch.rand(2, 16, 8)
    loss = lb(probs)
    assert loss.item() == 0.0


# ── 5. Tokenizer & Out-of-Vocabulary Robustness ──────────────────────────────

def test_tokenizer_out_of_bounds_token_ids():
    cfg = VocabConfig()
    bpe = ByteBPETokenizer(cfg)
    patcher = MegabytePatcher()
    tok = UnifiedTokenizer(cfg, bpe, patcher)
    
    # Invalid negative and huge token IDs
    invalid_ids = [-100, 9999999, 0, 255]
    decoded = tok.decode(invalid_ids, modality="text")
    assert isinstance(decoded, str)


def test_megabyte_patcher_empty_codebook():
    patcher = MegabytePatcher()
    # Codebook is None
    decoded = patcher.decode_to_bytes([1, 2, 3])
    assert len(decoded) == 3 * patcher.patch_size


# ── 6. Self-Repair & Auto-Growth Robustness ──────────────────────────────────

def test_self_repair_engine_corrupted_model():
    cfg = NeuroCoreConfig.small()
    cfg.block.num_layers = 1
    cfg.block.alra.dim = 32
    cfg.block.alra.num_heads = 4
    cfg.block.alra.head_dim = 8
    cfg.block.sgp.dim = 32
    model = NeuroCoreModel(cfg)
    
    # Inject NaNs, Infs, and dead neurons manually
    with torch.no_grad():
        model.embed.weight[0, 0] = float('nan')
        model.embed.weight[1, 0] = float('inf')
        model.layers[0].attn.w_q.weight.data.fill_(1000.0)  # exploded
        model.layers[0].mlp.w_up.weight.data[0].fill_(0.0)  # dead neuron
        
    repair = SelfRepairEngine()
    stats = repair.scan_and_repair(model, max_norm=50.0)
    
    assert stats["repaired_nans"] >= 2
    assert stats["repaired_explosions"] >= 1
    assert stats["repaired_dead"] >= 1
    
    # Verify no NaNs remain
    for p in model.parameters():
        assert not torch.isnan(p.data).any()
        assert not torch.isinf(p.data).any()


def test_auto_growth_empty_layers():
    controller = AutoGrowthController()
    dummy_model = nn.Module()
    dummy_model.layers = nn.ModuleList()
    
    # Grow on empty layers list should not crash
    controller.grow_capacity(dummy_model)
    assert len(dummy_model.layers) == 0


# ── 7. Config Integrity Test ──────────────────────────────────────────────────

def test_config_save_and_load_integrity():
    cfg = NeuroCoreConfig.small()
    cfg.block.num_layers = 17
    cfg.moe.num_experts = 12
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    try:
        cfg.save(path)
        loaded = NeuroCoreConfig.load(path)
        assert loaded.model_name == cfg.model_name
        assert loaded.block.num_layers == 17
        assert loaded.moe.num_experts == 12
    finally:
        if os.path.exists(path):
            os.remove(path)
