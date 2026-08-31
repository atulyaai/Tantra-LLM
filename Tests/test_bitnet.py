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
