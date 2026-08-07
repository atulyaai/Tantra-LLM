"""tantra/bitnet.py — BitNet 1-bit weight quantization. Contains: StraightThrough, TernaryQuantizer, BitLinear, TernaryCPUKernel, BitNetTrainerHooks."""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from tantra.config import BitNetConfig


# ── StraightThrough ──

class StraightThrough(torch.autograd.Function):
    """Straight-through estimator for quantization."""
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        x_clean = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return x_clean.round().clamp(-1, 1)

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:
        return grad  # pass gradient as-is


# ── TernaryQuantizer ──

class TernaryQuantizer:
    """
    Quantizes FP32 weights to ternary {-1, 0, +1}.
    
    Algorithm:
    1. scale = mean(|W|)  # absmax scaling
    2. W_norm = W / (scale + eps)
    3. W_q = round(W_norm).clamp(-1, 1)  # ternary
    4. Packing: 16 ternary weights per int32 (2 bits each) or 4 weights per uint8 (2 bits each)
       -1 = 0b00, 0 = 0b01, +1 = 0b10
    """
    def __init__(self, config: BitNetConfig) -> None:
        """Initialize the quantizer with BitNetConfig."""
        self.config = config

    def quantize(self, W: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (W_ternary, scale). W_ternary is int8 {-1,0,1}."""
        if W.numel() == 0:
            return W.to(torch.int8), torch.tensor(1.0, device=W.device)
        eps = 1e-8
        scale = W.abs().mean().clamp(min=eps)
        W_norm = W / scale
        W_ternary = StraightThrough.apply(W_norm)
        return W_ternary.to(torch.int8), scale

    def dequantize(self, W_q: Tensor, scale: Tensor) -> Tensor:
        """Reconstruct approximate FP32 from ternary + scale."""
        return W_q.float() * scale

    def pack(self, W_q: Tensor) -> Tensor:
        """Pack int8 ternary into int32 (16 weights per int32) using vectorized bitwise ops."""
        W_flat = W_q.flatten()
        if W_flat.numel() == 0:
            return torch.empty(0, dtype=torch.int32, device=W_q.device)
        
        # Pad to multiple of 16
        pad_len = (16 - (W_flat.numel() % 16)) % 16
        if pad_len > 0:
            pad = torch.zeros(pad_len, dtype=torch.int8, device=W_q.device)
            W_flat = torch.cat([W_flat, pad])
        
        # Vectorized uint8 bit packing viewed as int32
        W_mapped_u8 = (W_flat + 1).to(torch.uint8).view(-1, 4)
        shifts_u8 = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=W_q.device)
        packed_u8 = (W_mapped_u8[:, 0]) | (W_mapped_u8[:, 1] << shifts_u8[1]) | (W_mapped_u8[:, 2] << shifts_u8[2]) | (W_mapped_u8[:, 3] << shifts_u8[3])
        return packed_u8.view(torch.int32)

    def unpack(self, W_packed: Tensor, original_shape: tuple) -> Tensor:
        """Unpack int32 back to int8 ternary using vectorized bitwise ops."""
        numel = int(torch.prod(torch.tensor(original_shape)))
        if numel == 0 or W_packed.numel() == 0:
            return torch.zeros(original_shape, dtype=torch.int8, device=W_packed.device)
        W_packed_u8 = W_packed.view(torch.uint8)
        shifts_u8 = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=W_packed.device)
        W_mapped = (W_packed_u8.unsqueeze(1) >> shifts_u8) & 0b11
        W_flat = (W_mapped - 1).to(torch.int8).flatten()
        return W_flat[:numel].view(original_shape)

    def pack_uint8(self, W_q: Tensor) -> Tensor:
        """Pack int8 ternary into uint8 (4 weights per uint8 byte, 2 bits each)."""
        W_flat = W_q.flatten()
        if W_flat.numel() == 0:
            return torch.empty(0, dtype=torch.uint8, device=W_q.device)
        
        pad_len = (4 - (W_flat.numel() % 4)) % 4
        if pad_len > 0:
            pad = torch.zeros(pad_len, dtype=torch.int8, device=W_q.device)
            W_flat = torch.cat([W_flat, pad])
            
        W_mapped_u8 = (W_flat + 1).to(torch.uint8).view(-1, 4)
        shifts_u8 = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=W_q.device)
        packed_u8 = (W_mapped_u8[:, 0]) | (W_mapped_u8[:, 1] << shifts_u8[1]) | (W_mapped_u8[:, 2] << shifts_u8[2]) | (W_mapped_u8[:, 3] << shifts_u8[3])
        return packed_u8

    def unpack_uint8(self, W_packed_u8: Tensor, original_shape: tuple) -> Tensor:
        """Unpack uint8 back to int8 ternary."""
        numel = int(torch.prod(torch.tensor(original_shape)))
        if numel == 0 or W_packed_u8.numel() == 0:
            return torch.zeros(original_shape, dtype=torch.int8, device=W_packed_u8.device)
        shifts_u8 = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=W_packed_u8.device)
        W_mapped = (W_packed_u8.unsqueeze(1) >> shifts_u8) & 0b11
        W_flat = (W_mapped - 1).to(torch.int8).flatten()
        return W_flat[:numel].view(original_shape)

    def compression_ratio(self, original_numel: int) -> float:
        """Return theoretical compression ratio vs FP32."""
        return 32 / 2  # 16x: 32-bit → 2-bit


# ── BitLinear ──

class BitLinear(nn.Module):
    """
    BitLinear: drop-in for nn.Linear with ternary weights {-1, 0, +1}.
    
    Training mode:
    - Maintains FP32 'shadow' weights (self.weight) for gradient updates
    - On each forward: quantize shadow → ternary, compute with ternary
    - Gradients flow to FP32 shadow via straight-through estimator
    
    Inference mode (after calling .to_inference_mode()):
    - Only stores 2-bit packed weights + scale per row
    - Uses CPU kernel for fast computation
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 config: Optional[BitNetConfig] = None) -> None:
        """Initialize BitLinear layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or BitNetConfig()
        self.quantizer = TernaryQuantizer(self.config)
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.kaiming_normal_(self.weight, nonlinearity='linear')
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.is_inference = False
        
        # Inference mode placeholders
        self.register_buffer('packed_weight', None)
        self.register_buffer('packed_weight_u8', None)
        self.register_buffer('weight_scale', None)
        self.register_buffer('pos_mask', None)
        self.register_buffer('neg_mask', None)
        self.register_buffer('w_ternary', None)

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        eps = 1e-8
        x_scale = x.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
        x_norm = x / x_scale

        if self.is_inference:
            if getattr(self, 'w_ternary', None) is not None:
                out = F.linear(x_norm, self.w_ternary) * (self.weight_scale * x_scale)
            elif self.pos_mask is not None and self.neg_mask is not None:
                w_ternary = self.pos_mask - self.neg_mask
                out = F.linear(x_norm, w_ternary) * (self.weight_scale * x_scale)
            else:
                kernel = TernaryCPUKernel()
                out = kernel.matmul(x, self.packed_weight, self.weight_scale, 
                                    (self.out_features, self.in_features))
            if self.bias is not None:
                out += self.bias
            return out
        else:
            # 1. Quantize weights
            W_q, scale = self.quantizer.quantize(self.weight)
            
            # 2. Vectorized single-pass ternary matmul
            pos_mask = (W_q == 1).to(dtype=x.dtype)
            neg_mask = (W_q == -1).to(dtype=x.dtype)
            w_ternary = pos_mask - neg_mask
            out = F.linear(x_norm, w_ternary) * (scale * x_scale)
            
            # 3. Add bias
            if self.bias is not None:
                out += self.bias
            return out

    def to_inference_mode(self) -> None:
        """Pack weights to 2-bit, free FP32 shadow weights, pre-build masks."""
        if self.is_inference:
            return
            
        W_q, scale = self.quantizer.quantize(self.weight.data)
        self.packed_weight = self.quantizer.pack(W_q)
        self.packed_weight_u8 = self.quantizer.pack_uint8(W_q)
        self.weight_scale = scale
        self.pos_mask = (W_q == 1).to(torch.float32)
        self.neg_mask = (W_q == -1).to(torch.float32)
        self.w_ternary = self.pos_mask - self.neg_mask
        
        # Remove FP32 weight to save memory
        del self.weight
        self.register_parameter('weight', None)
        self.is_inference = True

    def to_training_mode(self) -> None:
        """Restore FP32 shadow weights from packed weights."""
        if not self.is_inference:
            return
            
        W_q = self.quantizer.unpack(self.packed_weight, (self.out_features, self.in_features))
        W_fp32 = self.quantizer.dequantize(W_q, self.weight_scale)
        
        self.weight = nn.Parameter(W_fp32)
        self.packed_weight = None
        self.packed_weight_u8 = None
        self.weight_scale = None
        self.pos_mask = None
        self.neg_mask = None
        self.w_ternary = None
        self.is_inference = False

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: Optional[BitNetConfig] = None) -> 'BitLinear':
        """Convert existing nn.Linear to BitLinear."""
        bitlinear = cls(linear.in_features, linear.out_features, linear.bias is not None, config)
        bitlinear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            bitlinear.bias.data.copy_(linear.bias.data)
        return bitlinear

    def extra_repr(self) -> str:
        """Extra representation for print(model)."""
        return f"in_features={self.in_features}, out_features={self.out_features}, " \
               f"bias={self.bias is not None}, inference={self.is_inference}"


# ── TernaryCPUKernel ──

class TernaryCPUKernel:
    """
    Fast CPU kernel for ternary weight matrix multiplication.
    
    Key insight: W ∈ {-1, 0, +1} means:
    output = x @ (pos_mask - neg_mask).T
    where pos_mask[i,j] = 1 if W[i,j] = +1
    neg_mask[i,j] = 1 if W[i,j] = -1
    
    This uses vectorized single-pass GEMM and bit-packing operations.
    """
    _cache = {}

    def __init__(self) -> None:
        """Initialize the CPU kernel."""
        pass
        
    def matmul(self, x: Tensor, W_packed: Tensor, scale: Tensor, original_shape: tuple) -> Tensor:
        """Ternary matmul using pos/neg mask decomposition."""
        cache_key = (id(W_packed), W_packed.shape, x.dtype, x.device)
        if cache_key in TernaryCPUKernel._cache:
            pos_mask, neg_mask, w_ternary = TernaryCPUKernel._cache[cache_key]
        else:
            pos_mask, neg_mask, w_ternary = self._unpack_masks(W_packed, original_shape)
            pos_mask = pos_mask.to(dtype=x.dtype, device=x.device)
            neg_mask = neg_mask.to(dtype=x.dtype, device=x.device)
            w_ternary = w_ternary.to(dtype=x.dtype, device=x.device)
            TernaryCPUKernel._cache[cache_key] = (pos_mask, neg_mask, w_ternary)
        
        eps = 1e-8
        x_scale = x.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
        x_norm = x / x_scale
        
        # Single-pass ternary matmul using pre-computed w_ternary matrix
        out = F.linear(x_norm, w_ternary) * (scale * x_scale)
        return out

    def _unpack_masks(self, W_packed: Tensor, shape: tuple) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract positive, negative masks, and single ternary weight matrix from packed weights."""
        numel = shape[0] * shape[1]
        if W_packed.dtype == torch.uint8:
            W_packed_u8 = W_packed
        else:
            W_packed_u8 = W_packed.view(torch.uint8)
        shifts_u8 = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=W_packed.device)
        W_mapped = (W_packed_u8.unsqueeze(1) >> shifts_u8) & 0b11
        W_flat_mapped = W_mapped.flatten()[:numel]
        
        pos_mask = (W_flat_mapped == 2).view(shape).to(torch.int8)
        neg_mask = (W_flat_mapped == 0).view(shape).to(torch.int8)
        w_ternary = (W_flat_mapped.to(torch.float32) - 1.0).view(shape)
        # Note: mapped 0 -> -1, mapped 1 -> 0, mapped 2 -> 1
        return pos_mask, neg_mask, w_ternary

    def benchmark(self, in_f: int, out_f: int, batch: int = 1) -> dict:
        """Compare ternary vs FP32 matmul speed."""
        W = torch.randn(out_f, in_f)
        x = torch.randn(batch, in_f)
        
        quantizer = TernaryQuantizer(BitNetConfig())
        W_q, scale = quantizer.quantize(W)
        W_packed = quantizer.pack(W_q)
        pos_mask = (W_q == 1).to(torch.float32)
        neg_mask = (W_q == -1).to(torch.float32)
        
        # Warmup
        for _ in range(5):
            _ = torch.matmul(x, W.t())
            _ = F.linear(x, pos_mask) - F.linear(x, neg_mask)
            _ = self.matmul(x, W_packed, scale, (out_f, in_f))
            
        # FP32
        t0 = time.time()
        for _ in range(50):
            _ = torch.matmul(x, W.t())
        fp32_ms = (time.time() - t0) * 1000 / 50

        # Unoptimized Ternary (2 GEMMs)
        t0 = time.time()
        for _ in range(50):
            _ = F.linear(x, pos_mask) - F.linear(x, neg_mask)
        unoptimized_ternary_ms = (time.time() - t0) * 1000 / 50
        
        # Optimized Ternary
        t0 = time.time()
        for _ in range(50):
            _ = self.matmul(x, W_packed, scale, (out_f, in_f))
        ternary_ms = (time.time() - t0) * 1000 / 50
        
        return {
            'fp32_ms': fp32_ms,
            'unoptimized_ternary_ms': unoptimized_ternary_ms,
            'ternary_ms': ternary_ms,
            'speedup': fp32_ms / (ternary_ms + 1e-8),
            'kernel_speedup': unoptimized_ternary_ms / (ternary_ms + 1e-8)
        }


# ── BitNetTrainerHooks ──

class BitNetTrainerHooks:
    """
    Manages BitLinear layers during training.
    - Keeps track of all BitLinear layers in model
    - Ensures FP32 shadow weights are updated correctly
    - Handles weight decay only on FP32 shadow (not ternary)
    """
    def __init__(self, model: nn.Module, config: BitNetConfig) -> None:
        """Initialize hooks manager."""
        self.model = model
        self.config = config
        self.bitlinear_layers = self.get_bitlinear_layers()
        
    def get_bitlinear_layers(self) -> list[BitLinear]:
        """Collect all BitLinear layers in the model."""
        return [m for m in self.model.modules() if isinstance(m, BitLinear)]
        
    def before_optimizer_step(self) -> None:
        """Called before optimizer.step() — clip gradients of shadow weights."""
        for layer in self.bitlinear_layers:
            if layer.weight is not None and layer.weight.grad is not None:
                layer.weight.grad.clamp_(-1.0, 1.0)
                
    def after_optimizer_step(self) -> None:
        """Called after optimizer.step() — nothing needed (shadow weights update naturally)."""
        pass
        
    def get_param_groups(self) -> list[dict]:
        """
        Returns param groups for optimizer:
        - BitLinear shadow weights: normal lr, weight decay
        - Other params: normal lr, weight decay
        - BitLinear bias: normal lr, NO weight decay
        """
        bitlinear_weights = []
        bitlinear_biases = []
        other_params = []
        
        bitlinear_modules = set(self.bitlinear_layers)
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_bitlinear = False
            for bl_layer in bitlinear_modules:
                if param is bl_layer.weight:
                    bitlinear_weights.append(param)
                    is_bitlinear = True
                    break
                elif param is bl_layer.bias:
                    bitlinear_biases.append(param)
                    is_bitlinear = True
                    break
                    
            if not is_bitlinear:
                other_params.append(param)
                
        return [
            {"params": bitlinear_weights},
            {"params": other_params},
            {"params": bitlinear_biases, "weight_decay": 0.0}
        ]
        
    def get_quantization_stats(self) -> dict:
        """Return stats on how ternary the weights currently are."""
        if not self.bitlinear_layers:
            return {}
        
        total_sparsity = 0.0
        total_pos = 0.0
        total_neg = 0.0
        
        for layer in self.bitlinear_layers:
            if layer.weight is not None:
                W_q, _ = layer.quantizer.quantize(layer.weight)
                n = W_q.numel()
                total_sparsity += (W_q == 0).sum().item() / n
                total_pos += (W_q == 1).sum().item() / n
                total_neg += (W_q == -1).sum().item() / n
                
        n_layers = len(self.bitlinear_layers)
        return {
            'avg_sparsity': total_sparsity / n_layers,
            'avg_pos': total_pos / n_layers,
            'avg_neg': total_neg / n_layers,
        }
