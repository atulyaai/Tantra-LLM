"""
tantra/utils.py — Shared utilities. Import from here, never duplicate.
"""
from __future__ import annotations
import os
import sys
import time
import logging
import hashlib
import struct
from typing import Any, Iterator, Optional
from contextlib import contextmanager

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import numpy as np

# ── Logging with Rich Fallback ────────────────────────────────────────────────

is_colab = os.path.exists('/content') or os.environ.get('COLAB_GPU') is not None or os.environ.get('COLAB_RELEASE_TAG') is not None

try:
    from rich.console import Console
    from rich.logging import RichHandler
    is_tty = getattr(sys.stdout, "isatty", lambda: False)()
    if is_colab or not is_tty:
        _HAS_RICH = False
        _console = None
    else:
        _console = Console(force_terminal=True, legacy_windows=False)
        _HAS_RICH = True
except ImportError:
    _console = None
    _HAS_RICH = False

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def get_logger(name: str) -> logging.Logger:
    """Get a logger (uses rich if available, standard unbuffered logging in Colab/containers)."""
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        if _HAS_RICH:
            handler = RichHandler(console=_console, rich_tracebacks=True)
            handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        else:
            handler = FlushStreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        # Suppress non-zero DDP ranks from flooding the console with duplicate INFO logs
        global_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        if global_rank != 0:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
    return logger


# ── Tensor Utilities ──────────────────────────────────────────────────────────

def count_parameters(module: "torch.nn.Module") -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_parameters_all(module: "torch.nn.Module") -> dict[str, int]:
    """Count params broken down by submodule name."""
    return {name: sum(p.numel() for p in m.parameters())
            for name, m in module.named_modules() if list(m.parameters(recurse=False))}


def tensor_memory_mb(t: "torch.Tensor") -> float:
    """Return tensor memory usage in megabytes."""
    return t.numel() * t.element_size() / 1024 / 1024


def human_params(n: int) -> str:
    """Format parameter count as 1.2M, 3.4B, etc."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_bytes(t: "torch.Tensor") -> bytes:
    """Convert tensor data to raw contiguous bytes."""
    t_cpu = t.detach().cpu().contiguous()
    return bytes(t_cpu.numpy().data)


def bytes_to_tensor(data: bytes, dtype: torch.dtype = torch.float32) -> "torch.Tensor":
    """Convert raw bytes back to a 1D tensor."""
    np_dtype = torch.zeros(1, dtype=dtype).numpy().dtype
    arr = np.frombuffer(data, dtype=np_dtype)
    return torch.from_numpy(arr.copy())


def elu_plus_one(x: "torch.Tensor") -> "torch.Tensor":
    """ELU(x) + 1 kernel activation (maps all values to positive reals)."""
    import torch.nn.functional as F
    return F.elu(x) + 1.0


def top_k_mask(gates: "torch.Tensor", k: int) -> "torch.Tensor":
    """
    Given gating weights of shape (..., hidden), returns a binary mask 
    of the same shape where only the top-k values are 1, rest 0.
    """
    _, indices = torch.topk(gates, k=k, dim=-1, sorted=False)
    mask = torch.zeros_like(gates, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask


@contextmanager
def timer(name: str = "Operation"):
    """Context manager to log elapsed execution time."""
    log = get_logger("timer")
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    log.info(f"{name} took {elapsed:.2f} ms")


def unwrap_model(model: Any) -> Any:
    """Fully unwrap nested model wrappers (DataParallel, DDP, torch.compile _orig_mod)."""
    raw = model
    while hasattr(raw, "module"):
        raw = raw.module
    while hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


def safe_load_checkpoint(path: str, map_location: Any = "cpu") -> Any:
    """
    Safely load a PyTorch checkpoint using weights_only=True.
    Registers all NeuroCore configuration dataclasses as safe globals to prevent
    arbitrary code execution (RCE via malicious pickles) while maintaining full
    support for custom checkpoint metadata.
    """
    if hasattr(torch.serialization, "add_safe_globals"):
        try:
            from Tantra.config import (
                NeuroCoreConfig, VocabConfig, ALRAConfig, SGPConfig,
                NeuroCoreBlockConfig, MoEConfig, AdapterConfig,
                BitNetConfig, CompressionConfig, InferenceConfig, TrainingConfig
            )
            torch.serialization.add_safe_globals([
                NeuroCoreConfig, VocabConfig, ALRAConfig, SGPConfig,
                NeuroCoreBlockConfig, MoEConfig, AdapterConfig,
                BitNetConfig, CompressionConfig, InferenceConfig, TrainingConfig
            ])
        except Exception:
            pass

    return torch.load(path, map_location=map_location, weights_only=True)


# ── Shared 2-Bit Bit-Packing Utility ──────────────────────────────────

_PACK_SHIFTS = np.array([6, 4, 2, 0], dtype=np.uint8)
_PACK_MASK = np.uint8(0b11)


def pack_2bit_to_4x(values: np.ndarray) -> np.ndarray:
    """Pack 4 2-bit values (0-3) into 1 byte using MSB-first layout.
    
    Values are packed as: v[0]<<6 | v[1]<<4 | v[2]<<2 | v[3]
    Both codec.py and bitnet.py used independent implementations of this.
    """
    values = values.reshape(-1, 4)
    packed = np.zeros((values.shape[0],), dtype=np.uint8)
    for i, shift in enumerate(_PACK_SHIFTS):
        packed |= (values[:, i] & _PACK_MASK) << shift
    return packed


def unpack_2bit_from_4x(packed: np.ndarray, numel: int) -> np.ndarray:
    """Unpack 1 byte into 4 2-bit values (0-3) using MSB-first layout.
    
    Inverse of pack_2bit_to_4x. Returns exactly numel values.
    """
    arr = packed.reshape(-1).copy()
    out = np.empty(arr.shape[0] * 4, dtype=np.uint8)
    out[0::4] = (arr >> 6) & _PACK_MASK
    out[1::4] = (arr >> 4) & _PACK_MASK
    out[2::4] = (arr >> 2) & _PACK_MASK
    out[3::4] = arr & _PACK_MASK
    return out[:numel]


def pack_4_ternary_to_1_byte(ternary_vals: "torch.Tensor") -> "torch.Tensor":
    """Pack 4 ternary values {-1,0,+1} into 1 uint8 byte (PyTorch version).
    
    Maps ternary to {0,1,2,3} via +1 offset, then packs with [6,4,2,0] shifts.
    bitnet.py's TernaryQuantizer.pack_uint8 used this independently — now shared.
    """
    W_mapped = (ternary_vals.flatten() + 1).to(torch.uint8)
    # Pad to multiple of 4
    pad_len = (4 - (W_mapped.numel() % 4)) % 4
    if pad_len > 0:
        W_mapped = torch.cat([W_mapped, torch.zeros(pad_len, dtype=torch.uint8, device=W_mapped.device)])
    W_mapped = W_mapped.view(-1, 4)
    shifts = torch.tensor([6, 4, 2, 0], dtype=torch.uint8, device=W_mapped.device)
    packed = torch.zeros(W_mapped.shape[0], dtype=torch.uint8, device=W_mapped.device)
    for i in range(4):
        packed |= (W_mapped[:, i] << shifts[i])
    return packed


def unpack_1_byte_to_4_ternary(packed: "torch.Tensor", original_shape: tuple) -> "torch.Tensor":
    """Unpack uint8 bytes back to int8 ternary values {-1,0,+1} (PyTorch version).
    
    Inverse of pack_4_ternary_to_1_byte. bitnet.py's TernaryQuantizer.unpack_uint8 used this independently.
    """
    shifts = torch.tensor([6, 4, 2, 0], dtype=torch.uint8, device=packed.device)
    W_mapped = (packed.unsqueeze(1) >> shifts) & 0b11
    W_flat = (W_mapped - 1).to(torch.int8).flatten()
    numel = int(torch.prod(torch.tensor(original_shape))) if isinstance(original_shape, tuple) else original_shape
    return W_flat[:numel].view(original_shape)

