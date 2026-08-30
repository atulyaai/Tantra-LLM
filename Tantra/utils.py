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
