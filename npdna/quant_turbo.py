"""CPU inference & training helpers for NP-DNA models.

The helpers here are deliberately opt-in. Training should keep full precision;
inference can call these after loading a checkpoint to reduce memory and speed
up CPU linear layers.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from torch import nn

_THREAD_POOL: ThreadPoolExecutor | None = None


def get_thread_pool() -> ThreadPoolExecutor:
    """Return a shared CPU worker pool for parallel inference helpers."""
    global _THREAD_POOL
    if _THREAD_POOL is None:
        workers = max(1, int(os.environ.get("TANTRA_CPU_THREADS", os.cpu_count() or 4)))
        _THREAD_POOL = ThreadPoolExecutor(max_workers=workers)
    return _THREAD_POOL


def enable_torch_cpu_optimizations(num_threads: int | None = None) -> None:
    """Enable conservative PyTorch CPU optimizations."""
    threads = max(1, int(num_threads or os.environ.get("TANTRA_CPU_THREADS", os.cpu_count() or 4)))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, threads)))
    except RuntimeError:
        pass
    try:
        torch.backends.mkldnn.enabled = True
    except Exception:
        pass


def quantize_model_for_cpu(core: Any, inplace: bool = True) -> Any:
    """Apply dynamic INT8 quantization to CPU-friendly Linear layers.

    Args:
        core: ``NpDnaCore``-like object with a ``model`` attribute.
        inplace: when False, returns a quantized copy through PyTorch.

    Returns:
        The same core object, with ``core.model`` replaced by the quantized model.
    """
    if not hasattr(core, "model"):
        raise TypeError("quantize_model_for_cpu expects an object with a model attribute")

    core.model.eval()
    core.model.cpu()
    core.model = torch.ao.quantization.quantize_dynamic(
        core.model,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=inplace,
    )
    return core


def apply_torch_compile(core: Any, mode: str = "reduce-overhead") -> Any:
    """Optionally compile the model for repeated inference calls."""
    if not hasattr(torch, "compile"):
        return core
    try:
        core.model = torch.compile(core.model, mode=mode, backend="inductor")
    except Exception:
        pass
    return core


def model_size_mb(model: nn.Module) -> float:
    """Approximate parameter storage size in MiB."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)


def enable_gradient_checkpointing(core: Any) -> None:
    """Enable gradient checkpointing for memory-efficient training.

    Trades compute for memory: activations are recomputed during backward
    instead of stored. Reduces memory by ~30-50% on CPU with ~20% speed cost.
    """
    if hasattr(core.model, "gradient_checkpointing"):
        core.model.gradient_checkpointing = True


def freeze_for_partial_training(core: Any, train_strands: bool = True,
                                train_embeddings: bool = False) -> int:
    """Freeze all params except genome seeds (and optionally embeddings).

    Enables 'partial training' / fine-tuning: only the strand-generating
    DNA seeds are updated, keeping all router/norm/embedding weights fixed.

    Args:
        core: NpDnaCore instance.
        train_strands: If True, keep genome seeds trainable.
        train_embeddings: If True, also keep embeddings trainable.

    Returns:
        Number of trainable parameters.
    """
    for name, param in core.model.named_parameters():
        if 'genome.seeds' in name:
            param.requires_grad = True
        elif 'embedding' in name and train_embeddings:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return sum(p.numel() for p in core.model.parameters() if p.requires_grad)


def count_active_parameters(model: nn.Module) -> dict[str, int]:
    """Count parameters by group for debugging memory usage."""
    counts = {}
    for name, param in model.named_parameters():
        group = name.split(".")[0] if "." in name else name
        counts[group] = counts.get(group, 0) + param.numel()
    return counts
