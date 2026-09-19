"""Tantra/cli_hardware_dispatch.py — CLI-level hardware startup dispatch.

Extracted from main.py (previously detect_hardware()). This is orchestration
on top of the reusable primitives in Tantra/hardware.py (HardwareDetector,
Profiler, RuntimeConfigBuilder, AdaptiveScheduler) — it decides whether to
run the full benchmarking path or a fast no-benchmark path for
containerized / non-interactive environments (Colab, CI, piped stdout).

Deliberately kept separate from Tantra/hardware.py rather than merged into
it: Tantra/hardware.py holds generic, reusable hardware-detection primitives
with no CLI/environment-specific branching. This module is the CLI's
decision of *how* to call those primitives, not a primitive itself.
"""
from __future__ import annotations

import os
import sys

from Tantra.utils import get_logger
from Tantra.hardware import (
    HardwareDetector,
    Profiler,
    RuntimeConfigBuilder,
    AdaptiveScheduler,
    GPUInfo,
    RuntimeConfig,
)

log = get_logger("tantra")


class NullScheduler:
    """No-op scheduler used for the fast (Colab/non-TTY/CI) startup path.

    Named NullScheduler rather than the original inline `FastScheduler` to
    avoid confusion with Tantra.hardware's own scheduling concepts — this
    class does nothing by design; it exists only so callers can call
    `.start()` / `.stop()` uniformly regardless of which path was taken.
    """

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def detect_hardware():
    """Detect hardware and build a RuntimeConfig + scheduler.

    Returns (RuntimeConfig, scheduler) where scheduler has .start()/.stop().
    Takes a fast, no-benchmark path in containerized or non-interactive
    environments (Google Colab, CI, piped/non-TTY stdout) so startup isn't
    blocked on hardware benchmarking; otherwise runs the full
    HardwareDetector -> Profiler -> RuntimeConfigBuilder pipeline.
    """
    log.info("== [1] HARDWARE AUTO-DETECTION & PROACTIVE HEALTH ==")

    is_colab = (
        'google.colab' in sys.modules
        or os.environ.get('COLAB_RELEASE_TAG') is not None
        or os.environ.get('COLAB_GPU') is not None
        or os.path.exists('/content')
    )
    is_non_tty = not getattr(sys.stdout, "isatty", lambda: False)()

    if is_colab or is_non_tty:
        log.info("  [INFO] Running in Container/Non-TTY mode. Skipping benchmarks for instant startup.")

        has_cuda = False
        gpus = []
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                gpus = [GPUInfo(
                    0,
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
                    "8.0",
                    "cuda",
                )]
        except Exception:
            pass

        device = 'cuda:0' if gpus else 'cpu'
        log.info(f"  Detected Device: {device} | strategy: {'full_gpu' if gpus else 'cpu_only'}")

        rt = RuntimeConfig(
            device=device,
            dtype='bfloat16' if gpus else 'int8',
            use_bitnet=True,
            batch_size=4 if gpus else 1,
            max_seq_len=8192,
            active_experts=1,
            expert_cache_size=8,
            prefetch_depth=2,
            compression_level='high',
            offload_strategy='full_gpu' if gpus else 'cpu_only',
            ram_budget_mb=8192,
            vram_budget_mb=12000,
            expert_size_mb=500,
            num_threads=4,
            prefill_chunk_size=512,
            profile_name="COLAB-GPU" if gpus else "COLAB-CPU"
        )
        return rt, NullScheduler()

    hw = HardwareDetector()
    profile = hw.detect()
    hw.print_profile(profile)
    perf = Profiler(profile).run()
    rt = RuntimeConfigBuilder().build(profile, perf)
    log.info(f"  Strategy   : {rt.offload_strategy}")
    log.info(f"  Device     : {rt.device} | dtype: {rt.dtype}")
    log.info(f"  Compression: {rt.compression_level}")
    log.info(f"  Expert Cache: {rt.expert_cache_size} in RAM | batch: {rt.batch_size}")

    sched = AdaptiveScheduler(rt)
    sched.start()
    return rt, sched
