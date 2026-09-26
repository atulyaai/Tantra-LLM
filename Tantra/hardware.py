"""
Tantra/hardware.py — Detect the machine and pick the compute device.

Everything the CLI and WebUI need: CPU/RAM/GPU summary, CPU thread setup,
and a single `detect_hardware()` call that returns a HardwareProfile.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from Tantra.utils import get_logger

try:
    import psutil
except ImportError:  # optional
    psutil = None

log = get_logger("tantra")


@dataclass
class HardwareProfile:
    cpu_name: str
    physical_cores: int
    logical_cores: int
    ram_gb: float
    simd: str
    gpus: List[str] = field(default_factory=list)
    device: str = "cpu"
    threads: int = 1

    def as_dict(self) -> dict:
        return {
            "cpu": self.cpu_name, "cpu_threads": self.threads,
            "physical_cores": self.physical_cores, "logical_cores": self.logical_cores,
            "ram_gb": self.ram_gb, "simd": self.simd, "gpus": self.gpus, "device": self.device,
        }


def configure_cpu_performance(num_threads: Optional[int] = None) -> int:
    """Use all physical cores for PyTorch/OpenMP. Returns the thread count."""
    if not num_threads or num_threads <= 0:
        num_threads = (psutil.cpu_count(logical=False) if psutil else None) or os.cpu_count() or 4
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(num_threads)
    try:
        torch.set_num_threads(num_threads)
    except RuntimeError:
        pass
    return num_threads


def pick_device(requested: str = "auto") -> str:
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but not available — using CPU.")
        return "cpu"
    return "cuda:0" if requested == "cuda" else requested


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if platform.system() == "Windows":   # platform.processor() only gives "AMD64 Family 23 ..."
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def detect_hardware(requested_device: str = "auto", num_threads: Optional[int] = None,
                    verbose: bool = True) -> HardwareProfile:
    threads = configure_cpu_performance(num_threads)
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append(f"{p.name} ({p.total_memory // 2**20} MB)")
    try:
        simd = torch.backends.cpu.get_cpu_capability()
    except Exception:
        simd = "unknown"
    hw = HardwareProfile(
        cpu_name=_cpu_name(),
        physical_cores=(psutil.cpu_count(logical=False) if psutil else None) or os.cpu_count() or 1,
        logical_cores=os.cpu_count() or 1,
        ram_gb=round(psutil.virtual_memory().total / 2**30, 1) if psutil else 0.0,
        simd=simd,
        gpus=gpus,
        device=pick_device(requested_device),
        threads=threads,
    )
    if verbose:
        log.info(f"Hardware: {hw.cpu_name} | {hw.physical_cores} cores | {hw.ram_gb} GB RAM | "
                 f"SIMD {hw.simd} | GPUs: {', '.join(gpus) or 'none'} | device -> {hw.device}")
    return hw
