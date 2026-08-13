"""Tests for tantra.hardware"""
import pytest
from Tantra.hardware import HardwareDetector, Profiler, RuntimeConfigBuilder


def test_detector_runs():
    hw = HardwareDetector()
    profile = hw.detect()
    assert profile.ram_total_mb > 0
    assert profile.cpu.physical_cores > 0

def test_profiler_runs():
    hw = HardwareDetector()
    profile = hw.detect()
    perf = Profiler(profile).run()
    assert perf.fp32_matmul_gflops > 0

def test_runtime_config():
    hw = HardwareDetector()
    profile = hw.detect()
    perf = Profiler(profile).run()
    rt = RuntimeConfigBuilder().build(profile, perf)
    assert rt.device in ("cpu", "cuda:0", "mps")
    assert rt.batch_size >= 1
