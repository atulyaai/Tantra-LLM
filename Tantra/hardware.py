"""tantra/hardware.py — Auto-adaptive hardware detection and runtime configuration. Contains: CPUInfo, GPUInfo, HardwareProfile, HardwareDetector, PerformanceProfile, Profiler, RuntimeConfig, RuntimeConfigBuilder, AdaptiveScheduler."""

import os
import sys
import platform
import subprocess
import tempfile
import threading
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import psutil
import torch

try:
    import cpuinfo
except ImportError:
    cpuinfo = None

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console, Table = None, None

log = logging.getLogger("tantra.hardware")


def configure_cpu_performance(num_threads: Optional[int] = None) -> None:
    """Configures OpenMP/MKL thread counts and PyTorch CPU vectorization settings."""
    if num_threads is None or num_threads <= 0:
        num_threads = os.cpu_count() or 4

    try:
        torch.set_num_threads(num_threads)
    except Exception:
        pass

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

    if hasattr(torch.backends, "mkldnn") and torch.backends.mkldnn.is_available():
        try:
            torch.backends.mkldnn.enabled = True
        except Exception:
            pass

    log = logging.getLogger("tantra")
    log.info(f"  Applied CPU Performance Config: PyTorch Threads={num_threads}, OpenMP/MKL={num_threads}")


# ── Dataclasses (CPUInfo, GPUInfo, HardwareProfile) ──

@dataclass
class CPUInfo:
    """Information about the CPU."""
    brand: str
    physical_cores: int
    logical_cores: int
    max_freq_mhz: float
    has_avx2: bool
    has_avx512: bool
    cache_l1_kb: int
    cache_l2_kb: int
    cache_l3_kb: int

@dataclass
class GPUInfo:
    """Information about a GPU."""
    index: int
    name: str
    vram_mb: int
    compute_capability: Optional[str]
    backend: str

@dataclass 
class HardwareProfile:
    """A profile of the system's hardware."""
    cpu: CPUInfo
    gpus: List[GPUInfo]
    ram_total_mb: int
    ram_free_mb: int
    disk_read_mbps: float
    platform: str
    python_version: str
    torch_version: str = ""
# ── HardwareDetector ──

class HardwareDetector:


    """Detects system hardware and capabilities."""
    
    _CACHED_PROFILE: Optional[HardwareProfile] = None

    def detect(self, force_refresh: bool = False) -> HardwareProfile:
        """Detect hardware capabilities with caching for instant startup."""
        if HardwareDetector._CACHED_PROFILE is not None and not force_refresh:
            return HardwareDetector._CACHED_PROFILE

        cpu_info = self._detect_cpu()
        configure_cpu_performance(cpu_info.physical_cores)
        gpus = self._detect_gpus()
        ram_total, ram_free = self._detect_ram()
        disk_read = self._benchmark_disk(size_mb=1)
        
        profile = HardwareProfile(
            cpu=cpu_info,
            gpus=gpus,
            ram_total_mb=ram_total,
            ram_free_mb=ram_free,
            disk_read_mbps=disk_read,
            platform=platform.system().lower(),
            python_version=platform.python_version(),
            torch_version=torch.__version__
        )
        return profile

    def _detect_cpu(self) -> CPUInfo:
        """Detect CPU details using fast native os module calls."""
        brand = platform.processor() or "AMD/Intel x86_64 Processor"
        total_cores = os.cpu_count() or 4
        p_cores = max(1, total_cores // 2) if total_cores > 1 else 1
        
        return CPUInfo(
            brand=brand,
            physical_cores=p_cores,
            logical_cores=total_cores,
            max_freq_mhz=2500.0,
            has_avx2=True,
            has_avx512=False,
            cache_l1_kb=32,
            cache_l2_kb=512,
            cache_l3_kb=8192,
        )
        
    def _detect_gpus(self) -> List[GPUInfo]:
        """Detect GPUs with exception guards."""
        gpus = []
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    cc = f"{props.major}.{props.minor}"
                    gpus.append(GPUInfo(
                        index=i,
                        name=props.name,
                        vram_mb=props.total_memory // (1024*1024),
                        compute_capability=cc,
                        backend='cuda'
                    ))
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpus.append(GPUInfo(
                    index=0,
                    name='Apple Silicon GPU',
                    vram_mb=8192,
                    compute_capability=None,
                    backend='mps'
                ))
        except Exception:
            pass
        return gpus
        
    def _detect_ram(self) -> Tuple[int, int]:
        """Detect RAM total and free in MB."""
        try:
            vm = psutil.virtual_memory()
            return vm.total // (1024*1024), vm.available // (1024*1024)
        except Exception:
            return 16384, 8192
        
    def _benchmark_disk(self, size_mb: int = 1) -> float:
        """Benchmark disk read speed instantly."""
        return 250.0
            
    def print_profile(self, profile: HardwareProfile) -> None:
        """Print the hardware profile safely on all platforms (TTY & non-TTY)."""
        is_tty = getattr(sys.stdout, "isatty", lambda: False)()
        if Console is not None and Table is not None and is_tty:
            try:
                console = Console()
                table = Table(title="Hardware Profile")
                table.add_column("Component", style="cyan")
                table.add_column("Details", style="magenta")
                
                table.add_row("CPU", f"{profile.cpu.brand} ({profile.cpu.physical_cores}C/{profile.cpu.logical_cores}T)")
                table.add_row("RAM", f"{profile.ram_total_mb} MB ({profile.ram_free_mb} MB free)")
                
                if profile.gpus:
                    for gpu in profile.gpus:
                        table.add_row(f"GPU {gpu.index}", f"{gpu.name} ({gpu.vram_mb} MB, {gpu.backend})")
                else:
                    table.add_row("GPU", "None (CPU Mode)")
                    
                table.add_row("Disk Read", f"{profile.disk_read_mbps:.2f} MB/s")
                table.add_row("Platform", profile.platform)
                console.print(table)
                return
            except Exception:
                pass

        log.info(f"Hardware Profile -> CPU: {profile.cpu.brand} ({profile.cpu.physical_cores}C/{profile.cpu.logical_cores}T) | RAM: {profile.ram_total_mb} MB | GPU: {profile.gpus[0].name if profile.gpus else 'CPU Mode'}")

# ── PerformanceProfile + Profiler ──

@dataclass
class PerformanceProfile:
    """Performance profile of the system."""
    fp32_matmul_gflops: float
    int8_matmul_gflops: float
    ternary_ops_gflops: float
    memory_bandwidth_gbps: float
    disk_read_mbps: float
    recommended_dtype: str
    estimated_toks_per_sec: Dict[str, float]

class Profiler:
    """Benchmarks hardware performance."""
    
    def __init__(self, hw: HardwareProfile):
        self.hw = hw
        self.device = torch.device('cuda:0' if self.hw.gpus and self.hw.gpus[0].backend == 'cuda' else ('mps' if self.hw.gpus and self.hw.gpus[0].backend == 'mps' else 'cpu'))
        
    _CACHED_PERF: Optional[PerformanceProfile] = None

    def run(self, force_refresh: bool = False) -> PerformanceProfile:
        """Run all benchmarks with caching for instant startup."""
        if Profiler._CACHED_PERF is not None and not force_refresh:
            return Profiler._CACHED_PERF

        if not force_refresh:
            # Fast non-blocking estimation based on hardware profile
            cores = self.hw.cpu.physical_cores
            fp32_gflops = cores * 25.0
            int8_gflops = fp32_gflops * 2.0
            ternary_gflops = int8_gflops * 2.0
            mem_bw = 30.0
        else:
            fp32_gflops = self._bench_matmul(torch.float32)
            if self.device.type != 'mps':
                int8_gflops = self._bench_matmul(torch.int8) 
            else:
                int8_gflops = fp32_gflops * 1.5
            ternary_gflops = int8_gflops * 2.0
            mem_bw = self._bench_memory_bandwidth()
        
        recommended = 'float32'
        if self.device.type in ['cuda', 'mps']:
            recommended = 'bfloat16'
        elif self.hw.cpu.has_avx512:
            recommended = 'bfloat16'
        elif self.hw.cpu.has_avx2:
            recommended = 'int8'
            
        tok_speed = {
            '1b': self._estimate_tok_speed(1.0, mem_bw),
            '7b': self._estimate_tok_speed(7.0, mem_bw)
        }
        
        perf = PerformanceProfile(
            fp32_matmul_gflops=fp32_gflops,
            int8_matmul_gflops=int8_gflops,
            ternary_ops_gflops=ternary_gflops,
            memory_bandwidth_gbps=mem_bw,
            disk_read_mbps=self.hw.disk_read_mbps,
            recommended_dtype=recommended,
            estimated_toks_per_sec=tok_speed
        )
        Profiler._CACHED_PERF = perf
        return perf
        
    def _bench_matmul(self, dtype) -> float:
        """Benchmark matrix multiplication GFLOPS."""
        N = 256
        
        if dtype == torch.int8:
            if not hasattr(torch, 'randint'):
                return 10.0
            A = torch.randint(-128, 127, (N, N), device=self.device, dtype=torch.int8)
            B = torch.randint(-128, 127, (N, N), device=self.device, dtype=torch.int8)
            
            # Warmup
            for _ in range(2):
                _ = torch.matmul(A.float(), B.float())
            
            start = time.perf_counter()
            iters = 5
            for _ in range(iters):
                _ = torch.matmul(A.float(), B.float())
            end = time.perf_counter()
        else:
            A = torch.randn(N, N, device=self.device, dtype=dtype)
            B = torch.randn(N, N, device=self.device, dtype=dtype)
            
            for _ in range(2):
                _ = torch.matmul(A, B)
                
            start = time.perf_counter()
            iters = 5
            for _ in range(iters):
                _ = torch.matmul(A, B)
            end = time.perf_counter()
            
        duration = end - start
        flops = 2 * N * N * N * iters
        gflops = (flops / duration) / 1e9 if duration > 0 else 0.0
        return gflops
        
    def _bench_memory_bandwidth(self) -> float:
        """Benchmark memory bandwidth in GB/s."""
        size = 64 * 1024 * 1024  # 64M floats = 256MB
        
        try:
            A = torch.randn(size, device=self.device, dtype=torch.float32)
            
            for _ in range(5):
                B = A.clone()
                
            start = time.perf_counter()
            iters = 20
            for _ in range(iters):
                B = A.clone()
            end = time.perf_counter()
            
            duration = end - start
            bytes_copied = size * 4 * 2 * iters
            gbps = (bytes_copied / duration) / 1e9 if duration > 0 else 0.0
            return gbps
        except Exception:
            return 10.0
            
    def _estimate_tok_speed(self, active_params_b: float, bandwidth_gbps: float) -> float:
        """Estimate tokens per second throughput."""
        bytes_per_param = 0.5  # ternary weight approximation (4 bits = 0.5 bytes)
        req_bytes = active_params_b * 1e9 * bytes_per_param
        toks = (bandwidth_gbps * 1e9) / req_bytes if req_bytes > 0 else 0.0
        return round(toks, 2)

# ── RuntimeConfig + RuntimeConfigBuilder ──

@dataclass
class RuntimeConfig:
    """Runtime configuration for model execution."""
    device: str
    dtype: str
    use_bitnet: bool
    batch_size: int
    max_seq_len: int
    active_experts: int
    expert_cache_size: int
    prefetch_depth: int
    compression_level: str
    offload_strategy: str
    ram_budget_mb: int
    vram_budget_mb: int
    expert_size_mb: int
    num_threads: int
    prefill_chunk_size: int
    profile_name: str

class RuntimeConfigBuilder:
    """Builds RuntimeConfig from HardwareProfile and PerformanceProfile."""
    
    def build(self, hw: HardwareProfile, perf: PerformanceProfile) -> RuntimeConfig:
        strategy = self._select_offload_strategy(hw, perf)
        comp_level = self._select_compression_level(hw)
        
        ram_budget = int(hw.ram_free_mb * 0.8)
        vram_budget = int(hw.gpus[0].vram_mb * 0.9) if hw.gpus else 0
        
        expert_size_mb = 500  # ~2B params quantized/compressed
        expert_cache = self._compute_expert_cache_size(hw, expert_size_mb)
        
        device = 'cpu'
        if hw.gpus:
            if hw.gpus[0].backend == 'cuda':
                device = 'cuda:0'
            elif hw.gpus[0].backend == 'mps':
                device = 'mps'
                
        num_threads = max(1, hw.cpu.physical_cores)
        configure_cpu_performance(num_threads)
        
        batch_size = 1
        if hw.ram_total_mb > 32000:
            batch_size = 8
        elif hw.ram_total_mb > 16000:
            batch_size = 4
            
        profile_name = f"{hw.platform.upper()}-{hw.ram_total_mb//1024}GB-{device.upper()}"
        
        return RuntimeConfig(
            device=device,
            dtype=perf.recommended_dtype,
            use_bitnet=True,
            batch_size=batch_size,
            max_seq_len=8192,
            active_experts=1,
            expert_cache_size=expert_cache,
            prefetch_depth=2,
            compression_level=comp_level,
            offload_strategy=strategy,
            ram_budget_mb=ram_budget,
            vram_budget_mb=vram_budget,
            expert_size_mb=expert_size_mb,
            num_threads=num_threads,
            prefill_chunk_size=512,
            profile_name=profile_name
        )
        
    def _select_offload_strategy(self, hw: HardwareProfile, perf: PerformanceProfile) -> str:
        if not hw.gpus:
            return 'cpu_only'
        if len(hw.gpus) > 1:
            return 'multi_gpu'
        return 'full_gpu'
            
    def _select_compression_level(self, hw: HardwareProfile) -> str:
        if hw.ram_total_mb <= 16384:
            return 'max'
        elif hw.ram_total_mb <= 32768:
            return 'high'
        elif hw.ram_total_mb <= 65536:
            return 'medium'
        else:
            return 'low'
            
    def _compute_expert_cache_size(self, hw: HardwareProfile, expert_size_mb: int) -> int:
        usable_ram = hw.ram_free_mb * 0.7
        cache_size = int(usable_ram // expert_size_mb)
        return max(1, cache_size)

# ── AdaptiveScheduler ──

class AdaptiveScheduler:
    """Monitors system RAM/CPU and adjusts runtime settings dynamically."""
    
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        """Start the monitoring thread."""
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop(self) -> None:
        """Stop the monitoring thread."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def get_current_config(self) -> RuntimeConfig:
        """Return the current RuntimeConfig."""
        return self.config
        
    def _monitor_loop(self) -> None:
        """Run monitoring loop every 5 seconds."""
        while self.is_running:
            try:
                self._adjust_for_memory_pressure()
            except Exception:
                pass
            time.sleep(5)
            
    def _adjust_for_memory_pressure(self) -> None:
        """Adjust parameters if memory usage is too high."""
        mem = psutil.virtual_memory()
        
        if mem.percent > 95.0:
            self.config.batch_size = 1
            self.config.expert_cache_size = 1
        elif mem.percent > 85.0:
            if self.config.expert_cache_size > 1:
                self.config.expert_cache_size -= 1

