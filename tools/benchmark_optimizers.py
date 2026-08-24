"""
tools/benchmark_optimizers.py — Controlled Optimizer Comparison Benchmark.
Directly compares AdamW vs. Lion across multiple learning rates and random seeds
on the official 38.6M CPU profile under identical data batches and steps.
Measures memory RSS footprint, step throughput (tok/s), and loss trajectory.
"""
import os
import sys
import time
import math
import statistics
import psutil
import torch

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Tantra.model import build_cpu_model
from Tantra.train import NeuroTrainer
from Tantra.dataset import JSONLDataset, DataLoader
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.config import VocabConfig


def get_process_rss_mb() -> float:
    """Return resident set size (RSS) memory in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_optimizer_trial(
    opt_name: str,
    lr: float,
    weight_decay: float,
    steps: int = 20,
    batch_size: int = 2,
    seq_len: int = 128,
    seed: int = 42,
    dataset_file: str = "Datasets/staged_master.jsonl"
) -> dict:
    torch.manual_seed(seed)
    model = build_cpu_model("dense", attention_kind="causal")
    
    bpe = ByteBPETokenizer.load("Model/tokenizer.json", VocabConfig())
    tok = UnifiedTokenizer(VocabConfig(), bpe, MegabytePatcher())
    ds = JSONLDataset(dataset_file, tokenizer=tok, seq_len=seq_len)
    loader = DataLoader(ds, batch_size=batch_size)

    rss_before = get_process_rss_mb()
    trainer = NeuroTrainer(
        model,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_name=opt_name,
        total_steps=steps,
        warmup_steps=max(5, steps // 10)
    )

    t0 = time.perf_counter()
    losses = trainer.train_dataset(loader, max_steps=steps, log_every=steps)
    wall_time = time.perf_counter() - t0
    rss_after = get_process_rss_mb()
    tok_s = (steps * batch_size * seq_len) / max(wall_time, 1e-6)

    return {
        "opt_name": opt_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "wall_time": wall_time,
        "tok_s": tok_s,
        "initial_loss": losses[0] if losses else 0.0,
        "final_loss": losses[-1] if losses else 0.0,
        "loss_delta": (losses[-1] - losses[0]) if len(losses) > 1 else 0.0,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": max(0.0, rss_after - rss_before)
    }


def run_benchmark(steps: int = 20, batch_size: int = 2, seq_len: int = 128, multi_seed: bool = True):
    print("=" * 78)
    print("         TANTRA OPTIMIZER BENCHMARK: EXTENDED ADAMW vs. LION SWEEP")
    print(f"         Official 38.6M CPU Profile | Device: CPU ({torch.get_num_threads()} threads)")
    print(f"         Steps: {steps} | Batch: {batch_size} | Context: {seq_len} tokens")
    print("=" * 78)

    # 1. Extended Learning Rate Sweep
    experiments = [
        ("adamw", 1e-4, 0.01),
        ("lion",  1e-5, 0.05),
        ("lion",  3e-5, 0.05),
        ("lion",  5e-5, 0.05),
        ("lion",  8e-5, 0.05),
        ("lion",  1e-4, 0.05),
    ]

    print("\n[PART 1/2] Extended Learning Rate Exploration (Seed 42):")
    results = []
    for opt_name, lr, wd in experiments:
        print(f"▶ Running {opt_name.upper()} (LR={lr:.1e}, WD={wd})...", end="", flush=True)
        res = run_optimizer_trial(opt_name, lr, wd, steps=steps, batch_size=batch_size, seq_len=seq_len, seed=42)
        results.append(res)
        print(f" -> {res['tok_s']:5.1f} tok/s | Loss: {res['initial_loss']:.4f} -> {res['final_loss']:.4f} ({res['wall_time']:.1f}s)")

    print("\n" + "-" * 78)
    print(f"{'Optimizer':<10} | {'LR':<8} | {'WD':<6} | {'Speed (tok/s)':<14} | {'Initial Loss':<12} | {'Final Loss':<11} | {'Time (s)'}")
    print("-" * 78)
    for r in results:
        print(f"{r['opt_name'].upper():<10} | {r['lr']:<8.1e} | {r['weight_decay']:<6.2f} | {r['tok_s']:<14.1f} | {r['initial_loss']:<12.4f} | {r['final_loss']:<11.4f} | {r['wall_time']:<7.2f}")
    print("-" * 78)

    # 2. Multi-Seed Variance Evaluation
    if multi_seed:
        print("\n[PART 2/2] Multi-Seed Variance Verification (3 Seeds: 42, 100, 2026):")
        configs_to_test = [
            ("adamw", 1e-4, 0.01),
            ("lion",  5e-5, 0.05),
            ("lion",  8e-5, 0.05),
        ]
        seeds = [42, 100, 2026]

        summary_stats = []
        for opt_name, lr, wd in configs_to_test:
            seed_losses = []
            seed_speeds = []
            seed_times = []
            for s in seeds:
                res = run_optimizer_trial(opt_name, lr, wd, steps=steps, batch_size=batch_size, seq_len=seq_len, seed=s)
                seed_losses.append(res["final_loss"])
                seed_speeds.append(res["tok_s"])
                seed_times.append(res["wall_time"])
            
            mean_loss, std_loss = statistics.mean(seed_losses), statistics.stdev(seed_losses)
            mean_speed, std_speed = statistics.mean(seed_speeds), statistics.stdev(seed_speeds)
            mean_time, std_time = statistics.mean(seed_times), statistics.stdev(seed_times)
            
            summary_stats.append({
                "config": f"{opt_name.upper()} (LR={lr:.1e})",
                "loss": f"{mean_loss:.4f} ± {std_loss:.4f}",
                "speed": f"{mean_speed:.1f} ± {std_speed:.1f} tok/s",
                "time": f"{mean_time:.2f} ± {std_time:.2f} s"
            })

        print("\n" + "=" * 78)
        print("                  MULTI-SEED VARIANCE SUMMARY (Mean ± Std)")
        print("=" * 78)
        print(f"{'Configuration':<22} | {'Final Loss':<18} | {'Throughput (tok/s)':<22} | {'Wall Time'}")
        print("-" * 78)
        for stat in summary_stats:
            print(f"{stat['config']:<22} | {stat['loss']:<18} | {stat['speed']:<22} | {stat['time']}")
        print("=" * 78)


if __name__ == "__main__":
    run_benchmark(steps=20, batch_size=2, seq_len=128, multi_seed=True)
