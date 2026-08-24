"""
tools/benchmark_optimizers.py — Fair & Symmetric Optimizer Peak Bracketing.
Sweeps both AdamW and Lion across their full operational LR spectra to bracket
their empirical optima, and compares Best-AdamW vs Best-Lion across 5 independent seeds.
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
    steps: int = 40,
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


def run_benchmark(steps: int = 40, batch_size: int = 2, seq_len: int = 128, multi_seed: bool = True):
    print("=" * 80)
    print("      TANTRA OPTIMIZER BENCHMARK: FAIR BEST-VS-BEST COMPARISON")
    print(f"      Official 38.6M CPU Profile | Device: CPU ({torch.get_num_threads()} threads)")
    print(f"      Horizon: {steps} steps | Batch: {batch_size} | Context: {seq_len} tokens")
    print("=" * 80)

    # 1. Symmetric Full-Spectrum Sweeps (Seed 42)
    adamw_lrs = [5.0e-5, 1.0e-4, 2.0e-4, 3.0e-4, 5.0e-4, 8.0e-4]
    lion_lrs  = [1.0e-4, 1.5e-4, 2.0e-4, 2.5e-4, 3.0e-4, 4.0e-4]

    print("\n[PART 1/3] Sweeping AdamW Across Learning Rates (Seed 42):")
    adamw_results = []
    for lr in adamw_lrs:
        print(f"▶ ADAMW (LR={lr:.1e}, WD=0.01)...", end="", flush=True)
        res = run_optimizer_trial("adamw", lr, weight_decay=0.01, steps=steps, batch_size=batch_size, seq_len=seq_len, seed=42)
        adamw_results.append(res)
        print(f" -> Loss: {res['initial_loss']:.4f} -> {res['final_loss']:.4f} | {res['tok_s']:5.1f} tok/s ({res['wall_time']:.1f}s)")

    print("\n[PART 2/3] Sweeping Lion Across Learning Rates (Seed 42):")
    lion_results = []
    for lr in lion_lrs:
        print(f"▶ LION  (LR={lr:.1e}, WD=0.05)...", end="", flush=True)
        res = run_optimizer_trial("lion", lr, weight_decay=0.05, steps=steps, batch_size=batch_size, seq_len=seq_len, seed=42)
        lion_results.append(res)
        print(f" -> Loss: {res['initial_loss']:.4f} -> {res['final_loss']:.4f} | {res['tok_s']:5.1f} tok/s ({res['wall_time']:.1f}s)")

    print("\n" + "-" * 80)
    print(f"{'Optimizer':<10} | {'LR':<8} | {'WD':<6} | {'Speed (tok/s)':<14} | {'Initial Loss':<12} | {'Final Loss':<11} | {'Time (s)'}")
    print("-" * 80)
    for r in adamw_results + lion_results:
        print(f"{r['opt_name'].upper():<10} | {r['lr']:<8.1e} | {r['weight_decay']:<6.2f} | {r['tok_s']:<14.1f} | {r['initial_loss']:<12.4f} | {r['final_loss']:<11.4f} | {r['wall_time']:<7.2f}")
    print("-" * 80)

    # 2. Identify Best Points
    best_adamw = min(adamw_results, key=lambda x: x["final_loss"])
    best_lion = min(lion_results, key=lambda x: x["final_loss"])
    print(f"\nEmpirical Optimums Found:")
    print(f"  • Best AdamW: LR = {best_adamw['lr']:.1e} (Loss = {best_adamw['final_loss']:.4f})")
    print(f"  • Best Lion : LR = {best_lion['lr']:.1e} (Loss = {best_lion['final_loss']:.4f})")

    # 3. 5-Seed Statistical Verification: Best AdamW vs Best Lion
    if multi_seed:
        print(f"\n[PART 3/3] 5-Seed Statistical Verification (Best AdamW @ {best_adamw['lr']:.1e} vs Best Lion @ {best_lion['lr']:.1e}):")
        
        configs_to_test = [
            ("adamw", best_adamw["lr"], 0.01),
            ("lion",  best_lion["lr"],  0.05),
        ]
        seeds = [42, 100, 2026, 777, 999]

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
            se_loss = std_loss / math.sqrt(len(seeds))
            mean_speed, std_speed = statistics.mean(seed_speeds), statistics.stdev(seed_speeds)
            mean_time, std_time = statistics.mean(seed_times), statistics.stdev(seed_times)
            
            summary_stats.append({
                "config": f"{opt_name.upper()} (LR={lr:.1e})",
                "loss_mean_std": f"{mean_loss:.4f} ± {std_loss:.4f}",
                "loss_se": f"±{se_loss:.4f}",
                "speed": f"{mean_speed:.1f} ± {std_speed:.1f} tok/s",
                "time": f"{mean_time:.2f} ± {std_time:.2f} s"
            })

        print("\n" + "=" * 80)
        print("         5-SEED STATISTICAL SUMMARY (BEST vs. BEST, SE = σ/√5)")
        print("=" * 80)
        print(f"{'Configuration':<22} | {'Mean Loss ± Std':<18} | {'Std Error':<10} | {'Throughput (tok/s)':<20} | {'Wall Time'}")
        print("-" * 80)
        for stat in summary_stats:
            print(f"{stat['config']:<22} | {stat['loss_mean_std']:<18} | {stat['loss_se']:<10} | {stat['speed']:<20} | {stat['time']}")
        print("=" * 80)


if __name__ == "__main__":
    run_benchmark(steps=40, batch_size=2, seq_len=128, multi_seed=True)
