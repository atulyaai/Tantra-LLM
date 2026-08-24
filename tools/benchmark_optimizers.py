"""
tools/benchmark_optimizers.py — Controlled Optimizer Comparison Benchmark.
Directly compares AdamW vs. Lion across multiple learning rates on the
official 38.6M CPU profile under identical random seeds, data batches, and steps.
Measures memory RSS footprint, step throughput (tok/s), and loss trajectory.
"""
import os
import sys
import time
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
    steps: int = 30,
    batch_size: int = 2,
    seq_len: int = 128,
    dataset_file: str = "Datasets/staged_master.jsonl"
) -> dict:
    torch.manual_seed(42)
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
        "wall_time": wall_time,
        "tok_s": tok_s,
        "initial_loss": losses[0] if losses else 0.0,
        "final_loss": losses[-1] if losses else 0.0,
        "loss_delta": (losses[-1] - losses[0]) if len(losses) > 1 else 0.0,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": max(0.0, rss_after - rss_before)
    }


def run_benchmark(steps: int = 25, batch_size: int = 2, seq_len: int = 128):
    print("=" * 75)
    print("         TANTRA OPTIMIZER BENCHMARK: ADAMW vs. LION SWEEP")
    print(f"         Official 38.6M CPU Profile | Device: CPU ({torch.get_num_threads()} threads)")
    print(f"         Steps: {steps} | Batch: {batch_size} | Context: {seq_len} tokens")
    print("=" * 75)

    experiments = [
        ("adamw", 1e-4, 0.01),
        ("lion",  1e-5, 0.05),
        ("lion",  3e-5, 0.05),
        ("lion",  5e-5, 0.05),
    ]

    results = []
    for opt_name, lr, wd in experiments:
        print(f"\n▶ Running Trial: {opt_name.upper()} (LR={lr:.1e}, WD={wd})...", flush=True)
        res = run_optimizer_trial(opt_name, lr, wd, steps=steps, batch_size=batch_size, seq_len=seq_len)
        results.append(res)
        print(f"  ✓ Finished in {res['wall_time']:.2f}s ({res['tok_s']:.1f} tok/s) | Final Loss: {res['final_loss']:.4f}")

    print("\n" + "=" * 75)
    print("                      OPTIMIZER COMPARISON SUMMARY")
    print("=" * 75)
    print(f"{'Optimizer':<10} | {'LR':<8} | {'WD':<6} | {'Speed (tok/s)':<14} | {'Initial Loss':<12} | {'Final Loss':<11} | {'Time (s)'}")
    print("-" * 75)
    for r in results:
        print(f"{r['opt_name'].upper():<10} | {r['lr']:<8.1e} | {r['weight_decay']:<6.2f} | {r['tok_s']:<14.1f} | {r['initial_loss']:<12.4f} | {r['final_loss']:<11.4f} | {r['wall_time']:<7.2f}")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark(steps=20, batch_size=2, seq_len=128)
