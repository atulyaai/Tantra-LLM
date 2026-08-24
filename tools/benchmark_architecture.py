"""
tools/benchmark_architecture.py — Objective Architectural Benchmark Harness.
Directly compares Standard Dense Transformer vs. NeuroCore ALRA Linear Attention
using the official 38.6M CPU profile (build_cpu_model) under identical parameters.
"""
import os
import sys
import time
import json
import torch

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Tantra.model import build_cpu_model
from Tantra.train import NeuroTrainer
from Tantra.dataset import JSONLDataset, DataLoader
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.config import VocabConfig


def run_benchmark(steps: int = 10, batch_size: int = 2, seq_len: int = 128, seq_scaling: bool = True):
    print("=" * 70)
    print("      TANTRA ARCHITECTURE BENCHMARK: DENSE vs. ALRA LINEAR")
    print(f"      Official 38.6M CPU Profile | Device: CPU ({torch.get_num_threads()} threads)")
    print("=" * 70)

    # 1. Model Instantiation using official CPU builders
    print("\n[1/3] Instantiating Official 38.6M CPU Profile (512-dim, 8-layers, expansion=2)...")
    dense_model = build_cpu_model("dense", attention_kind="causal")
    alra_model = build_cpu_model("dense", attention_kind="alra")

    dense_params = sum(p.numel() for p in dense_model.parameters())
    alra_params = sum(p.numel() for p in alra_model.parameters())

    print(f"  • Standard Dense Transformer Parameters : {dense_params:,} (~38.6M)")
    print(f"  • NeuroCore (ALRA Linear) Parameters    : {alra_params:,} (~38.6M)")

    # 2. Sequence Length Scaling Benchmark (Memory & Latency Scaling)
    if seq_scaling:
        print("\n[2/3] Sequence Length Latency Benchmark (Forward Pass):")
        print(f"  {'Seq Len':<10} | {'Dense (Causal Softmax)':<25} | {'NeuroCore (ALRA Linear)':<25} | {'Speedup'}")
        print("  " + "-" * 70)

        for sl in [64, 128, 256, 512]:
            x = torch.randint(0, 32768, (1, sl))
            
            # Warmup
            _ = dense_model(x)
            _ = alra_model(x)

            # Benchmark Dense
            t0 = time.perf_counter()
            for _ in range(5):
                _ = dense_model(x)
            dense_ms = ((time.perf_counter() - t0) / 5) * 1000

            # Benchmark ALRA
            t0 = time.perf_counter()
            for _ in range(5):
                _ = alra_model(x)
            alra_ms = ((time.perf_counter() - t0) / 5) * 1000

            speedup = dense_ms / max(alra_ms, 1e-6)
            print(f"  {sl:<10} | {dense_ms:10.2f} ms             | {alra_ms:10.2f} ms               | {speedup:5.2f}x")

    # 3. Training Step Throughput & Loss Convergence
    print(f"\n[3/3] Training Step Benchmark ({steps} optimizer steps, batch={batch_size}, seq_len={seq_len}):")
    dataset_file = "Datasets/staged_master.jsonl"
    
    bpe = ByteBPETokenizer.load("Model/tokenizer.json", VocabConfig())
    tok = UnifiedTokenizer(VocabConfig(), bpe, MegabytePatcher())
    ds = JSONLDataset(dataset_file, tokenizer=tok, seq_len=seq_len)
    loader = DataLoader(ds, batch_size=batch_size)

    print("  Running Dense Baseline...")
    dense_trainer = NeuroTrainer(dense_model, lr=1e-4, total_steps=steps)
    t0 = time.perf_counter()
    dense_losses = dense_trainer.train_dataset(loader, max_steps=steps, log_every=steps)
    dense_wall_time = time.perf_counter() - t0
    dense_tok_s = (steps * batch_size * seq_len) / max(dense_wall_time, 1e-6)

    print("  Running ALRA Linear...")
    alra_trainer = NeuroTrainer(alra_model, lr=1e-4, total_steps=steps)
    t0 = time.perf_counter()
    alra_losses = alra_trainer.train_dataset(loader, max_steps=steps, log_every=steps)
    alra_wall_time = time.perf_counter() - t0
    alra_tok_s = (steps * batch_size * seq_len) / max(alra_wall_time, 1e-6)

    print("\n" + "=" * 70)
    print("                    BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"  • Parameter Count      : Dense: {dense_params:,} | ALRA: {alra_params:,}")
    print(f"  • Training Throughput  : Dense: {dense_tok_s:8.1f} tok/s | ALRA: {alra_tok_s:8.1f} tok/s ({alra_tok_s/dense_tok_s:.2f}x)")
    print(f"  • Wall-Clock Time      : Dense: {dense_wall_time:8.2f} s     | ALRA: {alra_wall_time:8.2f} s")
    print(f"  • Final Loss @ {steps} steps: Dense: {dense_losses[-1]:8.4f}   | ALRA: {alra_losses[-1]:8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark(steps=10, batch_size=2, seq_len=128, seq_scaling=True)
