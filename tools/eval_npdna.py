"""
Tantra-LLM / NP-DNA Benchmark & Perplexity Evaluator

Evaluates NP-DNA checkpoints for:
  1. Loss & Perplexity (PPL = exp(loss)) over evaluation text datasets
  2. Single & Batch token throughput (tokens/sec)
  3. Latency per token (ms/token)
  4. CPU Thread scaling performance (1, 2, 4, 8 threads)

Usage:
    python tools/scripts/eval_npdna.py
    python tools/scripts/eval_npdna.py --checkpoint model/latest --threads 4
"""

import os
import sys
import math
import time
import argparse
import psutil
import torch
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from npdna import NpDnaCore
from npdna.train import eval_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate NP-DNA model perplexity, throughput, and latency")
    p.add_argument("--checkpoint", type=str, default="model/latest", help="Path to NP-DNA checkpoint")
    p.add_argument("--eval-file", type=str, default="data/seed_chat.jsonl", help="Evaluation text or jsonl file")
    p.add_argument("--seq-len", type=int, default=128, help="Evaluation sequence window length")
    p.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size")
    p.add_argument("--max-samples", type=int, default=100, help="Maximum evaluation samples")
    p.add_argument("--threads", type=int, default=None, help="Force specific CPU thread count")
    return p.parse_args()


def load_eval_texts(path: Path, max_samples: int) -> list[str]:
    texts = []
    if not path.exists():
        # Fallback default texts for benchmarking
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Write a Python function to compute Fibonacci numbers efficiently.",
            "Explain quantum computing principles in simple terms.",
            "NP-DNA is a high-performance neuro-plastic architecture optimized for CPU inference.",
        ] * (max_samples // 4 + 1)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{") and "text" in line:
                import json
                try:
                    data = json.loads(line)
                    texts.append(data.get("text", line))
                except Exception:
                    texts.append(line)
            else:
                texts.append(line)
            if len(texts) >= max_samples:
                break
    return texts


def evaluate_perplexity(core: NpDnaCore, texts: list[str], batch_size: int, seq_len: int) -> tuple[float, float]:
    """Computes average loss and perplexity (exp(loss))."""
    model = core.model
    tokenizer = core.tokenizer

    ids_list = [tokenizer.encode(t, allow_growth=False) for t in texts if t]
    ids_list = [ids for ids in ids_list if len(ids) > 1]

    if not ids_list:
        return float("nan"), float("nan")

    padded_ids_list = []
    for ids in ids_list:
        if len(ids) < seq_len + 1:
            padded = ids + [0] * (seq_len + 1 - len(ids))
        else:
            padded = ids[:seq_len + 1]
        padded_ids_list.append(padded)

    val_loss, perplexity = eval_model(model, padded_ids_list, batch_size=batch_size, seq_len=seq_len)
    return val_loss, perplexity


def benchmark_generation(core: NpDnaCore, prompt: str = "Explain machine learning.", n_tokens: int = 50) -> dict:
    """Measures generation latency and throughput."""
    # Warmup
    _ = core.generate(prompt, max_tokens=5, temperature=0.0)

    start_time = time.perf_counter()
    output = core.generate(prompt, max_tokens=n_tokens, temperature=0.35, top_k=30, top_p=0.9, context_window=256)
    elapsed_sec = time.perf_counter() - start_time

    generated_tokens = len(core.encode(output, allow_growth=False))
    tokens_per_sec = generated_tokens / elapsed_sec if elapsed_sec > 0 else 0.0
    ms_per_token = (elapsed_sec / generated_tokens * 1000) if generated_tokens > 0 else 0.0

    return {
        "output": output,
        "elapsed_sec": elapsed_sec,
        "generated_tokens": generated_tokens,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": ms_per_token,
    }


def main():
    args = parse_args()

    if args.threads:
        torch.set_num_threads(max(1, args.threads))

    active_threads = torch.get_num_threads()
    cpu_count = os.cpu_count() or 4
    mem_info = psutil.virtual_memory()

    print(f"\n{'='*65}")
    print(f"  NP-DNA BENCHMARK & PERPLEXITY EVALUATOR")
    print(f"  Checkpoint:     {args.checkpoint}")
    print(f"  CPU Threads:    {active_threads} / {cpu_count} logical cores")
    print(f"  System Memory:  {mem_info.used / (1024**3):.2f} GB used / {mem_info.total / (1024**3):.2f} GB total")
    print(f"{'='*65}\n")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint path '{ckpt_path}' does not exist.")
        sys.exit(1)

    print(f"--> Loading model checkpoint: {ckpt_path}...")
    core = NpDnaCore.load(str(ckpt_path))
    param_count = sum(p.numel() for p in core.model.parameters())
    print(f"    Loaded NP-DNA model with {param_count:,} parameters.")
    print(f"    Vocab size: {core.tokenizer.size} / {core.tokenizer.max_capacity}")

    print("\n--> Evaluating Perplexity on evaluation dataset...")
    eval_texts = load_eval_texts(Path(args.eval_file), args.max_samples)
    print(f"    Loaded {len(eval_texts)} evaluation records.")
    loss, ppl = evaluate_perplexity(core, eval_texts, batch_size=args.batch_size, seq_len=args.seq_len)
    print(f"    Validation Loss: {loss:.4f}")
    print(f"    Perplexity (PPL): {ppl:.2f}")

    print("\n--> Benchmarking Generation Speed...")
    gen_results = benchmark_generation(core, prompt="Explain the theory of relativity.", n_tokens=64)
    print(f"    Output Sample:    {repr(gen_results['output'][:80])}...")
    print(f"    Generated Tokens: {gen_results['generated_tokens']}")
    print(f"    Total Time:       {gen_results['elapsed_sec']:.3f}s")
    print(f"    Throughput:       {gen_results['tokens_per_sec']:.2f} tokens/sec")
    print(f"    Latency:          {gen_results['ms_per_token']:.2f} ms/token")

    print("\n--> Thread Scaling Benchmark...")
    print(f"    {'Threads':<10} | {'Tokens/Sec':<14} | {'ms/Token':<12} | {'Time (s)':<10}")
    print(f"    {'-'*10}-|-{'-'*14}-|-{'-'*12}-|-{'-'*10}")
    for t in [1, 2, 4, 8]:
        if t > cpu_count * 2:
            continue
        torch.set_num_threads(t)
        bench = benchmark_generation(core, prompt="Write a python script to sort numbers.", n_tokens=32)
        print(f"    {t:<10} | {bench['tokens_per_sec']:<14.2f} | {bench['ms_per_token']:<12.2f} | {bench['elapsed_sec']:<10.3f}")

    print(f"\n{'='*65}")
    print("  EVALUATION COMPLETE")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
