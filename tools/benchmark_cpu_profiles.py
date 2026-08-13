"""Benchmark CPU model alternatives without affecting a training checkpoint.

Run only while dataset training is stopped; competing with active training makes
the result meaningless.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Tantra.cpu_profiles import build_cpu_model
from Tantra.hardware import configure_cpu_performance


def benchmark(name: str, model: torch.nn.Module, batch_size: int, seq_len: int, steps: int) -> None:
    model.train()
    vocab_size = model.config.vocab.vocab_size
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, vocab_size, (batch_size, seq_len))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # Warm-up allocator/kernel paths, then measure full forward/backward/step.
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(x, return_mtp=False, use_latent_reasoning=False)
    criterion(logits.reshape(-1, vocab_size), y.reshape(-1)).backward()
    optimizer.step()
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x, return_mtp=False, use_latent_reasoning=False)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        if hasattr(model, "get_aux_loss"):
            loss = loss + model.get_aux_loss()
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    tokens = batch_size * seq_len * steps
    params = sum(p.numel() for p in model.parameters())
    print(f"{name:12s} | {params / 1e6:6.1f}M params | {tokens / elapsed:7.1f} train tok/s | {elapsed / steps:5.2f} s/step")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    configure_cpu_performance(args.threads)
    for name, profile, attention in [
        ("micro10-alra", "micro10", "alra"),
        ("dense-alra", "dense", "alra"),
        ("dense-causal", "dense", "causal"),
        ("top1-moe-2", "moe2", "alra"),
    ]:
        benchmark(name, build_cpu_model(profile, attention), args.batch_size, args.seq_len, args.steps)


if __name__ == "__main__":
    main()
