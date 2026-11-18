"""Run a real TantraModel forward/backward/optimizer smoke test on CPU or CUDA.

This is an execution/backend test, NOT a language-quality benchmark. It uses a
fixed synthetic token sequence to verify that gradients flow and parameters
update. It never reads or writes Model/, datasets, or existing checkpoints.

Examples (PowerShell):
  python -m Tantra.hybrid_smoke --device auto --preset tiny --steps 30
  python -m Tantra.hybrid_smoke --device cuda --preset small --steps 30
  python -m Tantra.hybrid_smoke --device cpu --preset tiny --steps 30

Start with tiny on unknown/old GPUs. Try small only after tiny succeeds.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import psutil
import torch

from Tantra import ROOT, get_preset
from Tantra.train import TantraModel


def mib(value: int) -> float:
    return round(value / (1024 ** 2), 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preset", choices=("tiny", "small", "base"), default="tiny")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "hybrid-smoke-001")
    args = parser.parse_args()

    if min(args.steps, args.batch_size, args.seq_len, args.threads) < 1:
        parser.error("steps, batch-size, seq-len and threads must be positive")

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    cuda_available = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_available:
        parser.error("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and cuda_available) else "cpu")

    cfg = get_preset(args.preset)
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    model = TantraModel(cfg).to(device)
    model.set_active([])
    model.train()

    # Fixed synthetic sequence: repeated training makes this a tiny overfit test.
    # It validates the compute/training path only; it says nothing about language.
    ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len),
                        dtype=torch.long, device=device)
    labels = ids.clone()
    optimizer = torch.optim.AdamW(model.base_parameters(), lr=args.lr, weight_decay=0.0)
    first_param = next(model.base_parameters())
    before_param = first_param.detach().clone()
    rss_before = psutil.Process().memory_info().rss

    gpu = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu = {
            "name": props.name,
            "total_vram_mib": mib(props.total_memory),
            "capability": f"{props.major}.{props.minor}",
            "torch_cuda": torch.version.cuda,
            "compiled_arches": torch.cuda.get_arch_list(),
        }
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    try:
        with torch.no_grad():
            initial_loss = float(model.loss(ids, labels).item())
        started = time.perf_counter()
        losses = []
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(ids, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            loss.backward()
            grad = first_param.grad
            if grad is None or not torch.isfinite(grad).all():
                raise FloatingPointError(f"missing/non-finite gradient at step {step + 1}")
            optimizer.step()
            losses.append(float(loss.detach().item()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        with torch.no_grad():
            final_loss = float(model.loss(ids, labels).item())
        parameter_delta = float((first_param.detach() - before_param).abs().max().item())
        result = {
            "status": "passed" if parameter_delta > 0 and math.isfinite(final_loss) else "failed",
            "scope": "synthetic fixed-batch compute smoke test; NOT language-quality evidence",
            "device": str(device),
            "gpu": gpu,
            "preset": args.preset,
            "parameters": model.num_params(),
            "batch_size": args.batch_size,
            "sequence_length": args.seq_len,
            "steps": args.steps,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "mean_step_loss": round(sum(losses) / len(losses), 6),
            "max_abs_parameter_change": parameter_delta,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(args.steps * args.batch_size * (args.seq_len - 1) / max(elapsed, 1e-9), 2),
            "process_rss_before_mib": mib(rss_before),
            "process_rss_after_mib": mib(psutil.Process().memory_info().rss),
            "peak_cuda_allocated_mib": mib(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_cuda_reserved_mib": mib(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torch_cuda_available": cuda_available,
            "note": "This test keeps the full model and optimizer on one device. It does NOT implement CPU/GPU offloading.",
        }
    except (RuntimeError, FloatingPointError) as exc:
        if device.type == "cuda":
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "device": str(device),
            "gpu": gpu,
            "preset": args.preset,
            "parameters": model.num_params(),
            "note": "No checkpoint or Model/ files were modified. CUDA OOM/unsupported-kernel errors are recorded as experiment failures.",
        }

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / f"{args.device}-{args.preset}.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Report: {report_path}", flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
