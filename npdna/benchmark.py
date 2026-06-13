"""Benchmarking and release helpers for NP-DNA checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from .model import NpDnaCore

PROMPTS = [
    "Hello. How are you?",
    "What is gravity?",
    "Write a Python function.",
    "What is machine learning?",
    "Explain memory in one sentence.",
]


def benchmark_checkpoint(
    checkpoint: str | Path = "model/npdna_v3/best",
    *,
    prompts: list[str] | None = None,
    max_tokens: int = 40,
) -> dict:
    checkpoint = Path(checkpoint)
    prompts = prompts or PROMPTS

    t0 = time.perf_counter()
    core = NpDnaCore.load(checkpoint)
    load_sec = time.perf_counter() - t0

    generations = []
    total_chars = 0
    total_sec = 0.0
    for prompt in prompts:
        g0 = time.perf_counter()
        text = core.generate(prompt, max_tokens=max_tokens)
        gen_sec = time.perf_counter() - g0
        total_sec += gen_sec
        total_chars += len(text)
        generations.append({
            "prompt": prompt,
            "text": text,
            "seconds": gen_sec,
            "chars": len(text),
        })

    meta_path = checkpoint / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "checkpoint": str(checkpoint),
        "load_seconds": load_sec,
        "generation_seconds": total_sec,
        "chars_per_second": total_chars / max(total_sec, 1e-9),
        "prompts": len(prompts),
        "max_tokens": max_tokens,
        "metadata": {
            "step": metadata.get("step"),
            "val_loss": metadata.get("val_loss"),
            "parameter_count": metadata.get("parameter_count"),
            "active_parameter_count": metadata.get("active_parameter_count"),
            "vocab_size": metadata.get("vocab_size"),
            "vocab_capacity": metadata.get("vocab_capacity"),
            "hidden_size": metadata.get("hidden_size"),
        },
        "generations": generations,
    }


def write_benchmark(
    checkpoint: str | Path = "model/npdna_v3/best",
    output: str | Path | None = None,
    *,
    max_tokens: int = 40,
) -> dict:
    result = benchmark_checkpoint(checkpoint, max_tokens=max_tokens)
    out_path = Path(output) if output else Path(checkpoint) / "benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    return result


def export_release(
    version: str,
    checkpoint: str | Path = "model/npdna_v3/best",
    releases_dir: str | Path = "model/releases",
) -> Path:
    checkpoint = Path(checkpoint)
    dest = Path(releases_dir) / version
    dest.mkdir(parents=True, exist_ok=True)

    for name in ["model.pt", "tokenizer.json", "metadata.json", "benchmark.json"]:
        src = checkpoint / name
        if src.exists():
            shutil.copy2(src, dest / name)

    cortex_src = checkpoint / "cortex"
    if cortex_src.exists():
        cortex_dest = dest / "cortex"
        if cortex_dest.exists():
            shutil.rmtree(cortex_dest)
        shutil.copytree(cortex_src, cortex_dest)

    bench_path = dest / "benchmark.json"
    if not bench_path.exists():
        write_benchmark(checkpoint, bench_path)

    samples = json.loads(bench_path.read_text(encoding="utf-8"))["generations"]
    (dest / "samples.txt").write_text(
        "\n\n".join(f"Q: {item['prompt']}\nA: {item['text']}" for item in samples),
        encoding="utf-8",
    )
    return dest


def benchmark_main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an NP-DNA checkpoint.")
    parser.add_argument("--checkpoint", default="model/npdna_v3/best")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-tokens", type=int, default=40)
    args = parser.parse_args()
    result = write_benchmark(args.checkpoint, args.output, max_tokens=args.max_tokens)
    print(json.dumps(result["metadata"], indent=2))
    print(f"load_seconds={result['load_seconds']:.3f}")
    print(f"chars_per_second={result['chars_per_second']:.1f}")


def release_main() -> None:
    parser = argparse.ArgumentParser(description="Export a versioned NP-DNA release.")
    parser.add_argument("version", help="Version folder name, e.g. npdna-seed-v0.1")
    parser.add_argument("--checkpoint", default="model/npdna_v3/best")
    parser.add_argument("--releases-dir", default="model/releases")
    args = parser.parse_args()
    dest = export_release(args.version, args.checkpoint, args.releases_dir)
    print(dest)
