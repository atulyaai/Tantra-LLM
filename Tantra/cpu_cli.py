"""Supported CPU-profile commands for training and chat.

This module owns the shared command-line implementation.  It intentionally
lives with the CPU profile code.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from Tantra.config import VocabConfig
from Tantra.model import build_cpu_model
from Tantra.hardware import configure_cpu_performance
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer


def _tokenizer(path: str, vocab_size: int) -> UnifiedTokenizer:
    cfg = VocabConfig(vocab_size=vocab_size)
    return UnifiedTokenizer(cfg, ByteBPETokenizer.load(path, cfg), MegabytePatcher())


def _resolve_device(requested: str) -> str:
    """Return the device string to build the model on.

    ``cpu`` keeps the historical single-threaded CPU path; ``auto`` (the
    default on GPU hosts) picks cuda when available, else cpu.  Colab / GPU
    hosts pass ``--device auto`` and get bf16 autocast for free because
    Tantra/NeuroTrainer already ties autocast to ``self.device``.
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def train(args: argparse.Namespace) -> None:
    # Imported lazily: main imports the package modules and this module must not
    # create an import cycle during normal CLI startup.
    from main import run_dataset_training

    tokenizer = _tokenizer(args.tokenizer, args.vocab_size)
    device = _resolve_device(args.device)
    if device == "cpu":
        configure_cpu_performance()
    model = build_cpu_model(args.profile, attention_kind=args.attention,
                            vocab_size=args.vocab_size).to(device)
    run_dataset_training(
        model, tokenizer, args.dataset, steps=args.steps, resume=args.resume,
        eval_every=args.eval_every, log_every=10,
        checkpoint_every=args.checkpoint_every, batch_size=args.batch_size,
        seq_len=args.seq_len, grad_accumulation_steps=args.grad_accum,
        data_workers=args.data_workers, use_latent_reasoning=False,
        use_mtp_loss=False, training_stage="pretrain",
        auto_growth=args.auto_growth, growth_patience=args.growth_patience,
        growth_min_delta=args.growth_min_delta, max_layers=args.max_layers,
        model_dir=args.model_dir, archive_checkpoints=False,
    )


def chat(args: argparse.Namespace) -> None:
    from main import run_interactive_chat

    checkpoint = os.path.join(args.model_dir, "Latest", "checkpoint_latest.pt")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"CPU checkpoint not found: {checkpoint}")
    tokenizer = _tokenizer(args.tokenizer, args.vocab_size)
    device = _resolve_device(args.device)
    model = build_cpu_model(args.profile, attention_kind=args.attention,
                            vocab_size=args.vocab_size).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    run_interactive_chat(model, tokenizer, device, args.temperature, args.top_p)


def benchmark(args: argparse.Namespace) -> None:
    """Measure comparable CPU profile training throughput."""
    device = _resolve_device(args.device)
    if device == "cpu":
        configure_cpu_performance(args.threads)
    for name, profile, attention in [
        ("micro10-alra", "micro10", "alra"),
        ("dense-alra", "dense", "alra"),
        ("dense-causal", "dense", "causal"),
        ("top1-moe-2", "moe2", "alra"),
    ]:
        model = build_cpu_model(profile, attention).to(device)
        model.train()
        vocab_size = model.config.vocab.vocab_size
        x = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
        y = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x, return_mtp=False, use_latent_reasoning=False)
        criterion(logits.reshape(-1, vocab_size), y.reshape(-1)).backward()
        optimizer.step()
        started = time.perf_counter()
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x, return_mtp=False, use_latent_reasoning=False)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
        elapsed = time.perf_counter() - started
        tokens = args.batch_size * args.seq_len * args.steps
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:12s} | {params / 1e6:6.1f}M params | {tokens / elapsed:7.1f} train tok/s")


def _common_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=["dense", "moe2", "micro10"], default="dense")
    parser.add_argument("--attention", choices=["causal", "alra"], default="causal")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--model-dir", default="Model/CPU_Dense32K")
    parser.add_argument("--tokenizer", default="Model/tokenizer.json")
    parser.add_argument("--device", default="auto", help="cuda/auto/cpu — 'auto' uses CUDA when present")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tantra maintained CPU profile commands")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train_parser = subcommands.add_parser("train", help="Train or resume a CPU profile")
    _common_profile_args(train_parser)
    train_parser.add_argument("--dataset", default="Datasets")
    train_parser.add_argument("--steps", type=int, default=50000)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--grad-accum", type=int, default=1)
    train_parser.add_argument("--seq-len", type=int, default=128)
    train_parser.add_argument("--data-workers", type=int, default=2)
    train_parser.add_argument("--checkpoint-every", type=int, default=500)
    train_parser.add_argument("--eval-every", type=int, default=1000)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--auto-growth", action="store_true")
    train_parser.add_argument("--growth-patience", type=int, default=2000)
    train_parser.add_argument("--growth-min-delta", type=float, default=0.005)
    train_parser.add_argument("--max-layers", type=int, default=None)

    chat_parser = subcommands.add_parser("chat", help="Chat with a CPU profile checkpoint")
    _common_profile_args(chat_parser)
    chat_parser.add_argument("--temperature", type=float, default=0.45)
    chat_parser.add_argument("--top-p", type=float, default=0.80)

    benchmark_parser = subcommands.add_parser("benchmark", help="Benchmark CPU/GPU profile throughput")
    benchmark_parser.add_argument("--batch-size", type=int, default=2)
    benchmark_parser.add_argument("--seq-len", type=int, default=128)
    benchmark_parser.add_argument("--steps", type=int, default=3)
    benchmark_parser.add_argument("--threads", type=int, default=4)
    benchmark_parser.add_argument("--device", default="auto", help="cuda/auto/cpu")

    args = parser.parse_args(argv)
    if args.command == "train":
        train(args)
    elif args.command == "chat":
        chat(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
