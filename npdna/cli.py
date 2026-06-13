"""Small command-line entrypoints for NP-DNA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .model import NpDnaCore


def _ensure_utf8() -> None:
    if sys.stdout.encoding.lower() != "utf-8":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def info_main() -> None:
    _ensure_utf8()
    core = NpDnaCore.from_config("seed")
    print("NP-DNA seed configuration")
    print(f"  hidden_size  = {core.config.hidden_size}")
    print(f"  layers       = {core.config.num_layers}")
    print(f"  initial_vocab = {core.config.initial_vocab}")
    print(f"  max_vocab    = {core.config.max_vocab}")
    print(f"  parameters   = {core.model.parameter_count():,}")


def chat_main() -> None:
    _ensure_utf8()
    parser = argparse.ArgumentParser(description="Generate text with NP-DNA.")
    parser.add_argument("prompt", nargs="?", default="Hello.", help="Prompt to generate from.")
    parser.add_argument("--checkpoint", default="model/npdna_v3/best", help="Checkpoint directory.")
    parser.add_argument("--max-tokens", type=int, default=80, help="Maximum generated tokens.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        core = NpDnaCore.load(checkpoint)
    else:
        core = NpDnaCore.from_config("seed")

    print(core.generate(args.prompt, max_tokens=args.max_tokens))
