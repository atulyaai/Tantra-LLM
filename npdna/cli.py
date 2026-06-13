"""Small command-line entrypoints for NP-DNA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .model import NpDnaCore

DEFAULT_CHECKPOINTS = (
    Path("model/npdna_v3/latest"),
    Path("model/npdna_v3/best"),
)


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
    parser.add_argument("prompt", nargs="?", default=None, help="Prompt to generate from.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory.")
    parser.add_argument("--max-tokens", type=int, default=80, help="Maximum generated tokens.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start a terminal chat loop.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else next(
        (path for path in DEFAULT_CHECKPOINTS if path.exists()),
        DEFAULT_CHECKPOINTS[-1],
    )
    if checkpoint.exists():
        core = NpDnaCore.load(checkpoint)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        core = NpDnaCore.from_config("seed")
        print("Loaded fresh seed model; no checkpoint found.")

    if args.interactive or args.prompt is None:
        print("Type /exit or /quit to stop.")
        while True:
            try:
                prompt = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.lower() in {"/exit", "/quit"}:
                break
            if not prompt:
                continue
            print("npdna>", core.generate(prompt, max_tokens=args.max_tokens))
        return

    print(core.generate(args.prompt, max_tokens=args.max_tokens))


if __name__ == "__main__":
    chat_main()
