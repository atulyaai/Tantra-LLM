"""Train an AtulyaTokenizer from a chat JSONL file."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_tokenizer_class():
    path = Path(__file__).resolve().parents[1] / "npdna" / "tokenizer.py"
    spec = importlib.util.spec_from_file_location("npdna_tokenizer_standalone", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load tokenizer module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.AtulyaTokenizer


AtulyaTokenizer = load_tokenizer_class()


def iter_texts(path: Path, limit: int | None = None):
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            system = row.get("system") or ""
            user = row.get("user") or row.get("instruction") or row.get("prompt") or ""
            assistant = row.get("assistant") or row.get("output") or row.get("answer") or ""
            text = f"{system}\nUser: {user}\nAssistant: {assistant}"
            yield text
            count += 1
            if limit is not None and count >= limit:
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("Download/tantra_train_plus_200k.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("model/tokenizer/tokenizer_tantra_16k.json"))
    parser.add_argument("--capacity", type=int, default=16_384)
    parser.add_argument("--max-capacity", type=int, default=65_536)
    parser.add_argument("--merges", type=int, default=12_000)
    parser.add_argument("--limit", type=int, default=150_000)
    parser.add_argument("--min-pair-freq", type=int, default=3)
    args = parser.parse_args()

    tok = AtulyaTokenizer(initial_capacity=args.capacity, max_capacity=args.max_capacity)
    print(f"Training tokenizer on {args.input}")
    print(f"capacity={args.capacity}, max_capacity={args.max_capacity}, target_merges={args.merges}")
    tok.train_bpe(
        iter_texts(args.input, limit=args.limit),
        target_merges=args.merges,
        max_words=250_000,
        min_pair_freq=args.min_pair_freq,
    )
    # Keep embedding capacity aligned to the requested capacity unless the
    # tokenizer grew above it.
    if tok.capacity < args.capacity:
        tok._capacity = args.capacity
    tok.save(args.output)
    print(f"Saved {args.output}")
    print(f"vocab_size={tok.size}, capacity={tok.capacity}, fill={tok.fill_ratio:.1%}, merges={len(tok.merges)}")


if __name__ == "__main__":
    main()
