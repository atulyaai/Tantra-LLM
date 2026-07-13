#!/usr/bin/env python3
"""
Merged utility scripts.
Usage:
    python tools/utils.py train-tokenizer --input path/to/data.jsonl
    python tools/utils.py merge-checkpoints ckpt1 ckpt2 -o merged
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Utility scripts")
    subparsers = parser.add_subparsers(dest="command")

    p = subparsers.add_parser("train-tokenizer", help="Train a tokenizer from JSONL")
    p.add_argument("--input", type=str, default="Download/tantra_train_plus_200k.jsonl")
    p.add_argument("--output", type=str, default="model/tokenizer/tokenizer_tantra_16k.json")
    p.add_argument("--capacity", type=int, default=16_384)
    p.add_argument("--max-capacity", type=int, default=65_536)
    p.add_argument("--merges", type=int, default=12_000)
    p.add_argument("--limit", type=int, default=150_000)
    p.add_argument("--min-pair-freq", type=int, default=3)

    p = subparsers.add_parser("merge-checkpoints", help="Merge checkpoints via weight averaging")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--output", "-o", required=True, type=str)
    p.add_argument("--weights", "-w", nargs="+", type=float, default=None)

    args = parser.parse_args()
    if args.command == "train-tokenizer":
        _train_tokenizer_main(args)
    elif args.command == "merge-checkpoints":
        _merge_checkpoints_main(args)
    else:
        parser.print_help()
        sys.exit(1)



# ═══════════════════════════════════════════════════════════
# TRAIN TOKENIZER
# ═══════════════════════════════════════════════════════════


import importlib.util
import json
from pathlib import Path


def TT_load_tokenizer_class():
    path = Path(__file__).resolve().parents[1] / "npdna" / "tokenizer.py"
    spec = importlib.util.spec_from_file_location("npdna_tokenizer_standalone", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load tokenizer module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.AtulyaTokenizer


TT_AtulyaTokenizer = TT_load_tokenizer_class()


def TT_iter_texts(path: Path, limit: int | None = None):
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


def _train_tokenizer_main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("Download/tantra_train_plus_200k.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("model/tokenizer/tokenizer_tantra_16k.json"))
    parser.add_argument("--capacity", type=int, default=16_384)
    parser.add_argument("--max-capacity", type=int, default=65_536)
    parser.add_argument("--merges", type=int, default=12_000)
    parser.add_argument("--limit", type=int, default=150_000)
    parser.add_argument("--min-pair-freq", type=int, default=3)
    args = args

    tok = TT_AtulyaTokenizer(initial_capacity=args.capacity, max_capacity=args.max_capacity)
    print(f"Training tokenizer on {args.input}")
    print(f"capacity={args.capacity}, max_capacity={args.max_capacity}, target_merges={args.merges}")
    tok.train_bpe(
        TT_iter_texts(Path(args.input), limit=args.limit),
        target_merges=args.merges,
        max_words=250_000,
        min_pair_freq=args.min_pair_freq,
    )
    # Keep embedding capacity aligned to the requested capacity unless the
    # tokenizer grew above it.
    if tok.capacity < args.capacity:
        tok._capacity = args.capacity
    tok.save(Path(args.output))
    print(f"Saved {args.output}")
    print(f"vocab_size={tok.size}, capacity={tok.capacity}, fill={tok.fill_ratio:.1%}, merges={len(tok.merges)}")


# ═══════════════════════════════════════════════════════════
# MERGE CHECKPOINTS
# ═══════════════════════════════════════════════════════════

"""NP-DNA Model Merging Utility.

Averages the weight parameters of multiple trained NP-DNA checkpoints to
create a unified/fused checkpoint (Stochastic Weight Averaging / SWA).
"""

import shutil
import torch

def _merge_checkpoints_main(args):
    parser = argparse.ArgumentParser(description="Merge multiple NP-DNA checkpoints via parameter averaging.")
    parser.add_argument("checkpoints", nargs="+", help="Paths to the checkpoint directories to merge.")
    parser.add_argument("--output", "-o", required=True, help="Path to save the merged checkpoint.")
    parser.add_argument("--weights", "-w", nargs="+", type=float, default=None,
                        help="Relative weights for each checkpoint. Defaults to uniform averaging.")

    args = args

    ckpt_paths = [Path(p) for p in args.checkpoints]
    for p in ckpt_paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint directory {p} does not exist.")
        if not (p / "model.pt").exists():
            raise FileNotFoundError(f"Checkpoint directory {p} does not contain model.pt.")
        if not (p / "metadata.json").exists():
            raise FileNotFoundError(f"Checkpoint directory {p} does not contain metadata.json.")

    weights = args.weights
    if weights:
        if len(weights) != len(ckpt_paths):
            raise ValueError("Number of weights must match the number of checkpoints.")
        # Normalize weights
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
    else:
        weights = [1.0 / len(ckpt_paths)] * len(ckpt_paths)

    print(f"Merging {len(ckpt_paths)} checkpoints:")
    for path, w in zip(ckpt_paths, weights):
        print(f"  - {path} (weight: {w:.3f})")

    # Load first model state dict
    merged_state = torch.load(ckpt_paths[0] / "model.pt", map_location="cpu", weights_only=True)

    # Scale first model state
    for k in merged_state:
        if isinstance(merged_state[k], torch.Tensor):
            merged_state[k] = merged_state[k].float() * weights[0]

    # Accumulate subsequent models
    for idx, path in enumerate(ckpt_paths[1:], start=1):
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        for k in merged_state:
            if k not in state:
                print(f"Warning: Key {k} not found in {path}. Skipping.")
                continue
            if isinstance(merged_state[k], torch.Tensor):
                merged_state[k] += state[k].float() * weights[idx]

    # Cast back to float32
    for k in merged_state:
        if isinstance(merged_state[k], torch.Tensor):
            merged_state[k] = merged_state[k].to(torch.float32)

    # Save output checkpoint
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(merged_state, out_dir / "model.pt")

    # Load first metadata and update it
    with open(ckpt_paths[0] / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["merged_checkpoints"] = [str(p) for p in ckpt_paths]
    meta["merge_weights"] = weights

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Copy tokenizer
    if (ckpt_paths[0] / "tokenizer.json").exists():
        shutil.copy(ckpt_paths[0] / "tokenizer.json", out_dir / "tokenizer.json")
    if (ckpt_paths[0] / "cortex").exists():
        # Copy cortex directory if it exists
        shutil.copytree(ckpt_paths[0] / "cortex", out_dir / "cortex", dirs_exist_ok=True)

    print(f"Successfully created merged checkpoint at: {out_dir}")

if __name__ == "__main__":
    main()
