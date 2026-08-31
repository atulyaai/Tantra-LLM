"""Initialize a smaller CPU model from a larger Tantra checkpoint.

This is structured compression, not lossless conversion. It preserves matching
token embeddings and samples evenly-spaced source layers, then projects tensor
prefixes into the smaller shapes. Follow with distillation/fine-tuning.
The source checkpoint is read-only and is never overwritten.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from typing import Dict

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Tantra.cpu_profiles import build_cpu_model
from Tantra.tokenizer import ByteBPETokenizer
from Tantra.config import VocabConfig


def vocabulary(path: str, size: int) -> Dict[str, int]:
    tokenizer = ByteBPETokenizer.load(path, VocabConfig(vocab_size=size))
    return tokenizer._tokenizer.get_vocab()


def copy_overlap(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Copy the common tensor prefix, retaining target initialization elsewhere."""
    result = target.clone()
    if source.ndim != target.ndim:
        return result
    slices = tuple(slice(0, min(a, b)) for a, b in zip(source.shape, target.shape))
    value = source[slices].to(dtype=target.dtype)
    # Preserve activation scale after narrowing the input dimension.
    if source.ndim == 2 and source.shape[1] > target.shape[1]:
        value = value * math.sqrt(source.shape[1] / target.shape[1])
    result[slices] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a CPU-profile initialization from a larger Tantra checkpoint.")
    parser.add_argument("--source", required=True, help="Existing 64K checkpoint (.pt), read-only")
    parser.add_argument("--source-tokenizer", required=True, help="Tokenizer used by the source checkpoint")
    parser.add_argument("--target-tokenizer", required=True, help="Separately trained 32K tokenizer")
    parser.add_argument("--profile", choices=["micro10", "dense", "moe2"], default="micro10")
    parser.add_argument("--attention", choices=["alra", "causal"], default="alra", help="Attention implementation of the target profile")
    parser.add_argument("--vocab-size", type=int, default=None, help="Target vocabulary size; defaults to the target tokenizer's actual size")
    parser.add_argument("--output", required=True, help="New checkpoint path; must not equal --source")
    args = parser.parse_args()
    if os.path.abspath(args.source) == os.path.abspath(args.output):
        raise ValueError("Refusing to overwrite the source checkpoint.")

    source_checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    source_state = source_checkpoint["model_state_dict"]
    source_vocab = vocabulary(args.source_tokenizer, source_state["embed.weight"].shape[0])
    target_vocab = vocabulary(args.target_tokenizer, args.vocab_size or 32768)
    target_vocab_size = args.vocab_size or (max(target_vocab.values()) + 1)
    model = build_cpu_model(args.profile, attention_kind=args.attention, vocab_size=target_vocab_size)
    target_state = model.state_dict()

    # Use evenly spaced source blocks when reducing depth: 12 -> 4 becomes
    # blocks 0, 4, 7, 11, retaining early/middle/late representations.
    source_layers = sorted({int(match.group(1)) for key in source_state for match in [re.match(r"layers\.(\d+)\.", key)] if match})
    target_layers = len(model.layers)
    layer_map = {
        target_index: source_layers[round(target_index * (len(source_layers) - 1) / max(1, target_layers - 1))]
        for target_index in range(target_layers)
    }
    for target_key, target_value in target_state.items():
        source_key = target_key
        match = re.match(r"layers\.(\d+)\.(.*)", target_key)
        if match:
            source_key = f"layers.{layer_map[int(match.group(1))]}.{match.group(2)}"
        if source_key in source_state:
            target_state[target_key] = copy_overlap(source_state[source_key], target_value)

    # Copy embeddings by token string, not token ID—two independently trained
    # BPE tokenizers generally assign different IDs to the same subword.
    embedding = target_state["embed.weight"].clone()
    matched = 0
    for token, target_id in target_vocab.items():
        source_id = source_vocab.get(token)
        if source_id is not None and source_id < source_state["embed.weight"].shape[0]:
            embedding[target_id] = copy_overlap(source_state["embed.weight"][source_id], embedding[target_id])
            matched += 1
    target_state["embed.weight"] = embedding
    target_state["output_proj.weight"] = embedding  # preserve tied weights
    model.load_state_dict(target_state, strict=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        # Keep the training position for scheduling/progress. Optimizer state
        # cannot be transferred across a changed architecture, so it begins
        # fresh while the new model continues from the same step number.
        "step_count": source_checkpoint.get("step_count", 0),
        "best_loss": source_checkpoint.get("best_loss", float("inf")),
        "total_tokens": source_checkpoint.get("total_tokens", 0),
        "total_steps": source_checkpoint.get("total_steps", 0),
        "conversion": {
            "source": os.path.abspath(args.source), "profile": args.profile,
            "matched_tokens": matched, "target_vocab": len(target_vocab), "layer_map": layer_map,
            "vocab_size": target_vocab_size, "attention": args.attention,
        },
    }, args.output)
    print(f"Created {args.profile} initialization: {args.output}")
    print(f"Token embedding transfer: {matched}/{len(target_vocab)} matched token strings")
    print(f"Layer map: {layer_map}")


if __name__ == "__main__":
    main()
