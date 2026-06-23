#!/usr/bin/env python3
"""NP-DNA Model Merging Utility.

Averages the weight parameters of multiple trained NP-DNA checkpoints to
create a unified/fused checkpoint (Stochastic Weight Averaging / SWA).
"""

import argparse
import json
import shutil
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser(description="Merge multiple NP-DNA checkpoints via parameter averaging.")
    parser.add_argument("checkpoints", nargs="+", help="Paths to the checkpoint directories to merge.")
    parser.add_argument("--output", "-o", required=True, help="Path to save the merged checkpoint.")
    parser.add_argument("--weights", "-w", nargs="+", type=float, default=None,
                        help="Relative weights for each checkpoint. Defaults to uniform averaging.")

    args = parser.parse_args()

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
