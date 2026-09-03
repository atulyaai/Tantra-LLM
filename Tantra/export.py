#!/usr/bin/env python3
"""
Tantra/export.py — Checkpoint Exporter & Weight Stripper
Strips optimizer momentum buffers to create lightweight, production-ready inference checkpoints.
"""

import os
import sys
import time
import argparse
import torch

from Tantra.codec import MultimodalWeightFormatter
from Tantra.config import CompressionConfig
from Tantra.utils import get_logger
log = get_logger(__name__)

def export_clean_checkpoint(input_path: str, output_path: str) -> str:
    """Strips optimizer momentum buffers and saves inference-only weights (~70% smaller)."""
    log.info(f"📦 Loading checkpoint from: {input_path}")
    raw = torch.load(input_path, map_location="cpu", weights_only=False)
    
    model_state = raw.get("model_state_dict", raw.get("model", raw))
    config_dict = raw.get("config", None)
    step = raw.get("step_count", raw.get("step", 0))

    clean_payload = {
        "model_state_dict": model_state,
        "config": config_dict,
        "step_count": step,
        "step": step,
        "best_loss": raw.get("best_loss", float('inf')),
        "total_tokens": raw.get("total_tokens", 0),
        "exported_at": time.time(),
        "format": "tantra-v1-production-clean"
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(clean_payload, output_path)
    
    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"✅ Clean inference checkpoint saved: {output_path} ({orig_size:.1f} MB -> {new_size:.1f} MB)")
    return output_path


def export_checkpoint_to_dna(input_path: str, output_path: str) -> dict:
    """Write a complete model state dict to a DNA container and verify reload.

    This is intentionally an explicit export, not a replacement for the
    normal .pt checkpoint.  It serializes every state-dict tensor, reloads
    the container, and checks keys, shapes, dtypes, and tensor values before
    reporting success.
    """
    raw = torch.load(input_path, map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw.get("model", raw))
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Checkpoint does not contain a tensor model_state_dict.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    formatter = MultimodalWeightFormatter(CompressionConfig())
    stats = formatter.format_weights(state, output_path)
    restored = formatter.parse_weights(output_path)
    if set(restored) != set(state):
        raise RuntimeError("DNA verification failed: restored tensor keys differ from the checkpoint.")
    for name, tensor in state.items():
        candidate = restored[name]
        if candidate.shape != tensor.shape or candidate.dtype != tensor.dtype or not torch.equal(candidate, tensor.cpu()):
            raise RuntimeError(f"DNA verification failed for tensor: {name}")

    result = {
        "output": output_path,
        "tensors": len(state),
        "original_bytes": stats.original_bytes,
        "container_bytes": stats.compressed_bytes,
        "compression_ratio": stats.compression_ratio,
        "sha256_match": stats.sha256_match,
    }
    log.info("✅ DNA checkpoint verified: %d tensors, %.3fx ratio, %s", result["tensors"], result["compression_ratio"], output_path)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tantra Checkpoint Exporter")
    parser.add_argument("--checkpoint", type=str, required=True, help="Input .pt checkpoint")
    parser.add_argument("--output", type=str, default="Model/Export/checkpoint_clean.pt", help="Output path")
    parser.add_argument("--dna", action="store_true", help="Export to compressed .dna format instead of .pt")
    args = parser.parse_args()

    if args.dna:
        # Auto-set output extension to .dna
        out = args.output
        if not out.endswith(".dna"):
            out = os.path.splitext(out)[0] + ".dna"
        print(f"📦 Exporting checkpoint to DNA format: {out}")
        result = export_checkpoint_to_dna(args.checkpoint, out)
        orig_mb = result["original_bytes"] / 1e6
        comp_mb = result["container_bytes"] / 1e6
        print(f"✅ DNA Export Complete!")
        print(f"   Tensors    : {result['tensors']}")
        print(f"   Original   : {orig_mb:.1f} MB")
        print(f"   Compressed : {comp_mb:.1f} MB")
        print(f"   Ratio      : {result['compression_ratio']:.3f}x")
        print(f"   SHA256 OK  : {result['sha256_match']}")
    else:
        export_clean_checkpoint(args.checkpoint, args.output)

