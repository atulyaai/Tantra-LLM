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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tantra Checkpoint Exporter")
    parser.add_argument("--checkpoint", type=str, required=True, help="Input .pt checkpoint")
    parser.add_argument("--output", type=str, default="Model/Export/checkpoint_clean.pt", help="Output path")
    args = parser.parse_args()

    export_clean_checkpoint(args.checkpoint, args.output)
