#!/usr/bin/env python
"""
Tantra Main Entry Point
========================
Command-line interface for Tantra Dense-MoA system.

Usage:
    python -m Tantra --mode train --taxonomy Tantra/config/taxonomy.yaml --base-model Model/best.pt
    python -m Tantra --mode inference --model Tantra/checkpoints/tantra_model.pt
    python -m Tantra --mode validate --taxonomy Tantra/config/taxonomy.yaml
    python -m Tantra --mode export --model Tantra/checkpoints/tantra_model.pt --output Tantra_int8.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import time
import logging
from pathlib import Path

import torch

# Add parent directory to path for Tantra imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tantra.config import (
    load_taxonomy,
    validate_taxonomy,
    print_taxonomy_summary,
    calculate_totals,
)
from Tantra.model.tantra_model import build_tantra_model, export_Tantra_int8
from Tantra.training.tantra_trainer import run_Tantra_training
from Tantra.data.expert_dataset import ExpertDatasetBuilder, validate_expert_datasets
from Tantra.inference.inference_engine import load_Tantra_for_inference, export_for_inference
from Tantra.utils.health_monitor import ExpertHealthMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
)
log = logging.getLogger("Tantra.main")


def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tantra Dense-MoA: CPU-optimized Mixture of Adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--mode", required=True, choices=[
        "train", "inference", "validate", "export", "benchmark", "health",
    ], help="Operation mode")
    
    # Common
    parser.add_argument("--taxonomy", default="Tantra/config/taxonomy.yaml", help="Taxonomy YAML path")
    parser.add_argument("--base-model", default="Model/best.pt", help="Base model checkpoint")
    parser.add_argument("--model", help="Tantra model checkpoint (for inference/export)")
    parser.add_argument("--data-dir", default="Datasets/experts", help="Expert datasets directory")
    parser.add_argument("--output", default="Tantra/checkpoints", help="Output directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"], help="Training dtype")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Training
    parser.add_argument("--expert", help="Train single expert (for debugging)")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per expert")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per expert")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adapter learning rate")
    parser.add_argument("--router-lr", type=float, default=5e-4, help="Router learning rate")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--max-experts", type=int, help="Limit number of experts to train")
    parser.add_argument("--resume-from", help="Resume from expert name")
    
    # Inference
    parser.add_argument("--quantization", default="int8", choices=["int8", "int4", "fp16", "fp32"])
    parser.add_argument("--prompt", help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--stream", action="store_true", help="Stream output")
    parser.add_argument("--interpret", action="store_true", help="Show expert attribution")
    
    # Export
    parser.add_argument("--export-format", default="int8", choices=["int8", "int4", "fp16"])
    
    return parser


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cmd_validate(args) -> int:
    """Validate taxonomy and datasets."""
    log.info(f"Validating taxonomy: {args.taxonomy}")
    taxonomy = load_taxonomy(args.taxonomy)
    
    errors = validate_taxonomy(taxonomy)
    if errors:
        log.error("Taxonomy validation failed:")
        for e in errors:
            log.error(f"  - {e}")
        return 1
    
    print_taxonomy_summary(taxonomy)
    
    # Validate datasets
    log.info(f"Validating datasets in: {args.data_dir}")
    results = validate_expert_datasets(args.data_dir, taxonomy)
    
    print(json.dumps(results, indent=2))
    
    if results["experts_missing"]:
        log.warning(f"Missing datasets for {len(results['experts_missing'])} experts")
    if results["experts_low_samples"]:
        log.warning(f"Low samples for {len(results['experts_low_samples'])} experts")
    
    return 0


def cmd_train(args) -> int:
    """Run Tantra sequential training."""
    set_seed(args.seed)
    
    log.info("=" * 60)
    log.info("Tantra SEQUENTIAL TRAINING")
    log.info("=" * 60)
    
    # Load taxonomy
    taxonomy = load_taxonomy(args.taxonomy)
    errors = validate_taxonomy(taxonomy)
    if errors:
        for e in errors:
            log.error(e)
        return 1
    
    print_taxonomy_summary(taxonomy)
    
    # Build model
    log.info(f"Loading base model: {args.base_model}")
    model = build_tantra_model(args.taxonomy, args.base_model, args.device)
    
    # Build datasets
    log.info(f"Building datasets from: {args.data_dir}")
    dataset_builder = ExpertDatasetBuilder(args.data_dir, taxonomy)
    
    # Run training
    if args.expert:
        # Single expert mode
        log.info(f"Training single expert: {args.expert}")
        # Would need to implement single expert training
        log.error("Single expert training not yet implemented via CLI")
        return 1
    else:
        # Full sequential training
        model = run_Tantra_training(
            args.taxonomy,
            args.base_model,
            args.data_dir,
            args.output,
            args.device,
        )
    
    log.info("Training complete!")
    return 0


def cmd_inference(args) -> int:
    """Run inference."""
    if not args.model:
        log.error("--model required for inference")
        return 1
    
    log.info(f"Loading model: {args.model}")
    engine = load_Tantra_for_inference(
        args.model,
        quantization=args.quantization,
    )
    
    if args.prompt:
        # Single generation
        log.info(f"Generating for prompt: {args.prompt[:60]}...")
        
        if args.interpret:
            analysis = engine.get_interpretability(args.prompt)
            print(json.dumps(analysis, indent=2))
        
        response = engine.chat(
            [{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stream=args.stream,
        )
        
        if args.stream:
            for chunk in response:
                print(chunk, end="", flush=True)
            print()
        else:
            print(response)
    else:
        # Interactive chat
        print("Tantra Chat (type 'quit' to exit)")
        print("-" * 40)
        messages = []
        while True:
            try:
                user = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user.lower() in ("quit", "exit"):
                break
            if not user:
                continue
            
            messages.append({"role": "user", "content": user})
            response = engine.chat(
                messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            messages.append({"role": "assistant", "content": response})
            print(f"Tantra: {response}")
    
    return 0


def cmd_export(args) -> int:
    """Export model for inference."""
    if not args.model:
        log.error("--model required for export")
        return 1
    
    log.info(f"Loading model: {args.model}")
    checkpoint = torch.load(args.model, map_location="cpu")
    
    from Tantra.model.tantra_model import TantraModel
    model = TantraModel(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    
    output = args.output or args.model.replace(".pt", f"_{args.export_format}.pt")
    log.info(f"Exporting to: {output} ({args.export_format})")
    
    export_for_inference(model, output, args.export_format)
    
    log.info("Export complete!")
    return 0


def cmd_benchmark(args) -> int:
    """Benchmark inference speed."""
    if not args.model:
        log.error("--model required for benchmark")
        return 1
    
    engine = load_Tantra_for_inference(args.model, quantization=args.quantization)
    
    results = engine.benchmark(seq_len=args.seq_len, num_runs=10)
    print(json.dumps(results, indent=2))
    
    return 0


def cmd_health(args) -> int:
    """Check expert health from training logs."""
    # Would read from training progress
    log.info("Health check not yet implemented via CLI")
    return 0


def main() -> int:
    parser = setup_args()
    args = parser.parse_args()
    
    # Dispatch
    commands = {
        "validate": cmd_validate,
        "train": cmd_train,
        "inference": cmd_inference,
        "export": cmd_export,
        "benchmark": cmd_benchmark,
        "health": cmd_health,
    }
    
    if args.mode not in commands:
        log.error(f"Unknown mode: {args.mode}")
        return 1
    
    try:
        return commands[args.mode](args)
    except Exception as e:
        log.exception(f"Error in {args.mode}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())