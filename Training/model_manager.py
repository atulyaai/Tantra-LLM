#!/usr/bin/env python3
"""
Model Management Utility for Tantra LLM
Provides command-line interface for managing models, weights, and configurations
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional

from weight_manager import get_weight_manager, WeightInfo
from config_manager import get_config_manager, ModelConfig

def list_models():
    """List all available model types"""
    config_manager = get_config_manager()
    models = config_manager.list_model_types()
    
    print("Available Model Types:")
    print("-" * 30)
    for model_type in models:
        config = config_manager.get_config(model_type)
        print(f"  {model_type}: {config.architecture} ({config.d_model}d, {config.n_layers}L)")
    print()

def list_weights(model_type: Optional[str] = None):
    """List all weights or weights for a specific model type"""
    weight_manager = get_weight_manager()
    
    if model_type:
        weights = weight_manager.list_weights(model_type)
        print(f"Weights for {model_type}:")
    else:
        weights = weight_manager.list_weights()
        print("All Weights:")
    
    print("-" * 50)
    if not weights:
        print("  No weights found")
        return
    
    for weight in weights:
        status = "ACTIVE" if weight.is_active else "inactive"
        print(f"  {weight.name} ({weight.model_type}) - {weight.version}")
        print(f"    Path: {weight.path}")
        print(f"    Size: {weight.size_mb:.1f} MB")
        print(f"    Status: {status}")
        print(f"    Created: {weight.created_at}")
        print()

def set_active_weight(name: str):
    """Set a weight as active"""
    weight_manager = get_weight_manager()
    
    if weight_manager.set_active_weight(name):
        print(f"✓ Set {name} as active weight")
    else:
        print(f"✗ Weight {name} not found")

def backup_weight(name: str):
    """Create a backup of a weight"""
    weight_manager = get_weight_manager()
    
    if weight_manager.backup_weight(name):
        print(f"✓ Created backup for {name}")
    else:
        print(f"✗ Failed to backup {name}")

def cleanup_weights(model_type: str, keep_count: int = 5):
    """Clean up old weights"""
    weight_manager = get_weight_manager()
    
    before_count = len(weight_manager.list_weights(model_type))
    weight_manager.cleanup_old_weights(model_type, keep_count)
    after_count = len(weight_manager.list_weights(model_type))
    
    cleaned = before_count - after_count
    print(f"✓ Cleaned up {cleaned} old weights for {model_type}")

def validate_weights(model_type: str, version: Optional[str] = None):
    """Validate weights for a model type"""
    weight_manager = get_weight_manager()
    
    if weight_manager.validate_weights(model_type, version):
        print(f"✓ Weights for {model_type} are valid")
    else:
        print(f"✗ Weights for {model_type} are invalid or missing")

def show_config(model_type: str):
    """Show configuration for a model type"""
    config_manager = get_config_manager()
    
    try:
        config = config_manager.get_config(model_type)
        print(f"Configuration for {model_type}:")
        print("-" * 40)
        print(f"  Architecture: {config.architecture}")
        print(f"  Model Dimension: {config.d_model}")
        print(f"  Layers: {config.n_layers}")
        print(f"  State Dimension: {config.d_state}")
        print(f"  Conv Kernel: {config.d_conv}")
        print(f"  Dropout: {config.dropout}")
        print(f"  Vocab Size: {config.vocab_size}")
        print(f"  Max Seq Length: {config.max_seq_len}")
        
        if config.extra_params:
            print(f"  Extra Parameters:")
            for key, value in config.extra_params.items():
                print(f"    {key}: {value}")
        print()
        
    except Exception as e:
        print(f"✗ Error loading config for {model_type}: {e}")

def create_model_config(model_type: str, **kwargs):
    """Create a new model configuration"""
    config_manager = get_config_manager()
    
    try:
        config = config_manager.create_model_config(model_type, **kwargs)
        print(f"✓ Created configuration for {model_type}")
        show_config(model_type)
    except Exception as e:
        print(f"✗ Error creating config for {model_type}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Tantra LLM Model Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List models
    subparsers.add_parser("list-models", help="List all available model types")
    
    # List weights
    list_weights_parser = subparsers.add_parser("list-weights", help="List weights")
    list_weights_parser.add_argument("--model-type", help="Filter by model type")
    
    # Set active weight
    set_active_parser = subparsers.add_parser("set-active", help="Set active weight")
    set_active_parser.add_argument("name", help="Weight name")
    
    # Backup weight
    backup_parser = subparsers.add_parser("backup", help="Backup a weight")
    backup_parser.add_argument("name", help="Weight name")
    
    # Cleanup weights
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old weights")
    cleanup_parser.add_argument("model_type", help="Model type")
    cleanup_parser.add_argument("--keep", type=int, default=5, help="Number of weights to keep")
    
    # Validate weights
    validate_parser = subparsers.add_parser("validate", help="Validate weights")
    validate_parser.add_argument("model_type", help="Model type")
    validate_parser.add_argument("--version", help="Specific version to validate")
    
    # Show config
    config_parser = subparsers.add_parser("config", help="Show model configuration")
    config_parser.add_argument("model_type", help="Model type")
    
    # Create config
    create_config_parser = subparsers.add_parser("create-config", help="Create model configuration")
    create_config_parser.add_argument("model_type", help="Model type")
    create_config_parser.add_argument("--d-model", type=int, help="Model dimension")
    create_config_parser.add_argument("--n-layers", type=int, help="Number of layers")
    create_config_parser.add_argument("--d-state", type=int, help="State dimension")
    create_config_parser.add_argument("--d-conv", type=int, help="Conv kernel size")
    create_config_parser.add_argument("--dropout", type=float, help="Dropout rate")
    create_config_parser.add_argument("--vocab-size", type=int, help="Vocabulary size")
    create_config_parser.add_argument("--max-seq-len", type=int, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "list-models":
            list_models()
        elif args.command == "list-weights":
            list_weights(args.model_type)
        elif args.command == "set-active":
            set_active_weight(args.name)
        elif args.command == "backup":
            backup_weight(args.name)
        elif args.command == "cleanup":
            cleanup_weights(args.model_type, args.keep)
        elif args.command == "validate":
            validate_weights(args.model_type, args.version)
        elif args.command == "config":
            show_config(args.model_type)
        elif args.command == "create-config":
            # Extract kwargs from args
            kwargs = {}
            for key in ['d_model', 'n_layers', 'd_state', 'd_conv', 'dropout', 'vocab_size', 'max_seq_len']:
                value = getattr(args, key.replace('-', '_'), None)
                if value is not None:
                    kwargs[key] = value
            
            create_model_config(args.model_type, **kwargs)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()