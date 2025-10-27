#!/usr/bin/env python3
"""
Setup script for dynamic weight management system
Initializes the weight and configuration management system
"""

import os
import sys
from pathlib import Path

# Add Training directory to path
sys.path.append(str(Path(__file__).parent / "Training"))

from weight_manager import get_weight_manager
from config_manager import get_config_manager

def setup_directories():
    """Create necessary directories"""
    directories = [
        "Model/weights",
        "Model/backups", 
        "Model/checkpoints",
        "logs",
        "Dataset"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def initialize_weight_manager():
    """Initialize the weight manager"""
    weight_manager = get_weight_manager()
    
    # Create initial weight registry if it doesn't exist
    if not weight_manager.config_file.exists():
        weight_manager._save_registry()
        print("✓ Initialized weight registry")
    else:
        print("✓ Weight registry already exists")

def initialize_config_manager():
    """Initialize the config manager"""
    config_manager = get_config_manager()
    
    # Ensure all default configs are available
    model_types = ["mamba", "mamba_multimodal", "ocr_native"]
    
    for model_type in model_types:
        if model_type not in config_manager.configs:
            config_manager.create_model_config(model_type)
            print(f"✓ Created default config for {model_type}")
        else:
            print(f"✓ Config for {model_type} already exists")

def migrate_existing_weights():
    """Migrate existing weight files to the new system"""
    weight_manager = get_weight_manager()
    
    # Look for existing weight files
    existing_weights = [
        ("Model/tantra_weights.safetensors", "mamba"),
        ("Model/tantra_multimodal_weights.safetensors", "mamba_multimodal"),
    ]
    
    migrated_count = 0
    for weight_path, model_type in existing_weights:
        if Path(weight_path).exists():
            try:
                # Register the existing weight
                weight_name = f"{model_type}_migrated"
                weight_manager.register_weight(
                    name=weight_name,
                    path=weight_path,
                    model_type=model_type,
                    is_active=True
                )
                print(f"✓ Migrated {weight_path} as {weight_name}")
                migrated_count += 1
            except Exception as e:
                print(f"✗ Failed to migrate {weight_path}: {e}")
    
    if migrated_count == 0:
        print("ℹ No existing weights found to migrate")
    else:
        print(f"✓ Migrated {migrated_count} existing weights")

def create_sample_weights():
    """Create sample weights for testing"""
    import torch
    from weight_manager import save_model_weights
    
    print("Creating sample weights for testing...")
    
    # Create sample weights for each model type
    model_types = ["mamba", "mamba_multimodal", "ocr_native"]
    
    for model_type in model_types:
        try:
            # Create a simple state dict
            state_dict = {
                "embed.weight": torch.randn(1000, 512),
                "layers.0.weight": torch.randn(512, 512),
                "norm.weight": torch.randn(512),
                "out.weight": torch.randn(1000, 512)
            }
            
            # Save using dynamic weight management
            saved_path = save_model_weights(
                state_dict, 
                model_type, 
                name=f"{model_type}_sample",
                version="sample",
                is_active=True
            )
            print(f"✓ Created sample weights for {model_type}: {saved_path}")
            
        except Exception as e:
            print(f"✗ Failed to create sample weights for {model_type}: {e}")

def main():
    print("Setting up dynamic weight management system...")
    print("=" * 50)
    
    try:
        # Setup directories
        setup_directories()
        
        # Initialize managers
        initialize_weight_manager()
        initialize_config_manager()
        
        # Migrate existing weights
        migrate_existing_weights()
        
        # Create sample weights if no weights exist
        weight_manager = get_weight_manager()
        if not weight_manager.list_weights():
            create_sample_weights()
        
        print("\n" + "=" * 50)
        print("✓ Dynamic weight management system setup complete!")
        print("\nYou can now use the model_manager.py script to manage weights:")
        print("  python Training/model_manager.py list-models")
        print("  python Training/model_manager.py list-weights")
        print("  python Training/model_manager.py config mamba")
        
    except Exception as e:
        print(f"✗ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()