#!/usr/bin/env python3
"""
Validate Tantra LLM weight files and format compatibility
"""

import sys
import torch
from pathlib import Path

# Add Training directory to path
sys.path.append(str(Path(__file__).parent / "Training"))

from weight_manager import get_weight_manager, load_model_weights
from config_manager import get_config_manager

def validate_weight_formats():
    """Validate all weight file formats"""
    print("Tantra LLM - Weight Validation")
    print("=" * 50)
    
    weight_manager = get_weight_manager()
    
    # Test different model types and formats
    test_cases = [
        ("mamba", "v1.0"),
        ("mamba_multimodal", "v1.0"),
        ("ocr_native", "v1.0")
    ]
    
    for model_type, version in test_cases:
        print(f"\nValidating {model_type} weights:")
        
        # List available weights for this model type
        weights = weight_manager.list_weights(model_type)
        print(f"  Available weights: {len(weights)}")
        
        for weight in weights:
            print(f"    - {weight.name} ({weight.version}) - {Path(weight.path).suffix}")
        
        # Test loading active weight
        print(f"\n  Loading active weight for {model_type}:")
        state_dict = load_model_weights(model_type, version)
        
        if state_dict:
            print(f"    ✓ Successfully loaded {len(state_dict)} tensors")
            
            # Show some tensor info
            for i, (key, tensor) in enumerate(list(state_dict.items())[:3]):
                print(f"      {key}: {tensor.shape} ({tensor.dtype})")
            
            # Test validation
            is_valid = weight_manager.validate_weights(model_type, version)
            print(f"    ✓ Weight validation: {'PASS' if is_valid else 'FAIL'}")
        else:
            print(f"    ✗ Failed to load weights")

def validate_format_compatibility():
    """Validate compatibility between different formats"""
    print(f"\n{'='*50}")
    print("Validating format compatibility...")
    
    weight_manager = get_weight_manager()
    
    # Test loading the same model in different formats
    model_type = "mamba"
    
    # Find weights in different formats
    weights = weight_manager.list_weights(model_type)
    format_weights = {}
    
    for weight in weights:
        if "safetensors" in weight.name:
            format_weights["safetensors"] = weight
        elif "pt" in weight.name:
            format_weights["pt"] = weight
        elif "bin" in weight.name:
            format_weights["bin"] = weight
    
    print(f"Found weights in formats: {list(format_weights.keys())}")
    
    # Load and compare
    loaded_weights = {}
    for format_name, weight_info in format_weights.items():
        print(f"\nLoading {format_name} format...")
        state_dict = weight_manager.load_weights(model_type, weight_info.version)
        if state_dict:
            loaded_weights[format_name] = state_dict
            print(f"  ✓ Loaded {len(state_dict)} tensors")
        else:
            print(f"  ✗ Failed to load {format_name}")
    
    # Compare tensor shapes and values
    if len(loaded_weights) >= 2:
        print(f"\nComparing formats...")
        formats = list(loaded_weights.keys())
        base_format = formats[0]
        base_weights = loaded_weights[base_format]
        
        for other_format in formats[1:]:
            other_weights = loaded_weights[other_format]
            print(f"\n  Comparing {base_format} vs {other_format}:")
            
            # Check if keys match
            base_keys = set(base_weights.keys())
            other_keys = set(other_weights.keys())
            
            if base_keys == other_keys:
                print(f"    ✓ Keys match")
                
                # Check tensor shapes
                shape_matches = 0
                for key in base_keys:
                    if base_weights[key].shape == other_weights[key].shape:
                        shape_matches += 1
                    else:
                        print(f"    ✗ Shape mismatch for {key}: {base_weights[key].shape} vs {other_weights[key].shape}")
                
                print(f"    ✓ Shape matches: {shape_matches}/{len(base_keys)}")
                
                # Check tensor values (sample a few)
                value_matches = 0
                sample_keys = list(base_keys)[:3]  # Check first 3 tensors
                
                for key in sample_keys:
                    if torch.allclose(base_weights[key], other_weights[key], atol=1e-6):
                        value_matches += 1
                    else:
                        print(f"    ✗ Value mismatch for {key}")
                
                print(f"    ✓ Value matches: {value_matches}/{len(sample_keys)}")
            else:
                print(f"    ✗ Key mismatch: {len(base_keys)} vs {len(other_keys)}")

def validate_file_sizes():
    """Validate file sizes for different formats"""
    print(f"\n{'='*50}")
    print("Validating file sizes...")
    
    weight_manager = get_weight_manager()
    
    for model_type in ["mamba", "mamba_multimodal", "ocr_native"]:
        print(f"\n{model_type.upper()}:")
        
        weights = weight_manager.list_weights(model_type)
        for weight in weights:
            if "v1.0" in weight.name:
                file_size_mb = Path(weight.path).stat().st_size / (1024 * 1024)
                format_name = Path(weight.path).suffix[1:]  # Remove the dot
                print(f"  {format_name.upper()}: {file_size_mb:.2f} MB")

def main():
    print("Tantra LLM Weight Validation System")
    print("=" * 50)
    
    try:
        validate_weight_formats()
        validate_format_compatibility()
        validate_file_sizes()
        
        print(f"\n{'='*50}")
        print("✓ All weight validations completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()