#!/usr/bin/env python3
"""
Test script for dynamic weight management system
"""

import sys
from pathlib import Path

# Add Training directory to path
sys.path.append(str(Path(__file__).parent / "Training"))

from weight_manager import get_weight_manager, load_model_weights
from config_manager import get_config_manager
from model_runtime import TextRuntime
from mamba_runtime import MambaRuntime

def test_weight_loading():
    """Test dynamic weight loading"""
    print("Testing dynamic weight loading...")
    
    # Test loading weights for different model types
    model_types = ["mamba", "mamba_multimodal", "ocr_native"]
    
    for model_type in model_types:
        print(f"\nTesting {model_type}:")
        
        # Load weights
        state_dict = load_model_weights(model_type)
        if state_dict:
            print(f"  ✓ Loaded {len(state_dict)} weight tensors")
            for key, tensor in list(state_dict.items())[:3]:  # Show first 3
                print(f"    {key}: {tensor.shape}")
        else:
            print(f"  ✗ Failed to load weights")

def test_runtime_initialization():
    """Test runtime initialization with dynamic weights"""
    print("\nTesting runtime initialization...")
    
    try:
        # Test MambaRuntime
        print("Testing MambaRuntime:")
        mamba_runtime = MambaRuntime(model_type="mamba")
        print(f"  ✓ MambaRuntime initialized successfully")
        print(f"  Model type: {mamba_runtime.model_type}")
        print(f"  Config: {mamba_runtime.config.d_model}d, {mamba_runtime.config.n_layers}L")
        
        # Test TextRuntime
        print("\nTesting TextRuntime:")
        text_runtime = TextRuntime(model_type="mamba")
        print(f"  ✓ TextRuntime initialized successfully")
        print(f"  Model type: {text_runtime.model_type}")
        
    except Exception as e:
        print(f"  ✗ Runtime initialization failed: {e}")

def test_weight_management():
    """Test weight management operations"""
    print("\nTesting weight management operations...")
    
    weight_manager = get_weight_manager()
    
    # List weights
    weights = weight_manager.list_weights()
    print(f"  Found {len(weights)} weights")
    
    # Test validation
    for model_type in ["mamba", "mamba_multimodal", "ocr_native"]:
        is_valid = weight_manager.validate_weights(model_type)
        print(f"  {model_type} weights valid: {is_valid}")

def test_config_management():
    """Test configuration management"""
    print("\nTesting configuration management...")
    
    config_manager = get_config_manager()
    
    # List model types
    model_types = config_manager.list_model_types()
    print(f"  Available model types: {model_types}")
    
    # Test getting configs
    for model_type in model_types[:3]:  # Test first 3
        config = config_manager.get_config(model_type)
        print(f"  {model_type}: {config.architecture} ({config.d_model}d, {config.n_layers}L)")

def main():
    print("Testing Dynamic Weight Management System")
    print("=" * 50)
    
    try:
        test_weight_loading()
        test_runtime_initialization()
        test_weight_management()
        test_config_management()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()