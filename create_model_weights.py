#!/usr/bin/env python3
"""
Create proper model weights in multiple formats (.pt, .bin, .safetensors)
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add Training directory to path
sys.path.append(str(Path(__file__).parent / "Training"))

from weight_manager import get_weight_manager, save_model_weights_multiple_formats
from config_manager import get_config_manager
from model_mamba import MambaDecoder, build_from_config

def create_mamba_weights(model_type: str = "mamba", vocab_size: int = 50000):
    """Create proper Mamba model weights"""
    print(f"Creating {model_type} model weights...")
    
    # Get configuration
    config_manager = get_config_manager()
    config = config_manager.get_config(model_type)
    
    # Create model
    model = MambaDecoder(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state,
        d_conv=config.d_conv,
        dropout=config.dropout
    )
    
    # Initialize with proper weights
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    model.apply(init_weights)
    
    # Get state dict
    state_dict = model.state_dict()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"State dict keys: {list(state_dict.keys())}")
    
    return state_dict

def create_multimodal_weights(model_type: str = "mamba_multimodal", vocab_size: int = 100000):
    """Create proper multimodal model weights"""
    print(f"Creating {model_type} model weights...")
    
    # Get configuration
    config_manager = get_config_manager()
    config = config_manager.get_config(model_type)
    
    # Create a simplified multimodal model structure
    class SimpleMultiModal(nn.Module):
        def __init__(self, vocab_size, d_model, n_layers):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.out = nn.Linear(d_model, vocab_size, bias=False)
            
        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.out(x)
    
    model = SimpleMultiModal(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers
    )
    
    # Initialize with proper weights
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    model.apply(init_weights)
    
    # Get state dict
    state_dict = model.state_dict()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"State dict keys: {list(state_dict.keys())}")
    
    return state_dict

def create_ocr_native_weights(model_type: str = "ocr_native", vocab_size: int = 50000):
    """Create proper OCR native model weights"""
    print(f"Creating {model_type} model weights...")
    
    # Get configuration
    config_manager = get_config_manager()
    config = config_manager.get_config(model_type)
    
    # Create OCR-optimized model
    class OCRNativeModel(nn.Module):
        def __init__(self, vocab_size, d_model, n_layers):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Parameter(torch.randn(1024, d_model))
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.out = nn.Linear(d_model, vocab_size, bias=False)
            
        def forward(self, x):
            x = self.embed(x)
            seq_len = x.size(1)
            if seq_len <= self.pos_embed.size(0):
                pos_enc = self.pos_embed[:seq_len].unsqueeze(0)
                x = x + pos_enc
            
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.out(x)
    
    model = OCRNativeModel(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers
    )
    
    # Initialize with proper weights
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    model.apply(init_weights)
    
    # Get state dict
    state_dict = model.state_dict()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"State dict keys: {list(state_dict.keys())}")
    
    return state_dict

def main():
    print("Creating proper model weights in multiple formats...")
    print("=" * 60)
    
    # Model configurations
    models = [
        ("mamba", 50000),
        ("mamba_multimodal", 100000),
        ("ocr_native", 50000)
    ]
    
    weight_manager = get_weight_manager()
    
    for model_type, vocab_size in models:
        print(f"\n{'='*20} {model_type.upper()} {'='*20}")
        
        try:
            # Create weights based on model type
            if model_type == "mamba":
                state_dict = create_mamba_weights(model_type, vocab_size)
            elif model_type == "mamba_multimodal":
                state_dict = create_multimodal_weights(model_type, vocab_size)
            elif model_type == "ocr_native":
                state_dict = create_ocr_native_weights(model_type, vocab_size)
            else:
                print(f"Unknown model type: {model_type}")
                continue
            
            # Save in multiple formats
            print(f"\nSaving {model_type} weights in multiple formats...")
            saved_paths = save_model_weights_multiple_formats(
                state_dict, 
                model_type, 
                name=f"{model_type}_v1.0",
                version="v1.0",
                is_active=True
            )
            
            print(f"Saved {model_type} weights:")
            for format_name, path in saved_paths.items():
                file_size = Path(path).stat().st_size / (1024 * 1024)  # MB
                print(f"  {format_name.upper()}: {path} ({file_size:.1f} MB)")
            
        except Exception as e:
            print(f"Error creating {model_type} weights: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✓ Model weight creation completed!")
    print("\nYou can now find the weights in Model/weights/ directory:")
    print("  - .safetensors files (recommended)")
    print("  - .pt files (PyTorch format)")
    print("  - .bin files (binary format)")

if __name__ == "__main__":
    main()