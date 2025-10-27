#!/usr/bin/env python3
"""
Create smaller sample weights suitable for GitHub
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add Training directory to path
sys.path.append(str(Path(__file__).parent / "Training"))

from weight_manager import get_weight_manager, save_model_weights_multiple_formats

def create_small_mamba_weights(model_type: str = "mamba", vocab_size: int = 1000):
    """Create smaller Mamba model weights for GitHub"""
    print(f"Creating small {model_type} model weights...")
    
    # Create a much smaller model
    class SmallMamba(nn.Module):
        def __init__(self, vocab_size, d_model=64, n_layers=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
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
    
    model = SmallMamba(vocab_size=vocab_size)
    
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
    
    print(f"Small model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"State dict keys: {list(state_dict.keys())}")
    
    return state_dict

def create_small_multimodal_weights(model_type: str = "mamba_multimodal", vocab_size: int = 1000):
    """Create smaller multimodal model weights for GitHub"""
    print(f"Creating small {model_type} model weights...")
    
    # Create a much smaller multimodal model
    class SmallMultiModal(nn.Module):
        def __init__(self, vocab_size, d_model=64, n_layers=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
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
    
    model = SmallMultiModal(vocab_size=vocab_size)
    
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
    
    print(f"Small model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"State dict keys: {list(state_dict.keys())}")
    
    return state_dict

def create_small_ocr_weights(model_type: str = "ocr_native", vocab_size: int = 1000):
    """Create smaller OCR model weights for GitHub"""
    print(f"Creating small {model_type} model weights...")
    
    # Create a much smaller OCR model
    class SmallOCRNative(nn.Module):
        def __init__(self, vocab_size, d_model=64, n_layers=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Parameter(torch.randn(128, d_model))
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
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
    
    model = SmallOCRNative(vocab_size=vocab_size)
    
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
    
    print(f"Small model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"State dict keys: {list(state_dict.keys())}")
    
    return state_dict

def main():
    print("Creating small sample weights for GitHub...")
    print("=" * 60)
    
    # Model configurations (smaller for GitHub)
    models = [
        ("mamba", 1000),
        ("mamba_multimodal", 1000),
        ("ocr_native", 1000)
    ]
    
    weight_manager = get_weight_manager()
    
    for model_type, vocab_size in models:
        print(f"\n{'='*20} {model_type.upper()} {'='*20}")
        
        try:
            # Create small weights based on model type
            if model_type == "mamba":
                state_dict = create_small_mamba_weights(model_type, vocab_size)
            elif model_type == "mamba_multimodal":
                state_dict = create_small_multimodal_weights(model_type, vocab_size)
            elif model_type == "ocr_native":
                state_dict = create_small_ocr_weights(model_type, vocab_size)
            else:
                print(f"Unknown model type: {model_type}")
                continue
            
            # Save in multiple formats
            print(f"\nSaving {model_type} weights in multiple formats...")
            saved_paths = save_model_weights_multiple_formats(
                state_dict, 
                model_type, 
                name=f"{model_type}_sample_small",
                version="sample_small",
                is_active=False  # Don't make these active
            )
            
            print(f"Saved {model_type} weights:")
            for format_name, path in saved_paths.items():
                file_size = Path(path).stat().st_size / (1024 * 1024)  # MB
                print(f"  {format_name.upper()}: {path} ({file_size:.2f} MB)")
            
        except Exception as e:
            print(f"Error creating {model_type} weights: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✓ Small sample weight creation completed!")
    print("\nThese weights are suitable for GitHub (small file sizes)")
    print("The large weights can be downloaded separately or hosted elsewhere.")

if __name__ == "__main__":
    main()