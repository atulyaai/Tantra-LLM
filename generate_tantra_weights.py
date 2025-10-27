#!/usr/bin/env python3
"""
Generate Tantra LLM weights - Single multimodal model in .pt format
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add Training directory to path
sys.path.append(str(Path(__file__).parent / "Training"))

from weight_manager import get_weight_manager, save_model_weights

def create_tantra_multimodal_weights(vocab_size: int = 10000):
    """Create Tantra multimodal model weights"""
    print("Creating Tantra Multimodal Model...")
    
    class TantraMultimodalModel(nn.Module):
        def __init__(self, vocab_size, d_model=128, n_layers=4):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Parameter(torch.randn(512, d_model))
            
            # Multimodal layers with attention
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
                for _ in range(n_layers)
            ])
            
            # OCR-specific components
            self.ocr_conv = nn.Conv2d(1, d_model//4, kernel_size=3, padding=1)
            self.ocr_proj = nn.Linear(d_model//4, d_model)
            
            # Final layers
            self.norm = nn.LayerNorm(d_model)
            self.out = nn.Linear(d_model, vocab_size, bias=False)
            
        def forward(self, x, image_input=None):
            # Text processing
            x = self.embed(x)
            seq_len = x.size(1)
            if seq_len <= self.pos_embed.size(0):
                pos_enc = self.pos_embed[:seq_len].unsqueeze(0)
                x = x + pos_enc
            
            # Multimodal processing
            for layer in self.layers:
                x = layer(x)
            
            # OCR processing if image provided
            if image_input is not None:
                # Simple OCR simulation
                ocr_features = self.ocr_conv(image_input)
                ocr_features = ocr_features.mean(dim=[2, 3])  # Global average pooling
                ocr_features = self.ocr_proj(ocr_features)
                # Add OCR features to text features
                x = x + ocr_features.unsqueeze(1)
            
            x = self.norm(x)
            return self.out(x)
    
    model = TantraMultimodalModel(vocab_size=vocab_size)
    
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
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    model.apply(init_weights)
    state_dict = model.state_dict()
    
    print(f"Tantra Multimodal Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Model supports text and OCR processing")
    print(f"State dict keys: {len(state_dict)} tensors")
    
    return state_dict

def main():
    print("Tantra LLM - Weight Generation")
    print("=" * 50)
    
    try:
        # Create Tantra multimodal weights
        state_dict = create_tantra_multimodal_weights(vocab_size=10000)
        
        # Save in .pt format only with Tantra naming
        weight_manager = get_weight_manager()
        
        print(f"\nSaving Tantra weights in .pt format...")
        weight_path = weight_manager.save_weights(
            state_dict, 
            model_type="tantra_multimodal",
            name="Tantra_v1.0",
            version="v1.0",
            is_active=True,
            format="pt"
        )
        
        file_size = Path(weight_path).stat().st_size / (1024 * 1024)  # MB
        print(f"✓ Saved: {weight_path} ({file_size:.2f} MB)")
        
        print(f"\n{'='*50}")
        print("✓ Tantra LLM weights generated successfully!")
        print(f"Model: Tantra Multimodal v1.0")
        print(f"Format: PyTorch (.pt)")
        print(f"Size: {file_size:.2f} MB")
        print(f"Location: {weight_path}")
        
    except Exception as e:
        print(f"Error creating Tantra weights: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()