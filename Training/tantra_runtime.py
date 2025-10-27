#!/usr/bin/env python3
"""
Tantra LLM Runtime - Single multimodal model with .pt weights
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from pathlib import Path

from .weight_manager import get_weight_manager, load_model_weights
from .config_manager import get_config_manager

class TantraRuntime:
    """Tantra LLM Runtime for multimodal processing"""
    
    def __init__(self, device: str = "cpu", version: str = None):
        self.device = device
        self.model_type = "tantra_multimodal"
        
        # Get configuration
        config_manager = get_config_manager()
        weight_manager = get_weight_manager()
        
        self.config = config_manager.get_config(self.model_type)
        self.paths = config_manager.get_paths(self.model_type)
        
        # Load tokenizer
        tokenizer_path = self.paths['tokenizer']
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"Loaded tokenizer from: {tokenizer_path}")
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {tokenizer_path}: {e}")
            # Create a simple tokenizer
            self.tokenizer = self._create_simple_tokenizer()
        
        # Load model weights
        state_dict = load_model_weights(self.model_type, version)
        if state_dict is None:
            print("Warning: Could not load weights, initializing with random weights")
            self.model = self._create_model()
        else:
            self.model = self._create_model()
            try:
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded Tantra weights successfully")
            except Exception as e:
                print(f"Warning: Could not load weights: {e}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def _create_simple_tokenizer(self):
        """Create a simple tokenizer for basic functionality"""
        class SimpleTokenizer:
            def __init__(self, vocab_size=10000):
                self.vocab_size = vocab_size
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.bos_token_id = 2
                self.unk_token_id = 3
            
            def encode(self, text, return_tensors=None):
                # Simple character-level encoding
                tokens = [ord(c) % self.vocab_size for c in text[:100]]  # Limit length
                if return_tensors == "pt":
                    return torch.tensor([tokens])
                return tokens
            
            def decode(self, tokens):
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
                return ''.join([chr(t % 256) for t in tokens if t > 0])
        
        return SimpleTokenizer(self.config.get('vocab_size', 10000))
    
    def _create_model(self):
        """Create the Tantra multimodal model"""
        class TantraMultimodalModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                vocab_size = config.get('vocab_size', 10000)
                d_model = config.get('d_model', 128)
                n_layers = config.get('n_layers', 4)
                
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
        
        return TantraMultimodalModel(self.config)
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8) -> str:
        """Generate text from prompt"""
        try:
            # Encode input
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Simple sampling
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Continue generation
                for _ in range(max_length - 1):
                    outputs = self.model(input_ids)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                    # Stop if EOS token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            # Decode result
            generated_text = self.tokenizer.decode(input_ids[0])
            return generated_text
            
        except Exception as e:
            print(f"Error generating text: {e}")
            return prompt  # Return original prompt on error
    
    def process_ocr(self, image_tensor: torch.Tensor, text_prompt: str = "") -> str:
        """Process OCR with image input"""
        try:
            # Encode text prompt
            if text_prompt:
                input_ids = self.tokenizer.encode(text_prompt, return_tensors="pt").to(self.device)
            else:
                input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)
            
            # Process with image
            with torch.no_grad():
                outputs = self.model(input_ids, image_input=image_tensor)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                result = self.tokenizer.decode(predictions[0])
                
            return result
            
        except Exception as e:
            print(f"Error processing OCR: {e}")
            return text_prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_type": self.model_type,
            "device": self.device,
            "config": self.config,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "supports_ocr": self.config.get('supports_ocr', False),
            "supports_text": self.config.get('supports_text', True)
        }

# Convenience function
def create_tantra_runtime(device: str = "cpu", version: str = None) -> TantraRuntime:
    """Create a Tantra runtime instance"""
    return TantraRuntime(device=device, version=version)