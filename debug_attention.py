#!/usr/bin/env python3
"""
Debug attention mechanism
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.core.ocr_native_llm import OCRNativeLLM
from src.configs.ocr_config import ConfigManager
import torch

def debug_attention():
    """Debug attention mechanism"""
    print("🔍 Debugging Attention Mechanism")
    print("=" * 50)
    
    # Use small config
    config = ConfigManager.get_small_config()
    model = OCRNativeLLM(config)
    
    # Test inputs
    test_inputs = {
        'text': 'Hello, OCR-native world!',
        'speech': [0.1, 0.2, 0.3, 0.4, 0.5],
        'image': None
    }
    
    # Process inputs to OCR format
    ocr_inputs = model._process_inputs_to_ocr(test_inputs)
    print(f"OCR inputs: {len(ocr_inputs)} images")
    
    # Convert OCR images to embeddings
    embeddings = model._ocr_to_embeddings(ocr_inputs)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test attention mechanism directly
    attention = model.blocks[0].attention
    print(f"Attention d_model: {attention.d_model}")
    print(f"Attention n_heads: {attention.n_heads}")
    print(f"Attention head_dim: {attention.head_dim}")
    
    # Test forward pass step by step
    x = embeddings
    print(f"Input x shape: {x.shape}")
    
    # Project to Q, K, V
    q = attention.q_proj(x)
    k = attention.k_proj(x)
    v = attention.v_proj(x)
    
    print(f"Q shape after projection: {q.shape}")
    print(f"K shape after projection: {k.shape}")
    print(f"V shape after projection: {v.shape}")
    
    batch_size, seq_len, _ = x.shape
    q = q.view(batch_size, seq_len, attention.n_heads, attention.head_dim)
    k = k.view(batch_size, seq_len, attention.n_heads, attention.head_dim)
    v = v.view(batch_size, seq_len, attention.n_heads, attention.head_dim)
    
    print(f"Q shape after view: {q.shape}")
    print(f"K shape after view: {k.shape}")
    print(f"V shape after view: {v.shape}")
    
    # Apply OCR bias
    ocr_bias = attention.ocr_bias.view(1, 1, 1, attention.d_model)
    ocr_bias = ocr_bias.view(1, 1, 1, attention.n_heads, attention.head_dim)
    print(f"OCR bias shape: {ocr_bias.shape}")
    
    q = q + ocr_bias
    print(f"Q shape after bias: {q.shape}")
    
    # Transpose for attention
    q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    print(f"Q shape after transpose: {q.shape}")
    print(f"K shape after transpose: {k.shape}")
    print(f"V shape after transpose: {v.shape}")
    
    # Compute attention scores
    k_t = k.transpose(-2, -1)
    print(f"K transpose shape: {k_t.shape}")
    
    try:
        scores = torch.matmul(q, k_t) / (attention.head_dim ** 0.5)
        print(f"Scores shape: {scores.shape}")
        print("✅ Attention computation successful!")
    except Exception as e:
        print(f"❌ Attention computation failed: {e}")
        print(f"Q shape: {q.shape}")
        print(f"K transpose shape: {k_t.shape}")

if __name__ == "__main__":
    debug_attention()