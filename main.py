#!/usr/bin/env python3
"""
Tantra v1.0 - Main Entry Point
OCR-Native LLM - Clean, focused implementation
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.tantra_llm import TantraLLM, TantraConfig


def main():
    """Main entry point for Tantra v1.0"""
    print("🔤 Tantra v1.0 - OCR-Native LLM")
    print("Revolutionary approach: All weights stored in OCR format")
    print("=" * 60)
    
    # Create Tantra model
    config = TantraConfig()
    model = TantraLLM(config)
    
    # Show model info
    info = model.get_model_info()
    print(f"Model: {info['name']}")
    print(f"Parameters: {info['total_parameters']:,}")
    print(f"Size: {info['model_size_mb']:.2f} MB")
    print(f"Layers: {info['n_layers']}, Heads: {info['n_heads']}")
    print(f"Type: {info['model_type']}")
    
    # Test basic functionality
    test_inputs = {
        'text': 'Hello, Tantra!',
        'speech': [0.1, 0.2, 0.3, 0.4, 0.5],
        'image': None
    }
    
    try:
        response = model.generate_response(test_inputs, "Hello, how are you?")
        print(f"\n🔤 Tantra Response:")
        print(response)
        print("\n✅ Tantra v1.0 is working correctly!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
