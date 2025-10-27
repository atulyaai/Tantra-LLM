#!/usr/bin/env python3
"""
OCR-Native LLM - Main Entry Point
Clean, modular implementation
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.ocr_native_llm import OCRNativeLLM
from configs.ocr_config import ConfigManager
from models.model_manager import ModelManager


def main():
    """Main entry point"""
    print("🔤 OCR-Native LLM - Clean Version")
    print("Revolutionary approach: All weights stored in OCR format")
    print("=" * 60)
    
    # Create model
    config = ConfigManager.get_default_config()
    model = OCRNativeLLM(config)
    
    # Show model info
    info = model.get_model_info()
    print(f"Model Parameters: {info['total_parameters']:,}")
    print(f"Model Size: {info['model_size_mb']:.2f} MB")
    print(f"Layers: {info['n_layers']}, Heads: {info['n_heads']}")
    
    # Test basic functionality
    test_inputs = {
        'text': 'Hello, OCR-native world!',
        'speech': [0.1, 0.2, 0.3, 0.4, 0.5],
        'image': None
    }
    
    try:
        response = model.generate_response(test_inputs, "Test prompt")
        print(f"\nResponse: {response}")
        print("\n✅ OCR-Native LLM is working correctly!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
