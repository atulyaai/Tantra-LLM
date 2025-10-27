#!/usr/bin/env python3
"""
Test Small OCR-Native LLM
Memory-efficient version for testing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.core.ocr_native_llm import OCRNativeLLM
from src.configs.ocr_config import ConfigManager


def test_small_model():
    """Test with small model configuration"""
    print("🔤 Testing Small OCR-Native LLM")
    print("=" * 50)
    
    # Use small config
    config = ConfigManager.get_small_config()
    print(f"Model Config: {config.d_model} dim, {config.n_layers} layers, {config.n_heads} heads")
    
    # Create model
    model = OCRNativeLLM(config)
    
    # Show model info
    info = model.get_model_info()
    print(f"Parameters: {info['total_parameters']:,}")
    print(f"Model Size: {info['model_size_mb']:.2f} MB")
    
    # Test inputs
    test_inputs = {
        'text': 'Hello, OCR-native world!',
        'speech': [0.1, 0.2, 0.3, 0.4, 0.5],
        'image': None
    }
    
    try:
        # Test forward pass
        print("\n🧪 Testing forward pass...")
        outputs = model(test_inputs)
        print("✅ Forward pass successful!")
        print(f"Text logits shape: {outputs['text_logits'].shape}")
        print(f"OCR output shape: {outputs['ocr_output'].shape}")
        
        # Test response generation
        print("\n💬 Testing response generation...")
        response = model.generate_response(test_inputs, "Test prompt")
        print(f"✅ Response: {response}")
        
        # Test OCR weight storage
        print("\n🔤 Testing OCR weight storage...")
        ocr_weights = model.store_weights_as_ocr()
        print(f"✅ Generated {len(ocr_weights)} OCR weight images")
        
        # Test memory management
        print("\n🧠 Testing memory management...")
        memory_id = model.add_to_memory("Test memory", "test", 0.8)
        memories = model.memory_bank.retrieve_ocr_memory("test", top_k=3)
        print(f"✅ Memory system working: {len(memories)} memories found")
        
        print("\n🎉 All tests passed! Small model is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_small_model()