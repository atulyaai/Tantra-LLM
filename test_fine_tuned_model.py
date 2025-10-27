#!/usr/bin/env python3
"""
Test script for the fine-tuned Tantra model
"""

import sys
import os
sys.path.append('/workspace/src')

from core.tantra_llm import TantraLLM, TantraConfig
import torch
import json

def test_fine_tuned_model():
    """Test the fine-tuned model"""
    print("🧪 Testing Fine-tuned Tantra Model")
    print("=" * 50)
    
    # Load the fine-tuned model
    model_path = "/workspace/Model/weights/Tantra_fine_tuned_v1.0.pt"
    config_path = "/workspace/Model/weights/Tantra_fine_tuned_config.json"
    
    if not os.path.exists(model_path):
        print("❌ Fine-tuned model not found!")
        return
    
    if not os.path.exists(config_path):
        print("❌ Fine-tuned config not found!")
        return
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    tantra_config_dict = config_data['tantra_config']
    tantra_config = TantraConfig(**tantra_config_dict)
    
    print(f"📊 Model Configuration:")
    print(f"   • Dimensions: {tantra_config.d_model}")
    print(f"   • Layers: {tantra_config.n_layers}")
    print(f"   • Heads: {tantra_config.n_heads}")
    print(f"   • Vocab Size: {tantra_config.vocab_size}")
    print(f"   • Max Sequence Length: {tantra_config.max_seq_length}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    model = TantraLLM(tantra_config).to(device)
    
    # Load the fine-tuned weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("✅ Fine-tuned model loaded successfully!")
    
    # Get model info
    info = model.get_model_info()
    print(f"\n📈 Model Information:")
    print(f"   • Total Parameters: {info['total_parameters']:,}")
    print(f"   • Model Size: {info['model_size_mb']:.2f} MB")
    print(f"   • Fine-tuned: {config_data.get('fine_tuned', False)}")
    
    # Test with a simple input
    print(f"\n🧪 Testing model inference...")
    test_input = "Hello, this is a test of the fine-tuned model."
    
    try:
        with torch.no_grad():
            # Create proper input dictionary
            inputs = {
                'text': test_input,
                'speech': None,
                'image': None
            }
            
            # Forward pass
            output = model(inputs)
            
            print(f"✅ Model inference successful!")
            print(f"   • Input text: {test_input}")
            print(f"   • Output keys: {list(output.keys()) if isinstance(output, dict) else 'Not a dict'}")
            if isinstance(output, dict):
                for key, value in output.items():
                    if hasattr(value, 'shape'):
                        print(f"   • {key} shape: {value.shape}")
            
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        return
    
    print(f"\n🎉 Fine-tuned model test completed successfully!")
    print(f"   • Model file: {model_path}")
    print(f"   • Config file: {config_path}")
    print(f"   • Model size: {info['model_size_mb']:.2f} MB")

if __name__ == "__main__":
    test_fine_tuned_model()