"""
Tantra v1.0 Demo - Clean Version
Demonstrates the revolutionary OCR-native approach
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.tantra_llm import TantraLLM, TantraConfig
from src.models.model_manager import TantraModelManager


def demonstrate_ocr_weight_storage(model):
    """Demonstrate OCR weight storage"""
    print("\n" + "=" * 60)
    print("🔤 OCR WEIGHT STORAGE DEMONSTRATION")
    print("=" * 60)
    
    print("Converting model weights to OCR format...")
    start_time = time.time()
    
    ocr_weights = model.store_weights_as_ocr()
    
    end_time = time.time()
    print(f"✅ Converted {len(ocr_weights)} weight layers to OCR images in {end_time - start_time:.2f}s")
    
    # Show examples
    print(f"\n📊 OCR Weight Examples:")
    for i, weight_img in enumerate(ocr_weights[:3]):
        layer_name = f"weight_layer_{i+1}"
        print(f"  {i+1}. {layer_name}: {weight_img.size} pixels, {weight_img.mode} mode")
        
        # Save example
        os.makedirs("demo_output", exist_ok=True)
        weight_img.save(f"demo_output/ocr_weight_{i+1}_{layer_name}.png")
        print(f"     Saved to: ./demo_output/ocr_weight_{i+1}_{layer_name}.png")
    
    return ocr_weights


def demonstrate_ocr_input_processing(model):
    """Demonstrate OCR input processing"""
    print("\n" + "=" * 60)
    print("🔄 OCR INPUT PROCESSING DEMONSTRATION")
    print("=" * 60)
    
    # Test inputs
    test_text = "Hello, I am an OCR-native LLM!"
    test_speech = np.random.randn(1600)
    test_image = Image.new('RGB', (224, 224), color='white')
    
    print("1. Text to OCR Processing:")
    print(f"   Input text: '{test_text}'")
    text_ocr = model.input_processor.process_text_to_ocr(test_text)
    print(f"   OCR image: {text_ocr.size} pixels, {text_ocr.mode} mode")
    text_ocr.save("demo_output/ocr_text_input.png")
    print(f"   Saved to: ./demo_output/ocr_text_input.png")
    
    print("\n2. Speech to OCR Processing:")
    print(f"   Input speech: {test_speech.shape} samples")
    speech_ocr = model.input_processor.process_speech_to_ocr(test_speech)
    print(f"   OCR image: {speech_ocr.size} pixels, {speech_ocr.mode} mode")
    speech_ocr.save("demo_output/ocr_speech_input.png")
    print(f"   Saved to: ./demo_output/ocr_speech_input.png")
    
    print("\n3. Image to OCR Processing:")
    print(f"   Input image: {test_image.size} pixels, {test_image.mode} mode")
    image_ocr = model.input_processor.process_image_to_ocr(test_image)
    print(f"   OCR image: {image_ocr.size} pixels, {image_ocr.mode} mode")
    image_ocr.save("demo_output/ocr_image_input.png")
    print(f"   Saved to: ./demo_output/ocr_image_input.png")


def demonstrate_ocr_memory_system(model):
    """Demonstrate OCR memory system"""
    print("\n" + "=" * 60)
    print("🧠 OCR MEMORY SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    print("Adding memories to OCR memory bank...")
    
    # Add various types of memories
    memory_types = [
        ("AI is artificial intelligence", "knowledge", 0.9),
        ("User asked about OCR processing", "conversation", 0.7),
        ("Pattern: text -> OCR -> processing", "pattern", 0.8),
        ("Context: working on language model", "context", 0.6),
        ("Model weights stored as images", "weights", 0.9)
    ]
    
    for content, mem_type, importance in memory_types:
        memory_id = model.add_to_memory(content, mem_type, importance)
        print(f"  ✅ Added {mem_type}: {memory_id}")
    
    print("\nRetrieving memories...")
    
    # Test retrieval
    queries = ["AI", "OCR", "conversation", "pattern"]
    for query in queries:
        memories = model.memory_bank.retrieve_memory(query, top_k=3)
        print(f"  Query '{query}': Found {len(memories)} memories")
    
    # Get conversation history
    history = model.get_conversation_history()
    print(f"\nConversation history length: {len(history)}")


def demonstrate_response_generation(model):
    """Demonstrate response generation"""
    print("\n" + "=" * 60)
    print("💬 RESPONSE GENERATION DEMONSTRATION")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            'text': 'Hello, how are you?',
            'speech': np.random.randn(1600),
            'image': Image.new('RGB', (224, 224), color='white')
        },
        {
            'text': 'What is artificial intelligence?',
            'speech': np.random.randn(2000),
            'image': Image.new('RGB', (128, 128), color='blue')
        },
        {
            'text': 'Explain OCR processing',
            'speech': np.random.randn(1200),
            'image': Image.new('RGB', (256, 256), color='green')
        }
    ]
    
    for i, inputs in enumerate(test_cases, 1):
        print(f"\n{i}. Test Case {i}:")
        print(f"   Text: '{inputs['text']}'")
        print(f"   Speech: {inputs['speech'].shape} samples")
        print(f"   Image: {inputs['image'].size} pixels")
        
        try:
            response = model.generate_response(inputs, f"Test prompt {i}")
            print(f"   Response: {response}")
        except Exception as e:
            print(f"   ❌ Error: {e}")


def demonstrate_model_info(model):
    """Demonstrate model information"""
    print("\n" + "=" * 60)
    print("📊 MODEL INFORMATION")
    print("=" * 60)
    
    info = model.get_model_info()
    
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"Model Size: {info['model_size_mb']:.2f} MB")
    print(f"Model Dimension: {info['d_model']}")
    print(f"Number of Layers: {info['n_layers']}")
    print(f"Number of Heads: {info['n_heads']}")
    print(f"Vocabulary Size: {info['vocab_size']:,}")
    print(f"Max Sequence Length: {info['max_seq_length']:,}")


def main():
    """Main demo function"""
    print("🚀 Tantra v1.0 Demo - Clean Version")
    print("Revolutionary approach: All weights and data stored in OCR format")
    print("=" * 80)
    
    # Create output directory
    os.makedirs("demo_output", exist_ok=True)
    
    try:
        # Initialize model with small configuration for demo
        print("Initializing Tantra v1.0...")
        config = TantraConfig()  # Default config for demo
        model = TantraLLM(config)
        print("✅ Model initialized successfully!")
        
        # Demonstrate features
        demonstrate_model_info(model)
        demonstrate_ocr_weight_storage(model)
        demonstrate_ocr_input_processing(model)
        demonstrate_ocr_memory_system(model)
        demonstrate_response_generation(model)
        
        print("\n" + "=" * 80)
        print("🎉 Demo completed successfully!")
        print("Check the 'demo_output' directory for generated OCR images")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()