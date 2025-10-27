"""
Demo Script for OCR-Native LLM
Showcases the revolutionary OCR-native approach to language modeling

Copyright (c) 2024 OCR-Native LLM Contributors
Licensed under the MIT License
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import sys
import time
from typing import Dict, Any, List

# Add Training directory to path
sys.path.append('/workspace/Training')

from ocr_native_llm import OCRNativeLLM, OCRNativeConfig


def create_demo_image(text: str, size: tuple = (224, 224)) -> Image.Image:
    """Create a demo image with text"""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw text
    draw.text((10, 10), text, fill='black', font=font)
    return img


def demonstrate_ocr_weight_storage(model: OCRNativeLLM):
    """Demonstrate OCR weight storage"""
    print("\n" + "="*60)
    print("🔤 OCR WEIGHT STORAGE DEMONSTRATION")
    print("="*60)
    
    # Store weights as OCR images
    print("Converting model weights to OCR format...")
    start_time = time.time()
    ocr_weights = model.store_weights_as_ocr()
    end_time = time.time()
    
    print(f"✅ Converted {len(ocr_weights)} weight layers to OCR images in {end_time - start_time:.2f}s")
    
    # Show some examples
    print("\n📊 OCR Weight Examples:")
    for i, (layer_name, ocr_image) in enumerate(list(ocr_weights.items())[:3]):
        print(f"  {i+1}. {layer_name}: {ocr_image.size} pixels, {ocr_image.mode} mode")
        
        # Save example OCR weight
        output_path = f"./demo_output/ocr_weight_{i+1}_{layer_name.replace('.', '_')}.png"
        os.makedirs("./demo_output", exist_ok=True)
        ocr_image.save(output_path)
        print(f"     Saved to: {output_path}")
    
    return ocr_weights


def demonstrate_ocr_input_processing(model: OCRNativeLLM):
    """Demonstrate OCR input processing"""
    print("\n" + "="*60)
    print("🔄 OCR INPUT PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Text input
    print("1. Text to OCR Processing:")
    text = "Hello, I am an OCR-native LLM!"
    print(f"   Input text: '{text}'")
    
    ocr_text = model.input_processor.process_text_to_ocr(text)
    print(f"   OCR image: {ocr_text.size} pixels, {ocr_text.mode} mode")
    
    # Save OCR text
    ocr_text.save("./demo_output/ocr_text_input.png")
    print("   Saved to: ./demo_output/ocr_text_input.png")
    
    # Speech input
    print("\n2. Speech to OCR Processing:")
    speech_data = np.random.randn(1600)  # 0.1 seconds at 16kHz
    print(f"   Input speech: {speech_data.shape} samples")
    
    ocr_speech = model.input_processor.process_speech_to_ocr(speech_data)
    print(f"   OCR image: {ocr_speech.size} pixels, {ocr_speech.mode} mode")
    
    # Save OCR speech
    ocr_speech.save("./demo_output/ocr_speech_input.png")
    print("   Saved to: ./demo_output/ocr_speech_input.png")
    
    # Image input
    print("\n3. Image to OCR Processing:")
    demo_image = create_demo_image("This is a demo image for OCR processing")
    print(f"   Input image: {demo_image.size} pixels, {demo_image.mode} mode")
    
    ocr_image = model.input_processor.process_image_to_ocr(demo_image)
    print(f"   OCR image: {ocr_image.size} pixels, {ocr_image.mode} mode")
    
    # Save OCR image
    ocr_image.save("./demo_output/ocr_image_input.png")
    print("   Saved to: ./demo_output/ocr_image_input.png")


def demonstrate_memory_system(model: OCRNativeLLM):
    """Demonstrate OCR memory system"""
    print("\n" + "="*60)
    print("🧠 OCR MEMORY SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Add various types of memories
    print("Adding memories to OCR memory bank...")
    
    memories = [
        ("knowledge", "Artificial Intelligence is the simulation of human intelligence in machines.", 0.9),
        ("conversation", "User asked about AI capabilities", 0.7),
        ("pattern", "OCR patterns for better recognition", 0.8),
        ("context", "Current conversation about OCR-native LLMs", 0.6),
        ("weights", "Model weights stored as OCR images", 0.95)
    ]
    
    memory_ids = []
    for memory_type, content, importance in memories:
        memory_id = model.add_to_memory(content, memory_type, importance)
        memory_ids.append(memory_id)
        print(f"  ✅ Added {memory_type}: {memory_id}")
    
    # Retrieve memories
    print("\nRetrieving memories...")
    for query in ["AI", "OCR", "conversation"]:
        retrieved = model.memory_bank.retrieve_ocr_memory(query, top_k=3)
        print(f"  Query '{query}': Found {len(retrieved)} memories")
    
    # Show conversation history
    print(f"\nConversation history length: {len(model.get_conversation_history())}")
    
    return memory_ids


def demonstrate_response_generation(model: OCRNativeLLM):
    """Demonstrate response generation"""
    print("\n" + "="*60)
    print("💬 RESPONSE GENERATION DEMONSTRATION")
    print("="*60)
    
    # Test cases
    test_cases = [
        {
            "text": "Hello, how are you?",
            "speech": np.random.randn(1600),
            "image": create_demo_image("Hello World")
        },
        {
            "text": "What is artificial intelligence?",
            "speech": np.random.randn(1600),
            "image": create_demo_image("AI Question")
        },
        {
            "text": "Can you help me with OCR?",
            "speech": np.random.randn(1600),
            "image": create_demo_image("OCR Help")
        }
    ]
    
    for i, inputs in enumerate(test_cases, 1):
        print(f"\n{i}. Test Case {i}:")
        print(f"   Text: '{inputs['text']}'")
        print(f"   Speech: {inputs['speech'].shape} samples")
        print(f"   Image: {inputs['image'].size} pixels")
        
        # Generate response
        start_time = time.time()
        response = model.generate_response(inputs, f"Test prompt {i}")
        end_time = time.time()
        
        print(f"   Response: {response}")
        print(f"   Generation time: {end_time - start_time:.3f}s")


def demonstrate_attention_patterns(model: OCRNativeLLM):
    """Demonstrate OCR attention patterns"""
    print("\n" + "="*60)
    print("🎯 OCR ATTENTION PATTERNS DEMONSTRATION")
    print("="*60)
    
    # Create sample input
    inputs = {
        'text': "This is a test for OCR attention patterns",
        'speech': np.random.randn(1600),
        'image': create_demo_image("Attention Test")
    }
    
    # Process through model
    outputs = model(inputs)
    
    print("Processing multi-modal input through OCR-native attention...")
    print(f"  Input embeddings shape: {outputs['embeddings'].shape}")
    print(f"  Text logits shape: {outputs['text_logits'].shape}")
    print(f"  OCR output shape: {outputs['ocr_output'].shape}")
    
    # Show attention patterns
    print("\nOCR Attention Pattern Analysis:")
    for i, block in enumerate(model.blocks):
        print(f"  Block {i+1}: {block.attention.ocr_bias.shape} OCR bias parameters")
        print(f"    - Attention heads: {block.attention.n_heads}")
        print(f"    - Head dimension: {block.attention.head_dim}")


def demonstrate_performance_metrics(model: OCRNativeLLM):
    """Demonstrate performance metrics"""
    print("\n" + "="*60)
    print("📊 PERFORMANCE METRICS DEMONSTRATION")
    print("="*60)
    
    # Model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model dimension: {model.config.d_model}")
    print(f"  Number of layers: {model.config.n_layers}")
    print(f"  Number of heads: {model.config.n_heads}")
    print(f"  Max sequence length: {model.config.max_seq_length}")
    
    # Memory usage
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    print(f"\nMemory Usage:")
    print(f"  Current memory: {memory_mb:.1f} MB")
    print(f"  Memory window size: {model.config.memory_window_size:,}")
    print(f"  OCR memory bank size: {model.config.ocr_memory_bank_size}")
    
    # OCR processing stats
    print(f"\nOCR Processing:")
    print(f"  OCR image size: {model.config.ocr_image_width}x{model.config.ocr_image_height}")
    print(f"  OCR font size: {model.config.ocr_font_size}")
    print(f"  OCR precision: {model.config.ocr_precision}")
    print(f"  OCR compression ratio: {model.config.ocr_compression_ratio}")


def create_visualization(ocr_weights: Dict[str, Image.Image]):
    """Create visualization of OCR weights"""
    print("\n" + "="*60)
    print("📈 OCR WEIGHTS VISUALIZATION")
    print("="*60)
    
    # Create a grid of OCR weight images
    weight_items = list(ocr_weights.items())[:6]  # Show first 6 weights
    
    if len(weight_items) == 0:
        print("No OCR weights to visualize")
        return
    
    # Create subplot grid
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.suptitle('OCR-Native LLM Weight Visualizations', fontsize=16)
    
    for i, (layer_name, ocr_image) in enumerate(weight_items):
        row = i // cols
        col = i % cols
        
        if row < rows and col < cols:
            axes[row, col].imshow(ocr_image, cmap='gray')
            axes[row, col].set_title(f'{layer_name[:20]}...', fontsize=10)
            axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(len(weight_items), rows * cols):
        row = i // cols
        col = i % cols
        if row < rows and col < cols:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('./demo_output/ocr_weights_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved OCR weights visualization to: ./demo_output/ocr_weights_visualization.png")
    plt.show()


def main():
    """Main demo function"""
    print("🚀 OCR-Native LLM Demo")
    print("Revolutionary approach: All weights and data stored in OCR format")
    print("="*80)
    
    # Create output directory
    os.makedirs("./demo_output", exist_ok=True)
    
    # Create model
    print("Initializing OCR-Native LLM...")
    config = OCRNativeConfig(
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        vocab_size=50000,
        max_seq_length=8192,
        ocr_image_width=1024,
        ocr_image_height=1024
    )
    
    model = OCRNativeLLM(config)
    print("✅ Model initialized successfully!")
    
    # Run demonstrations
    try:
        # 1. OCR Weight Storage
        ocr_weights = demonstrate_ocr_weight_storage(model)
        
        # 2. OCR Input Processing
        demonstrate_ocr_input_processing(model)
        
        # 3. Memory System
        demonstrate_memory_system(model)
        
        # 4. Response Generation
        demonstrate_response_generation(model)
        
        # 5. Attention Patterns
        demonstrate_attention_patterns(model)
        
        # 6. Performance Metrics
        demonstrate_performance_metrics(model)
        
        # 7. Visualization
        create_visualization(ocr_weights)
        
        print("\n" + "="*80)
        print("🎉 OCR-Native LLM Demo Completed Successfully!")
        print("="*80)
        print("\nKey Features Demonstrated:")
        print("✅ OCR weight storage - All model weights stored as OCR images")
        print("✅ OCR input processing - Text, speech, and images converted to OCR")
        print("✅ OCR memory system - Long-term memory stored in OCR format")
        print("✅ Response generation - Multi-modal responses with OCR context")
        print("✅ Attention patterns - OCR-optimized attention mechanisms")
        print("✅ Performance metrics - Model architecture and memory usage")
        print("✅ Visualization - OCR weight patterns and structures")
        
        print(f"\nOutput files saved to: ./demo_output/")
        print("Check the generated OCR images to see the revolutionary approach!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()