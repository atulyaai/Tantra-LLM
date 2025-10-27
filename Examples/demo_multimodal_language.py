"""
Demo Script for Multi-Modal Language Model
Showcases text, audio, vision, reasoning, response generation, greeting, training, and domain knowledge
"""

import torch
import numpy as np
import json
import logging
from pathlib import Path
import time
from typing import Dict, Any, List

# Import the model
import sys
sys.path.append('/workspace/Training')
from multimodal_language_model import MultiModalLanguageModel, MultiModalLanguageConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_model():
    """Create a demo model with sample configuration"""
    config = MultiModalLanguageConfig(
        d_model=512,
        n_layers=6,
        vocab_size=10000,
        ocr_enabled=True,
        memory_capacity=1000,
        domain_knowledge_size=1000
    )
    
    model = MultiModalLanguageModel(config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def demo_text_processing(model):
    """Demo text processing capabilities"""
    logger.info("\n📝 Demo: Text Processing")
    
    # Create text input
    text_input = torch.randint(0, model.config.vocab_size, (1, 128))
    inputs = {"text": text_input}
    
    # Process text
    start_time = time.time()
    outputs = model.forward(inputs)
    end_time = time.time()
    
    logger.info(f"✅ Text processing completed in {end_time - start_time:.4f} seconds")
    logger.info(f"   Input shape: {text_input.shape}")
    logger.info(f"   Output shape: {outputs['text'].shape}")
    
    return outputs


def demo_audio_processing(model):
    """Demo audio processing capabilities"""
    logger.info("\n🎵 Demo: Audio Processing")
    
    # Create audio input
    audio_input = torch.randn(1, 128, model.config.audio_dim)
    inputs = {"audio": audio_input}
    
    # Process audio
    start_time = time.time()
    outputs = model.forward(inputs)
    end_time = time.time()
    
    logger.info(f"✅ Audio processing completed in {end_time - start_time:.4f} seconds")
    logger.info(f"   Input shape: {audio_input.shape}")
    logger.info(f"   Output shape: {outputs['audio'].shape}")
    
    return outputs


def demo_vision_processing(model):
    """Demo vision processing capabilities"""
    logger.info("\n🖼️ Demo: Vision Processing")
    
    # Create vision input
    vision_input = torch.randn(1, 3, 224, 224)
    inputs = {"vision": vision_input}
    
    # Process vision
    start_time = time.time()
    outputs = model.forward(inputs)
    end_time = time.time()
    
    logger.info(f"✅ Vision processing completed in {end_time - start_time:.4f} seconds")
    logger.info(f"   Input shape: {vision_input.shape}")
    logger.info(f"   Output shape: {outputs['vision'].shape}")
    
    return outputs


def demo_multimodal_processing(model):
    """Demo multi-modal processing capabilities"""
    logger.info("\n🔄 Demo: Multi-Modal Processing")
    
    # Create multi-modal input
    inputs = {
        "text": torch.randint(0, model.config.vocab_size, (1, 128)),
        "audio": torch.randn(1, 128, model.config.audio_dim),
        "vision": torch.randn(1, 3, 224, 224)
    }
    
    # Process multi-modal input
    start_time = time.time()
    outputs = model.forward(inputs, use_reasoning=True)
    end_time = time.time()
    
    logger.info(f"✅ Multi-modal processing completed in {end_time - start_time:.4f} seconds")
    logger.info(f"   Text output shape: {outputs['text'].shape}")
    logger.info(f"   Audio output shape: {outputs['audio'].shape}")
    logger.info(f"   Vision output shape: {outputs['vision'].shape}")
    
    return outputs


def demo_reasoning_capabilities(model):
    """Demo reasoning capabilities"""
    logger.info("\n🧠 Demo: Reasoning Capabilities")
    
    # Create input
    inputs = {"text": torch.randint(0, model.config.vocab_size, (1, 128))}
    
    # Test with and without reasoning
    start_time = time.time()
    outputs_with_reasoning = model.forward(inputs, use_reasoning=True)
    end_time = time.time()
    
    start_time2 = time.time()
    outputs_without_reasoning = model.forward(inputs, use_reasoning=False)
    end_time2 = time.time()
    
    # Check if outputs are different
    outputs_different = not torch.equal(
        outputs_with_reasoning["text"], 
        outputs_without_reasoning["text"]
    )
    
    logger.info(f"✅ Reasoning processing completed")
    logger.info(f"   With reasoning: {end_time - start_time:.4f} seconds")
    logger.info(f"   Without reasoning: {end_time2 - start_time2:.4f} seconds")
    logger.info(f"   Outputs different: {outputs_different}")
    
    return outputs_with_reasoning


def demo_response_generation(model):
    """Demo response generation capabilities"""
    logger.info("\n💬 Demo: Response Generation")
    
    # Create input
    inputs = {"text": torch.randint(0, model.config.vocab_size, (1, 128))}
    
    # Test different types of responses
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain quantum computing",
        "What is the meaning of life?",
        "How do neural networks learn?"
    ]
    
    for query in queries:
        start_time = time.time()
        response = model.generate_response(inputs, query)
        end_time = time.time()
        
        logger.info(f"   Query: '{query}'")
        logger.info(f"   Response: '{response}'")
        logger.info(f"   Time: {end_time - start_time:.4f} seconds")
        logger.info("")
    
    # Test greeting
    start_time = time.time()
    greeting = model.generate_response(inputs)
    end_time = time.time()
    
    logger.info(f"   Greeting: '{greeting}'")
    logger.info(f"   Time: {end_time - start_time:.4f} seconds")


def demo_domain_knowledge(model):
    """Demo domain knowledge capabilities"""
    logger.info("\n📚 Demo: Domain Knowledge")
    
    # Add domain knowledge
    knowledge_entries = [
        ("technology", "ai", "Artificial Intelligence is the simulation of human intelligence in machines."),
        ("science", "physics", "Physics is the study of matter, energy, and their interactions."),
        ("medicine", "anatomy", "Anatomy is the study of the structure of living organisms."),
        ("mathematics", "calculus", "Calculus is the mathematical study of continuous change."),
        ("philosophy", "ethics", "Ethics is the branch of philosophy that deals with moral principles.")
    ]
    
    for category, topic, information in knowledge_entries:
        model.add_domain_knowledge(category, topic, information)
        logger.info(f"   Added: {category}/{topic}")
    
    # Test knowledge retrieval
    test_queries = ["artificial intelligence", "physics", "anatomy", "calculus", "ethics"]
    
    for query in test_queries:
        knowledge = model.knowledge_base.retrieve_knowledge(query)
        logger.info(f"   Query: '{query}'")
        logger.info(f"   Retrieved: {len(knowledge)} pieces of knowledge")
        if knowledge:
            logger.info(f"   First result: '{knowledge[0][:100]}...'")
        logger.info("")


def demo_training_capabilities(model):
    """Demo training capabilities"""
    logger.info("\n🎓 Demo: Training Capabilities")
    
    # Create sample training data
    training_data = [
        {
            "text": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "text_target": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "audio": np.random.randn(64, model.config.audio_dim).tolist(),
            "audio_target": np.random.randn(64, model.config.audio_dim).tolist()
        },
        {
            "text": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "text_target": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            "audio": np.random.randn(64, model.config.audio_dim).tolist(),
            "audio_target": np.random.randn(64, model.config.audio_dim).tolist()
        }
    ]
    
    # Train model
    start_time = time.time()
    model.train_on_data(training_data, epochs=2, learning_rate=0.001)
    end_time = time.time()
    
    logger.info(f"✅ Training completed in {end_time - start_time:.4f} seconds")
    logger.info(f"   Training samples: {len(training_data)}")
    logger.info(f"   Epochs: 2")
    
    # Test that model still works after training
    inputs = {"text": torch.randint(0, model.config.vocab_size, (1, 128))}
    outputs = model.forward(inputs)
    logger.info(f"   Model still functional after training: {outputs['text'].shape}")


def demo_ocr_weight_storage(model):
    """Demo OCR weight storage capabilities"""
    logger.info("\n🔤 Demo: OCR Weight Storage")
    
    # Store weights as OCR
    start_time = time.time()
    model.store_weights_as_ocr()
    end_time = time.time()
    
    # Check stored weights
    num_weight_images = len(model.ocr_manager.weight_images)
    num_weight_metadata = len(model.ocr_manager.weight_metadata)
    
    logger.info(f"✅ OCR weight storage completed in {end_time - start_time:.4f} seconds")
    logger.info(f"   Weight images stored: {num_weight_images}")
    logger.info(f"   Weight metadata stored: {num_weight_metadata}")
    
    # Show some weight metadata
    if model.ocr_manager.weight_metadata:
        first_weight = list(model.ocr_manager.weight_metadata.keys())[0]
        metadata = model.ocr_manager.weight_metadata[first_weight]
        logger.info(f"   Example weight: {first_weight}")
        logger.info(f"   Shape: {metadata['shape']}")
        logger.info(f"   Dtype: {metadata['dtype']}")


def demo_memory_management(model):
    """Demo memory management capabilities"""
    logger.info("\n🧠 Demo: Memory Management")
    
    # Clear memory
    model.clear_memory()
    logger.info("   Memory cleared")
    
    # Add some conversation
    inputs = {"text": torch.randint(0, model.config.vocab_size, (1, 128))}
    
    queries = [
        "What is AI?",
        "How does it work?",
        "What are the applications?",
        "What are the challenges?",
        "What is the future?"
    ]
    
    for query in queries:
        response = model.generate_response(inputs, query)
        logger.info(f"   Q: {query}")
        logger.info(f"   A: {response}")
    
    # Check conversation history
    history = model.get_conversation_history()
    logger.info(f"✅ Conversation history: {len(history)} entries")
    
    # Show conversation details
    for i, entry in enumerate(history):
        logger.info(f"   Entry {i+1}:")
        logger.info(f"     Query: {entry['query']}")
        logger.info(f"     Response: {entry['response'][:50]}...")
        logger.info(f"     Timestamp: {entry['timestamp']}")


def demo_performance_benchmarks(model):
    """Demo performance benchmarks"""
    logger.info("\n⚡ Demo: Performance Benchmarks")
    
    # Benchmark different input types
    input_types = [
        ("text", {"text": torch.randint(0, model.config.vocab_size, (1, 128))}),
        ("audio", {"audio": torch.randn(1, 128, model.config.audio_dim)}),
        ("vision", {"vision": torch.randn(1, 3, 224, 224)}),
        ("multimodal", {
            "text": torch.randint(0, model.config.vocab_size, (1, 128)),
            "audio": torch.randn(1, 128, model.config.audio_dim),
            "vision": torch.randn(1, 3, 224, 224)
        })
    ]
    
    for input_type, inputs in input_types:
        times = []
        for _ in range(10):
            start_time = time.time()
            outputs = model.forward(inputs)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        logger.info(f"   {input_type.capitalize()}:")
        logger.info(f"     Average: {avg_time:.4f} seconds")
        logger.info(f"     Min: {min_time:.4f} seconds")
        logger.info(f"     Max: {max_time:.4f} seconds")


def main():
    """Main demo function"""
    logger.info("🚀 Starting Multi-Modal Language Model Demo")
    
    # Create model
    model = create_demo_model()
    
    # Run demos
    demo_text_processing(model)
    demo_audio_processing(model)
    demo_vision_processing(model)
    demo_multimodal_processing(model)
    demo_reasoning_capabilities(model)
    demo_response_generation(model)
    demo_domain_knowledge(model)
    demo_training_capabilities(model)
    demo_ocr_weight_storage(model)
    demo_memory_management(model)
    demo_performance_benchmarks(model)
    
    logger.info("\n🎉 Demo completed successfully!")
    logger.info("The Multi-Modal Language Model demonstrates:")
    logger.info("  ✅ Text, audio, and vision processing")
    logger.info("  ✅ Multi-modal fusion and reasoning")
    logger.info("  ✅ Response generation and greeting")
    logger.info("  ✅ Domain knowledge integration")
    logger.info("  ✅ Training capabilities")
    logger.info("  ✅ OCR weight storage")
    logger.info("  ✅ Memory management")
    logger.info("  ✅ Performance optimization")


if __name__ == "__main__":
    main()