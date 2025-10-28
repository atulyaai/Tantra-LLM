#!/usr/bin/env python3
"""
Advanced SpikingBrain Demo
Demonstrates the full multimodal reasoning system with memory, personality, and orchestration.
"""

import os
import sys
import torch
import logging
from typing import Optional

# Add workspace to path
sys.path.insert(0, '/workspace')

from core.control.brain_orchestrator import BrainOrchestrator
from core.control.perception import Perception
from core.control.decision_engine import DecisionEngine
from core.control.response_generator import ResponseGenerator
from core.memory.advanced_memory import AdvancedMemoryManager
from core.fusion.orchestrator import FusionOrchestrator
from core.fusion.multimodal_fusion import MultimodalFusion
from personality.personality_layer import PersonalityLayer
from utils.model_loader import ModelLoader
from config import model_config
import json

# Load personality config
with open('/workspace/config/personality_config.json', 'r') as f:
    PERSONALITY_CONFIG = json.load(f)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_advanced_demo():
    """Build the advanced SpikingBrain demo system."""
    logger.info("Building advanced SpikingBrain demo...")
    
    # Load configuration
    cfg = model_config.MODEL_CONFIG
    
    # Initialize model loader
    model_loader = ModelLoader(cfg)
    
    # Load SpikingBrain model
    logger.info("Loading SpikingBrain model...")
    spikingbrain_data = model_loader.load_spikingbrain()
    spikingbrain_model = spikingbrain_data.get("model")
    tokenizer = spikingbrain_data.get("tokenizer")
    
    if spikingbrain_model is None:
        logger.warning("SpikingBrain model not loaded, using fallback mode")
    
    # Initialize components
    logger.info("Initializing system components...")
    
    # Memory system
    memory_manager = AdvancedMemoryManager(
        embedding_dim=cfg["memory"]["embedding_dim"],
        max_episodic=cfg["memory"]["max_episodic"]
    )
    
    # Personality system
    personality_layer = PersonalityLayer(PERSONALITY_CONFIG)
    
    # Fusion orchestrator
    fusion_orchestrator = FusionOrchestrator(cfg)
    
    # Multimodal fusion
    multimodal_fusion = MultimodalFusion(
        text_dim=cfg["model_dim"],
        vision_dim=cfg["vision"]["embed_dim"],
        audio_dim=cfg["audio"]["embed_dim"],
        hidden_dim=cfg["model_dim"]
    )
    
    # Core components
    perception = Perception(fusion_orchestrator, tokenizer)
    decision_engine = DecisionEngine()
    response_generator = ResponseGenerator(
        spikingbrain_model, 
        fusion_orchestrator, 
        tokenizer, 
        personality_layer
    )
    
    # Override with advanced components
    response_generator.memory_manager = memory_manager
    response_generator.multimodal_fusion = multimodal_fusion
    
    # Brain orchestrator
    brain = BrainOrchestrator(perception, decision_engine, response_generator, memory_manager)
    
    logger.info("Advanced SpikingBrain demo built successfully!")
    return brain

def run_interactive_demo():
    """Run an interactive demo session."""
    print("🧠 Advanced SpikingBrain Demo")
    print("=" * 50)
    print("This demo showcases:")
    print("• Custom SpikingBrain model architecture")
    print("• Advanced memory system (episodic + semantic)")
    print("• Multimodal fusion (text, vision, audio)")
    print("• Personality-driven responses")
    print("• Compute orchestration")
    print("• Context-aware reasoning")
    print("=" * 50)
    
    # Build the system
    brain = build_advanced_demo()
    
    # Store some initial facts
    brain.memory.store_fact("ai_capabilities", "I am an advanced AI with multimodal reasoning capabilities")
    brain.memory.store_fact("system_status", "I'm running in advanced mode with full functionality")
    
    print("\n💬 Interactive Session (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thanks for testing SpikingBrain!")
                break
            
            if not user_input:
                continue
            
            # Process with brain
            print("🧠 Thinking...")
            response = brain.step(text=user_input)
            
            # Display response
            print(f"SpikingBrain: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Demo error: {e}")

def run_test_scenarios():
    """Run predefined test scenarios."""
    print("🧪 Running Test Scenarios")
    print("=" * 50)
    
    # Build the system
    brain = build_advanced_demo()
    
    # Test scenarios
    test_cases = [
        {
            "name": "Basic Greeting",
            "input": "Hello! How are you today?",
            "expected_features": ["greeting", "personality", "memory"]
        },
        {
            "name": "Memory Test",
            "input": "What do you know about AI capabilities?",
            "expected_features": ["memory_recall", "fact_retrieval"]
        },
        {
            "name": "Complex Reasoning",
            "input": "Explain how multimodal AI systems work and what makes them different from traditional systems.",
            "expected_features": ["complex_reasoning", "detailed_explanation"]
        },
        {
            "name": "Mode Switching",
            "input": "mode: creative - Give me some creative ideas for a science fiction story",
            "expected_features": ["mode_switching", "creative_response"]
        },
        {
            "name": "Context Awareness",
            "input": "What was the last thing I asked you about?",
            "expected_features": ["context_awareness", "memory_recall"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        print("Processing...")
        
        try:
            response = brain.step(text=test_case['input'])
            print(f"Response: {response}")
            print(f"✅ Test {i} completed")
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
        
        print("-" * 30)

def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_test_scenarios()
    else:
        run_interactive_demo()

if __name__ == "__main__":
    main()