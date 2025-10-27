"""
Comprehensive Test Suite for Multi-Modal Language Model
Tests text, audio, vision, reasoning, response generation, greeting, training, and domain knowledge
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple
import unittest
from unittest.mock import Mock, patch

# Import the model
import sys
sys.path.append('/workspace/Training')
from multimodal_language_model import (
    MultiModalLanguageModel, 
    MultiModalLanguageConfig,
    DomainKnowledgeBase,
    ReasoningEngine,
    ResponseGenerator
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMultiModalLanguageModel(unittest.TestCase):
    """Test cases for Multi-Modal Language Model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MultiModalLanguageConfig(
            d_model=512,
            n_layers=6,
            vocab_size=10000,
            ocr_enabled=True,
            memory_capacity=1000,
            domain_knowledge_size=1000
        )
        self.model = MultiModalLanguageModel(self.config)
        self.device = torch.device('cpu')
        self.model.to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, MultiModalLanguageModel)
        self.assertEqual(self.model.config.d_model, 512)
        self.assertEqual(self.model.config.n_layers, 6)
        self.assertTrue(self.model.config.ocr_enabled)
    
    def test_text_processing(self):
        """Test text processing capabilities"""
        # Test text input
        text_input = torch.randint(0, self.config.vocab_size, (2, 128))
        inputs = {"text": text_input}
        
        # Forward pass
        outputs = self.model.forward(inputs)
        
        # Check output shape
        self.assertIn("text", outputs)
        self.assertEqual(outputs["text"].shape, (2, 128, self.config.vocab_size))
    
    def test_audio_processing(self):
        """Test audio processing capabilities"""
        # Test audio input
        audio_input = torch.randn(2, 128, self.config.audio_dim)
        inputs = {"audio": audio_input}
        
        # Forward pass
        outputs = self.model.forward(inputs)
        
        # Check output shape
        self.assertIn("audio", outputs)
        self.assertEqual(outputs["audio"].shape, (2, 128, self.config.audio_dim))
    
    def test_vision_processing(self):
        """Test vision processing capabilities"""
        # Test vision input
        vision_input = torch.randn(2, 3, 224, 224)
        inputs = {"vision": vision_input}
        
        # Forward pass
        outputs = self.model.forward(inputs)
        
        # Check output shape
        self.assertIn("vision", outputs)
        self.assertEqual(outputs["vision"].shape, (2, 1, self.config.vision_dim))
    
    def test_multimodal_processing(self):
        """Test multi-modal processing"""
        # Test combined inputs
        inputs = {
            "text": torch.randint(0, self.config.vocab_size, (2, 128)),
            "audio": torch.randn(2, 128, self.config.audio_dim),
            "vision": torch.randn(2, 3, 224, 224)
        }
        
        # Forward pass
        outputs = self.model.forward(inputs)
        
        # Check all outputs are present
        self.assertIn("text", outputs)
        self.assertIn("audio", outputs)
        self.assertIn("vision", outputs)
    
    def test_reasoning_capabilities(self):
        """Test reasoning capabilities"""
        # Test with reasoning enabled
        inputs = {"text": torch.randint(0, self.config.vocab_size, (1, 128))}
        
        # Forward pass with reasoning
        outputs_with_reasoning = self.model.forward(inputs, use_reasoning=True)
        
        # Forward pass without reasoning
        outputs_without_reasoning = self.model.forward(inputs, use_reasoning=False)
        
        # Check that outputs are different (reasoning should affect output)
        self.assertFalse(torch.equal(
            outputs_with_reasoning["text"], 
            outputs_without_reasoning["text"]
        ))
    
    def test_response_generation(self):
        """Test response generation capabilities"""
        inputs = {"text": torch.randint(0, self.config.vocab_size, (1, 128))}
        
        # Test response generation with query
        response = self.model.generate_response(inputs, "What is artificial intelligence?")
        
        # Check response is a string
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Test greeting (no query)
        greeting = self.model.generate_response(inputs)
        
        # Check greeting is a string
        self.assertIsInstance(greeting, str)
        self.assertGreater(len(greeting), 0)
    
    def test_domain_knowledge(self):
        """Test domain knowledge capabilities"""
        # Add domain knowledge
        self.model.add_domain_knowledge("technology", "ai", "Artificial Intelligence is the simulation of human intelligence.")
        
        # Test knowledge retrieval
        knowledge = self.model.knowledge_base.retrieve_knowledge("artificial intelligence")
        
        # Check knowledge was retrieved
        self.assertGreater(len(knowledge), 0)
        self.assertIn("Artificial Intelligence", knowledge[0])
    
    def test_conversation_memory(self):
        """Test conversation memory"""
        inputs = {"text": torch.randint(0, self.config.vocab_size, (1, 128))}
        
        # Generate responses to build memory
        self.model.generate_response(inputs, "What is AI?")
        self.model.generate_response(inputs, "How does it work?")
        
        # Get conversation history
        history = self.model.get_conversation_history()
        
        # Check memory was stored
        self.assertEqual(len(history), 2)
        self.assertIn("query", history[0])
        self.assertIn("response", history[0])
        self.assertIn("timestamp", history[0])
    
    def test_ocr_weight_storage(self):
        """Test OCR weight storage"""
        # Store weights as OCR
        self.model.store_weights_as_ocr()
        
        # Check that weights were stored
        self.assertGreater(len(self.model.ocr_manager.weight_images), 0)
        self.assertGreater(len(self.model.ocr_manager.weight_metadata), 0)
    
    def test_training_capabilities(self):
        """Test training capabilities"""
        # Create sample training data
        training_data = [
            {
                "text": [1, 2, 3, 4, 5],
                "text_target": [2, 3, 4, 5, 6],
                "audio": np.random.randn(128, 256).tolist(),
                "audio_target": np.random.randn(128, 256).tolist()
            }
        ]
        
        # Test training
        initial_loss = float('inf')
        self.model.train_on_data(training_data, epochs=1, learning_rate=0.001)
        
        # Check that model can still forward pass after training
        inputs = {"text": torch.randint(0, self.config.vocab_size, (1, 128))}
        outputs = self.model.forward(inputs)
        self.assertIn("text", outputs)
    
    def test_memory_management(self):
        """Test memory management"""
        # Clear memory
        self.model.clear_memory()
        
        # Check memory is empty
        history = self.model.get_conversation_history()
        self.assertEqual(len(history), 0)
        
        # Add some memory
        inputs = {"text": torch.randint(0, self.config.vocab_size, (1, 128))}
        self.model.generate_response(inputs, "Test question")
        
        # Check memory was added
        history = self.model.get_conversation_history()
        self.assertEqual(len(history), 1)


class TestDomainKnowledgeBase(unittest.TestCase):
    """Test cases for Domain Knowledge Base"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MultiModalLanguageConfig()
        self.knowledge_base = DomainKnowledgeBase(self.config)
    
    def test_knowledge_initialization(self):
        """Test knowledge base initialization"""
        self.assertIsInstance(self.knowledge_base.knowledge_base, dict)
        self.assertGreater(len(self.knowledge_base.knowledge_base), 0)
    
    def test_add_knowledge(self):
        """Test adding knowledge"""
        self.knowledge_base.add_knowledge("test", "topic", "Test information")
        
        # Check knowledge was added
        self.assertIn("test", self.knowledge_base.knowledge_base)
        self.assertIn("topic", self.knowledge_base.knowledge_base["test"])
        self.assertEqual(self.knowledge_base.knowledge_base["test"]["topic"], "Test information")
    
    def test_retrieve_knowledge(self):
        """Test knowledge retrieval"""
        # Add test knowledge
        self.knowledge_base.add_knowledge("science", "physics", "Physics is the study of matter and energy")
        
        # Test retrieval
        knowledge = self.knowledge_base.retrieve_knowledge("physics")
        
        # Check knowledge was retrieved
        self.assertGreater(len(knowledge), 0)
        self.assertIn("Physics is the study of matter and energy", knowledge)


class TestReasoningEngine(unittest.TestCase):
    """Test cases for Reasoning Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MultiModalLanguageConfig(d_model=512)
        self.reasoning_engine = ReasoningEngine(self.config)
    
    def test_reasoning_forward(self):
        """Test reasoning forward pass"""
        # Create test input
        x = torch.randn(2, 128, self.config.d_model)
        
        # Forward pass
        output = self.reasoning_engine.forward(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
    
    def test_reasoning_components(self):
        """Test reasoning components"""
        # Check that all reasoning components exist
        self.assertIsInstance(self.reasoning_engine.logical_reasoning, nn.Linear)
        self.assertIsInstance(self.reasoning_engine.causal_reasoning, nn.Linear)
        self.assertIsInstance(self.reasoning_engine.analogical_reasoning, nn.Linear)
        self.assertIsInstance(self.reasoning_engine.reasoning_fusion, nn.Sequential)


class TestResponseGenerator(unittest.TestCase):
    """Test cases for Response Generator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MultiModalLanguageConfig()
        self.knowledge_base = DomainKnowledgeBase(self.config)
        self.response_generator = ResponseGenerator(self.config, self.knowledge_base)
    
    def test_greeting_generation(self):
        """Test greeting generation"""
        context = torch.randn(1, self.config.d_model)
        greeting = self.response_generator.generate_response(context)
        
        # Check greeting is valid
        self.assertIsInstance(greeting, str)
        self.assertGreater(len(greeting), 0)
        self.assertIn(greeting, self.response_generator.greeting_templates)
    
    def test_informed_response_generation(self):
        """Test informed response generation"""
        context = torch.randn(1, self.config.d_model)
        query = "What is artificial intelligence?"
        
        response = self.response_generator.generate_response(context, query)
        
        # Check response is valid
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MultiModalLanguageConfig(
            d_model=256,
            n_layers=4,
            vocab_size=5000,
            ocr_enabled=True
        )
        self.model = MultiModalLanguageModel(self.config)
        self.device = torch.device('cpu')
        self.model.to(self.device)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # 1. Add domain knowledge
        self.model.add_domain_knowledge("technology", "ai", "AI is the simulation of human intelligence.")
        
        # 2. Process multi-modal input
        inputs = {
            "text": torch.randint(0, self.config.vocab_size, (1, 64)),
            "audio": torch.randn(1, 64, self.config.audio_dim),
            "vision": torch.randn(1, 3, 224, 224)
        }
        
        # 3. Forward pass
        outputs = self.model.forward(inputs, use_reasoning=True)
        
        # 4. Generate response
        response = self.model.generate_response(inputs, "What is AI?")
        
        # 5. Store weights as OCR
        self.model.store_weights_as_ocr()
        
        # 6. Check conversation memory
        history = self.model.get_conversation_history()
        
        # Verify all components worked
        self.assertIn("text", outputs)
        self.assertIn("audio", outputs)
        self.assertIn("vision", outputs)
        self.assertIsInstance(response, str)
        self.assertEqual(len(history), 1)
        self.assertGreater(len(self.model.ocr_manager.weight_images), 0)
    
    def test_performance_benchmark(self):
        """Test performance benchmarks"""
        inputs = {
            "text": torch.randint(0, self.config.vocab_size, (1, 128)),
            "audio": torch.randn(1, 128, self.config.audio_dim),
            "vision": torch.randn(1, 3, 224, 224)
        }
        
        # Benchmark forward pass
        start_time = time.time()
        for _ in range(10):
            outputs = self.model.forward(inputs)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        logger.info(f"Average forward pass time: {avg_time:.4f} seconds")
        
        # Check that time is reasonable (less than 1 second per pass)
        self.assertLess(avg_time, 1.0)
    
    def test_memory_usage(self):
        """Test memory usage"""
        # Get initial memory usage
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Create large model
        large_config = MultiModalLanguageConfig(
            d_model=1024,
            n_layers=12,
            vocab_size=50000
        )
        large_model = MultiModalLanguageModel(large_config)
        
        # Get final memory usage
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Check memory usage is reasonable
        if torch.cuda.is_available():
            memory_used = final_memory - initial_memory
            logger.info(f"Memory used: {memory_used / 1024**2:.2f} MB")
            self.assertLess(memory_used, 1024**3)  # Less than 1GB


def run_comprehensive_tests():
    """Run comprehensive test suite"""
    logger.info("🧪 Starting Comprehensive Test Suite")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestMultiModalLanguageModel,
        TestDomainKnowledgeBase,
        TestReasoningEngine,
        TestResponseGenerator,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print results
    logger.info(f"\n📊 Test Results:")
    logger.info(f"  Tests run: {result.testsRun}")
    logger.info(f"  Failures: {len(result.failures)}")
    logger.info(f"  Errors: {len(result.errors)}")
    
    if result.failures:
        logger.error("❌ Test Failures:")
        for test, traceback in result.failures:
            logger.error(f"  {test}: {traceback}")
    
    if result.errors:
        logger.error("❌ Test Errors:")
        for test, traceback in result.errors:
            logger.error(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        logger.info("✅ All tests passed!")
    else:
        logger.error("❌ Some tests failed!")
    
    return result.wasSuccessful()


def test_model_capabilities():
    """Test specific model capabilities"""
    logger.info("🔍 Testing Model Capabilities")
    
    # Create model
    config = MultiModalLanguageConfig(
        d_model=512,
        n_layers=6,
        vocab_size=10000,
        ocr_enabled=True
    )
    model = MultiModalLanguageModel(config)
    
    # Test 1: Text Processing
    logger.info("Testing text processing...")
    text_input = {"text": torch.randint(0, config.vocab_size, (1, 128))}
    text_output = model.forward(text_input)
    assert "text" in text_output
    logger.info("✅ Text processing works")
    
    # Test 2: Audio Processing
    logger.info("Testing audio processing...")
    audio_input = {"audio": torch.randn(1, 128, config.audio_dim)}
    audio_output = model.forward(audio_input)
    assert "audio" in audio_output
    logger.info("✅ Audio processing works")
    
    # Test 3: Vision Processing
    logger.info("Testing vision processing...")
    vision_input = {"vision": torch.randn(1, 3, 224, 224)}
    vision_output = model.forward(vision_input)
    assert "vision" in vision_output
    logger.info("✅ Vision processing works")
    
    # Test 4: Multi-modal Processing
    logger.info("Testing multi-modal processing...")
    multimodal_input = {
        "text": torch.randint(0, config.vocab_size, (1, 128)),
        "audio": torch.randn(1, 128, config.audio_dim),
        "vision": torch.randn(1, 3, 224, 224)
    }
    multimodal_output = model.forward(multimodal_input)
    assert all(modality in multimodal_output for modality in ["text", "audio", "vision"])
    logger.info("✅ Multi-modal processing works")
    
    # Test 5: Response Generation
    logger.info("Testing response generation...")
    response = model.generate_response(text_input, "What is artificial intelligence?")
    assert isinstance(response, str) and len(response) > 0
    logger.info(f"✅ Response generation works: '{response}'")
    
    # Test 6: Greeting
    logger.info("Testing greeting...")
    greeting = model.generate_response(text_input)
    assert isinstance(greeting, str) and len(greeting) > 0
    logger.info(f"✅ Greeting works: '{greeting}'")
    
    # Test 7: Domain Knowledge
    logger.info("Testing domain knowledge...")
    model.add_domain_knowledge("technology", "ai", "AI is artificial intelligence.")
    knowledge = model.knowledge_base.retrieve_knowledge("ai")
    assert len(knowledge) > 0
    logger.info("✅ Domain knowledge works")
    
    # Test 8: OCR Weight Storage
    logger.info("Testing OCR weight storage...")
    model.store_weights_as_ocr()
    assert len(model.ocr_manager.weight_images) > 0
    logger.info("✅ OCR weight storage works")
    
    # Test 9: Training
    logger.info("Testing training...")
    training_data = [{
        "text": [1, 2, 3, 4, 5],
        "text_target": [2, 3, 4, 5, 6]
    }]
    model.train_on_data(training_data, epochs=1, learning_rate=0.001)
    logger.info("✅ Training works")
    
    # Test 10: Memory Management
    logger.info("Testing memory management...")
    model.clear_memory()
    assert len(model.get_conversation_history()) == 0
    model.generate_response(text_input, "Test question")
    assert len(model.get_conversation_history()) == 1
    logger.info("✅ Memory management works")
    
    logger.info("🎉 All capability tests passed!")


if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    if success:
        # Run capability tests
        test_model_capabilities()
        
        logger.info("🎉 All tests completed successfully!")
    else:
        logger.error("❌ Some tests failed!")
        exit(1)