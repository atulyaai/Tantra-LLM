"""
Unit tests for OCR-Native LLM
Clean, focused test suite
"""

import unittest
import torch
import numpy as np
from PIL import Image
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.ocr_native_llm import OCRNativeLLM, OCRNativeConfig
from src.configs.ocr_config import ConfigManager
from src.models.model_manager import ModelManager


class TestOCRNativeLLM(unittest.TestCase):
    """Test OCR-Native LLM functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Use small config for testing
        self.config = ConfigManager.get_small_config()
        self.model = OCRNativeLLM(self.config)
        
        # Test inputs
        self.test_inputs = {
            'text': 'Hello, world!',
            'speech': np.random.randn(1600),
            'image': Image.new('RGB', (224, 224), color='white')
        }
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, OCRNativeLLM)
        self.assertEqual(self.model.config.d_model, self.config.d_model)
        self.assertEqual(len(self.model.blocks), self.config.n_layers)
    
    def test_forward_pass(self):
        """Test forward pass with proper tensor shapes"""
        try:
            outputs = self.model(self.test_inputs)
            
            # Check output structure
            self.assertIn('text_logits', outputs)
            self.assertIn('ocr_output', outputs)
            self.assertIn('embeddings', outputs)
            
            # Check tensor shapes
            self.assertEqual(outputs['text_logits'].shape[0], 1)  # batch size
            self.assertEqual(outputs['text_logits'].shape[1], 1)  # sequence length
            self.assertEqual(outputs['text_logits'].shape[2], self.config.vocab_size)
            
            print("✓ Forward pass test passed")
            
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")
    
    def test_ocr_weight_encoding(self):
        """Test OCR weight encoding"""
        try:
            # Get a sample weight
            sample_weight = next(iter(self.model.parameters()))
            ocr_image = self.model.weight_encoder.encode_weights_to_ocr(sample_weight, "test_layer")
            
            self.assertIsInstance(ocr_image, Image.Image)
            self.assertEqual(ocr_image.size, (self.config.ocr_image_width, self.config.ocr_image_height))
            
            print("✓ OCR weight encoding test passed")
            
        except Exception as e:
            self.fail(f"OCR weight encoding failed: {e}")
    
    def test_ocr_input_processing(self):
        """Test OCR input processing"""
        try:
            # Test text processing
            text_ocr = self.model.input_processor.process_text_to_ocr("Test text")
            self.assertIsInstance(text_ocr, Image.Image)
            
            # Test speech processing
            speech_ocr = self.model.input_processor.process_speech_to_ocr(np.random.randn(1000))
            self.assertIsInstance(speech_ocr, Image.Image)
            
            # Test image processing
            image_ocr = self.model.input_processor.process_image_to_ocr(Image.new('RGB', (100, 100)))
            self.assertIsInstance(image_ocr, Image.Image)
            
            print("✓ OCR input processing test passed")
            
        except Exception as e:
            self.fail(f"OCR input processing failed: {e}")
    
    def test_memory_management(self):
        """Test OCR memory management"""
        try:
            # Add memory
            memory_id = self.model.add_to_memory("Test memory", "test", 0.8)
            self.assertIsInstance(memory_id, str)
            
            # Retrieve memory
            memories = self.model.memory_bank.retrieve_ocr_memory("test", top_k=5)
            self.assertIsInstance(memories, list)
            
            # Get conversation history
            history = self.model.get_conversation_history()
            self.assertIsInstance(history, list)
            
            print("✓ Memory management test passed")
            
        except Exception as e:
            self.fail(f"Memory management failed: {e}")
    
    def test_response_generation(self):
        """Test response generation"""
        try:
            response = self.model.generate_response(self.test_inputs, "Test prompt")
            self.assertIsInstance(response, str)
            self.assertIn("OCR-Native Response", response)
            
            print("✓ Response generation test passed")
            
        except Exception as e:
            self.fail(f"Response generation failed: {e}")
    
    def test_model_info(self):
        """Test model information"""
        try:
            info = self.model.get_model_info()
            
            self.assertIn('total_parameters', info)
            self.assertIn('trainable_parameters', info)
            self.assertIn('model_size_mb', info)
            self.assertIn('d_model', info)
            self.assertIn('n_layers', info)
            
            print(f"✓ Model info test passed - {info['total_parameters']:,} parameters")
            
        except Exception as e:
            self.fail(f"Model info test failed: {e}")
    
    def test_weight_storage(self):
        """Test weight storage as OCR"""
        try:
            ocr_weights = self.model.store_weights_as_ocr()
            self.assertIsInstance(ocr_weights, list)
            self.assertGreater(len(ocr_weights), 0)
            
            for weight_img in ocr_weights:
                self.assertIsInstance(weight_img, Image.Image)
            
            print(f"✓ Weight storage test passed - {len(ocr_weights)} weight images")
            
        except Exception as e:
            self.fail(f"Weight storage test failed: {e}")


class TestModelManager(unittest.TestCase):
    """Test Model Manager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_manager = ModelManager("test_models")
        self.config = ConfigManager.get_small_config()
    
    def test_model_creation(self):
        """Test model creation"""
        try:
            model = self.model_manager.create_model(self.config, "test_model")
            self.assertIsInstance(model, OCRNativeLLM)
            
            print("✓ Model creation test passed")
            
        except Exception as e:
            self.fail(f"Model creation failed: {e}")
    
    def test_model_saving_loading(self):
        """Test model saving and loading"""
        try:
            # Create and save model
            model = self.model_manager.create_model(self.config, "test_save_load")
            weights_path = self.model_manager.save_model(model, "test_save_load", "v1.0")
            
            # Load model
            loaded_model = self.model_manager.load_model("test_save_load", "v1.0")
            self.assertIsInstance(loaded_model, OCRNativeLLM)
            
            print("✓ Model saving/loading test passed")
            
        except Exception as e:
            self.fail(f"Model saving/loading failed: {e}")
    
    def test_model_registry(self):
        """Test model registry"""
        try:
            # List models
            models = self.model_manager.list_models()
            self.assertIsInstance(models, list)
            
            # Get model info
            if models:
                info = self.model_manager.get_model_info(models[0])
                self.assertIsInstance(info, dict)
            
            print("✓ Model registry test passed")
            
        except Exception as e:
            self.fail(f"Model registry test failed: {e}")


def run_tests():
    """Run all tests"""
    print("🧪 Running OCR-Native LLM Unit Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add OCR-Native LLM tests
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestOCRNativeLLM))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestModelManager))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed'}")
    
    return success


if __name__ == "__main__":
    run_tests()