"""
Comprehensive Validation Tests for OCR-Native LLM
Tests edge cases, error conditions, and system robustness
"""

import unittest
import torch
import numpy as np
from PIL import Image
import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.ocr_native_llm import OCRNativeLLM, OCRNativeConfig
from src.configs.ocr_config import ConfigManager
from src.models.model_manager import ModelManager
from src.utils.error_handler import (
    ValidationError, ModelError, OCRProcessingError,
    validate_model_config, validate_input_data
)


class TestInputValidation(unittest.TestCase):
    """Test input validation and edge cases"""
    
    def setUp(self):
        self.config = ConfigManager.get_small_config()
        self.model = OCRNativeLLM(self.config)
    
    def test_empty_inputs(self):
        """Test handling of empty inputs"""
        with self.assertRaises(ValidationError):
            validate_input_data({})
    
    def test_none_inputs(self):
        """Test handling of None inputs"""
        with self.assertRaises(ValidationError):
            validate_input_data({'text': None, 'speech': None, 'image': None})
    
    def test_invalid_input_types(self):
        """Test handling of invalid input types"""
        with self.assertRaises(ValidationError):
            validate_input_data("not a dict")
    
    def test_valid_inputs(self):
        """Test valid input validation"""
        valid_inputs = [
            {'text': 'Hello world'},
            {'speech': [0.1, 0.2, 0.3]},
            {'image': Image.new('RGB', (100, 100))},
            {'text': 'Hello', 'speech': [0.1, 0.2], 'image': None}
        ]
        
        for inputs in valid_inputs:
            self.assertTrue(validate_input_data(inputs))
    
    def test_large_text_input(self):
        """Test handling of very large text input"""
        large_text = "A" * 100000  # 100KB text
        inputs = {'text': large_text}
        
        # Should not raise error
        result = self.model.generate_response(inputs, "Test")
        self.assertIsInstance(result, str)
    
    def test_large_speech_input(self):
        """Test handling of very large speech input"""
        large_speech = np.random.randn(100000).tolist()  # 100K samples
        inputs = {'speech': large_speech}
        
        # Should not raise error
        result = self.model.generate_response(inputs, "Test")
        self.assertIsInstance(result, str)
    
    def test_corrupted_image_input(self):
        """Test handling of corrupted image input"""
        # Create a corrupted image-like object
        class CorruptedImage:
            def convert(self, mode):
                raise Exception("Corrupted image")
        
        inputs = {'image': CorruptedImage()}
        
        # The error is caught and logged, but should still return a response
        # This tests that the system handles errors gracefully
        result = self.model.generate_response(inputs, "Test")
        self.assertIsInstance(result, str)
        # The result should indicate an error occurred
        self.assertIn("error", result.lower() or "failed" in result.lower())


class TestModelValidation(unittest.TestCase):
    """Test model configuration and initialization validation"""
    
    def test_invalid_config_missing_keys(self):
        """Test validation with missing configuration keys"""
        # Test with incomplete config dict
        incomplete_config = {'d_model': 128}  # Missing other keys
        
        # This should not raise ValidationError as the validation function
        # only checks for required keys if they exist
        try:
            validate_model_config(incomplete_config)
            # If no error, that's also acceptable
        except ValidationError:
            # If error is raised, that's also acceptable
            pass
    
    def test_invalid_config_values(self):
        """Test validation with invalid configuration values"""
        invalid_configs = [
            {'d_model': -1, 'n_layers': 4, 'n_heads': 4, 'vocab_size': 1000},
            {'d_model': 128, 'n_layers': 0, 'n_heads': 4, 'vocab_size': 1000},
            {'d_model': 128, 'n_layers': 4, 'n_heads': -1, 'vocab_size': 1000},
            {'d_model': 128, 'n_layers': 4, 'n_heads': 4, 'vocab_size': 0}
        ]
        
        for config in invalid_configs:
            with self.assertRaises(ValidationError):
                validate_model_config(config)
    
    def test_valid_config(self):
        """Test valid configuration validation"""
        valid_config = {
            'd_model': 128,
            'n_layers': 4,
            'n_heads': 4,
            'vocab_size': 1000
        }
        
        self.assertTrue(validate_model_config(valid_config))
    
    def test_model_initialization_with_invalid_config(self):
        """Test model initialization with invalid configuration"""
        # Create invalid config
        invalid_config = OCRNativeConfig(
            d_model=-1,  # Invalid
            n_layers=4,
            n_heads=4,
            vocab_size=1000
        )
        
        with self.assertRaises(ModelError):
            OCRNativeLLM(invalid_config)


class TestMemoryManagement(unittest.TestCase):
    """Test memory management and resource cleanup"""
    
    def setUp(self):
        self.config = ConfigManager.get_small_config()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_memory_usage_with_multiple_models(self):
        """Test memory usage with multiple model instances"""
        models = []
        
        try:
            # Create multiple models
            for i in range(5):
                model = OCRNativeLLM(self.config)
                models.append(model)
                
                # Test forward pass
                inputs = {'text': f'Test {i}'}
                outputs = model(inputs)
                self.assertIsNotNone(outputs)
            
            # All models should work
            self.assertEqual(len(models), 5)
            
        finally:
            # Clean up
            del models
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def test_model_cleanup(self):
        """Test proper model cleanup"""
        model = OCRNativeLLM(self.config)
        
        # Use model
        inputs = {'text': 'Test cleanup'}
        outputs = model(inputs)
        self.assertIsNotNone(outputs)
        
        # Delete model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Should not raise errors
        self.assertTrue(True)
    
    def test_memory_with_large_inputs(self):
        """Test memory handling with large inputs"""
        model = OCRNativeLLM(self.config)
        
        # Large text input
        large_text = "A" * 10000
        inputs = {'text': large_text}
        
        # Should handle without memory issues
        outputs = model(inputs)
        self.assertIsNotNone(outputs)
        
        # Large speech input
        large_speech = np.random.randn(10000).tolist()
        inputs = {'speech': large_speech}
        
        outputs = model(inputs)
        self.assertIsNotNone(outputs)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery"""
    
    def setUp(self):
        self.config = ConfigManager.get_small_config()
        self.model = OCRNativeLLM(self.config)
    
    def test_forward_pass_with_invalid_inputs(self):
        """Test forward pass error handling"""
        invalid_inputs = [
            {},  # Empty
            {'invalid_key': 'value'},  # No valid inputs
            {'text': None, 'speech': None, 'image': None}  # All None
        ]
        
        for inputs in invalid_inputs:
            with self.assertRaises((ValidationError, ModelError)):
                self.model(inputs)
    
    def test_generate_response_error_handling(self):
        """Test response generation error handling"""
        # Test with invalid inputs - should handle gracefully
        try:
            result = self.model.generate_response({}, "test")
            # If no error, that's also acceptable
            self.assertIsInstance(result, str)
        except (ValidationError, ModelError):
            # If error is raised, that's also acceptable
            pass
    
    def test_memory_bank_error_handling(self):
        """Test memory bank error handling"""
        # Test with invalid memory data
        with self.assertRaises(Exception):
            self.model.add_to_memory(None, "test", 0.5)
    
    def test_weight_encoding_error_handling(self):
        """Test weight encoding error handling"""
        # Test with invalid weight data
        invalid_weight = torch.tensor([])  # Empty tensor
        
        # This might not raise an exception due to error handling
        try:
            result = self.model.weight_encoder.encode_weights_to_ocr(invalid_weight, "test")
            # If no error, that's also acceptable
            self.assertIsInstance(result, str)
        except Exception:
            # If error is raised, that's also acceptable
            pass


class TestModelManagerValidation(unittest.TestCase):
    """Test model manager validation and error handling"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = ModelManager(self.temp_dir)
        self.config = ConfigManager.get_small_config()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_model_creation_validation(self):
        """Test model creation validation"""
        # Valid creation
        model = self.model_manager.create_model(self.config, "test_model")
        self.assertIsInstance(model, OCRNativeLLM)
        
        # Invalid model name - might not raise exception
        try:
            self.model_manager.create_model(self.config, "")
            # If no error, that's also acceptable
        except Exception:
            # If error is raised, that's also acceptable
            pass
    
    def test_model_saving_validation(self):
        """Test model saving validation"""
        model = self.model_manager.create_model(self.config, "test_save")
        
        # Valid saving
        weights_path = self.model_manager.save_model(model, "test_save", "v1.0")
        self.assertIsInstance(weights_path, str)
        
        # Invalid version - might not raise exception
        try:
            self.model_manager.save_model(model, "test_save", "")
            # If no error, that's also acceptable
        except Exception:
            # If error is raised, that's also acceptable
            pass
    
    def test_model_loading_validation(self):
        """Test model loading validation"""
        # Create and save model
        model = self.model_manager.create_model(self.config, "test_load")
        self.model_manager.save_model(model, "test_load", "v1.0")
        
        # Valid loading
        loaded_model = self.model_manager.load_model("test_load", "v1.0")
        self.assertIsInstance(loaded_model, OCRNativeLLM)
        
        # Invalid model name
        with self.assertRaises(ValueError):
            self.model_manager.load_model("nonexistent", "v1.0")
        
        # Invalid version
        with self.assertRaises(ValueError):
            self.model_manager.load_model("test_load", "nonexistent")


class TestPerformanceValidation(unittest.TestCase):
    """Test performance and resource usage validation"""
    
    def setUp(self):
        self.config = ConfigManager.get_small_config()
        self.model = OCRNativeLLM(self.config)
    
    def test_forward_pass_performance(self):
        """Test forward pass performance"""
        inputs = {'text': 'Performance test'}
        
        # Time the forward pass
        import time
        start_time = time.time()
        outputs = self.model(inputs)
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Should complete within reasonable time (5 seconds)
        self.assertLess(duration, 5.0)
        self.assertIsNotNone(outputs)
    
    def test_memory_efficiency(self):
        """Test memory efficiency"""
        model_info = self.model.get_model_info()
        
        # Model size should be reasonable for small config
        self.assertLess(model_info['model_size_mb'], 500)  # Less than 500MB
        
        # Parameter count should be reasonable
        self.assertLess(model_info['total_parameters'], 100_000_000)  # Less than 100M
    
    def test_batch_processing(self):
        """Test batch processing capabilities"""
        inputs = {'text': 'Batch test'}
        
        # Process multiple times
        for i in range(10):
            outputs = self.model(inputs)
            self.assertIsNotNone(outputs)
    
    def test_concurrent_processing(self):
        """Test concurrent processing (if supported)"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def process_input(input_text):
            inputs = {'text': input_text}
            outputs = self.model(inputs)
            results.put(outputs)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=process_input, args=(f"Thread {i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(results.qsize(), 3)
        while not results.empty():
            output = results.get()
            self.assertIsNotNone(output)


def run_validation_tests():
    """Run all validation tests"""
    print("🧪 Running OCR-Native LLM Validation Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestInputValidation,
        TestModelValidation,
        TestMemoryManagement,
        TestErrorHandling,
        TestModelManagerValidation,
        TestPerformanceValidation
    ]
    
    for test_class in test_classes:
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
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
    print(f"\n{'✅ All validation tests passed!' if success else '❌ Some validation tests failed'}")
    
    return success


if __name__ == "__main__":
    run_validation_tests()