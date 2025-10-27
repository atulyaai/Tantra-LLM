"""
Comprehensive Test Suite for OCR-Native LLM
Tests all OCR-native functionality and capabilities
"""

import unittest
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import json
from typing import Dict, Any, List

# Import the OCR-native LLM
import sys
sys.path.append('/workspace/Training')
from ocr_native_llm import OCRNativeLLM, OCRNativeConfig, OCRWeightEncoder, OCRMemoryBank, OCRInputProcessor


class TestOCRNativeLLM(unittest.TestCase):
    """Test cases for OCR-Native LLM"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = OCRNativeConfig(
            d_model=256,  # Smaller for testing
            n_layers=4,
            n_heads=4,
            d_ff=1024,
            vocab_size=1000,
            max_seq_length=1024,
            ocr_image_width=512,
            ocr_image_height=512
        )
        self.model = OCRNativeLLM(self.config)
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, OCRNativeLLM)
        self.assertEqual(self.model.config.d_model, 256)
        self.assertEqual(len(self.model.blocks), 4)
        self.assertIsNotNone(self.model.input_processor)
        self.assertIsNotNone(self.model.memory_bank)
        self.assertIsNotNone(self.model.weight_encoder)
    
    def test_ocr_weight_encoder(self):
        """Test OCR weight encoding"""
        # Create sample weights
        weights = torch.randn(10, 20)
        layer_name = "test_layer"
        
        # Encode to OCR
        ocr_image = self.model.weight_encoder.encode_weights_to_ocr(weights, layer_name)
        
        # Check image properties
        self.assertIsInstance(ocr_image, Image.Image)
        self.assertEqual(ocr_image.size, (512, 512))
        self.assertEqual(ocr_image.mode, 'L')  # Grayscale
    
    def test_ocr_input_processing(self):
        """Test OCR input processing"""
        # Test text processing
        text = "Hello, world!"
        ocr_text = self.model.input_processor.process_text_to_ocr(text)
        self.assertIsInstance(ocr_text, Image.Image)
        
        # Test speech processing
        audio_data = np.random.randn(1600)
        ocr_speech = self.model.input_processor.process_speech_to_ocr(audio_data)
        self.assertIsInstance(ocr_speech, Image.Image)
        
        # Test image processing
        image = Image.new('RGB', (224, 224), color='white')
        ocr_image = self.model.input_processor.process_image_to_ocr(image)
        self.assertIsInstance(ocr_image, Image.Image)
    
    def test_ocr_memory_bank(self):
        """Test OCR memory bank functionality"""
        # Store data in memory
        data = torch.randn(5, 10)
        memory_id = self.model.memory_bank.store_ocr_memory(data, "test_memory", 0.8)
        
        self.assertIsInstance(memory_id, str)
        self.assertIn("test_memory", memory_id)
        
        # Retrieve memory
        memories = self.model.memory_bank.retrieve_ocr_memory("test", top_k=3)
        self.assertIsInstance(memories, list)
    
    def test_forward_pass(self):
        """Test forward pass with OCR processing"""
        # Create sample inputs
        inputs = {
            'text': "Hello, how are you?",
            'speech': np.random.randn(1600),
            'image': Image.new('RGB', (224, 224), color='white')
        }
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Check outputs
        self.assertIn('text_logits', outputs)
        self.assertIn('ocr_output', outputs)
        self.assertIn('embeddings', outputs)
        
        # Check shapes
        self.assertEqual(outputs['text_logits'].shape[0], 1)  # Batch size
        self.assertEqual(outputs['ocr_output'].shape[0], 1)  # Batch size
    
    def test_response_generation(self):
        """Test response generation"""
        inputs = {
            'text': "What is AI?",
            'speech': np.random.randn(1600),
            'image': Image.new('RGB', (224, 224), color='white')
        }
        
        # Generate response
        response = self.model.generate_response(inputs, "Tell me about AI")
        
        self.assertIsInstance(response, str)
        self.assertIn("OCR Response", response)
    
    def test_memory_management(self):
        """Test memory management"""
        # Add to memory
        memory_id = self.model.add_to_memory("AI is artificial intelligence", "knowledge", 0.9)
        self.assertIsInstance(memory_id, str)
        
        # Get conversation history
        history = self.model.get_conversation_history()
        self.assertIsInstance(history, list)
        
        # Clear memory
        self.model.clear_memory()
        history_after = self.model.get_conversation_history()
        self.assertEqual(len(history_after), 0)
    
    def test_weight_storage_as_ocr(self):
        """Test storing weights as OCR images"""
        ocr_weights = self.model.store_weights_as_ocr()
        
        self.assertIsInstance(ocr_weights, dict)
        self.assertGreater(len(ocr_weights), 0)
        
        # Check that all weights are images
        for name, weight_image in ocr_weights.items():
            self.assertIsInstance(weight_image, Image.Image)
            self.assertIn(name, str(self.model.state_dict().keys()))
    
    def test_conversation_memory(self):
        """Test conversation memory functionality"""
        # Generate multiple responses
        inputs = {'text': "Hello"}
        
        for i in range(5):
            response = self.model.generate_response(inputs, f"Message {i}")
            self.assertIsInstance(response, str)
        
        # Check memory length
        history = self.model.get_conversation_history()
        self.assertLessEqual(len(history), self.config.memory_window_size)
    
    def test_ocr_attention_patterns(self):
        """Test OCR attention patterns"""
        # Create sample input
        x = torch.randn(1, 10, self.config.d_model)
        
        # Test attention block
        attention = self.model.blocks[0].attention
        output = attention(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertIsInstance(output, torch.Tensor)
    
    def test_multi_modal_processing(self):
        """Test multi-modal processing"""
        inputs = {
            'text': "Hello world",
            'speech': np.random.randn(1600),
            'image': Image.new('RGB', (224, 224), color='white')
        }
        
        # Process all modalities
        outputs = self.model(inputs)
        
        # Check that all modalities are processed
        self.assertIn('text_logits', outputs)
        self.assertIn('ocr_output', outputs)
        self.assertIn('embeddings', outputs)
        
        # Check embeddings shape
        self.assertEqual(outputs['embeddings'].shape[0], 1)  # Batch size


class TestOCRWeightEncoder(unittest.TestCase):
    """Test cases for OCR Weight Encoder"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = OCRNativeConfig()
        self.encoder = OCRWeightEncoder(self.config)
    
    def test_weight_to_ocr_text(self):
        """Test weight to OCR text conversion"""
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        layer_name = "test_layer"
        
        ocr_text = self.encoder._weights_to_ocr_text(weights, layer_name)
        
        self.assertIsInstance(ocr_text, str)
        self.assertIn("LAYER: test_layer", ocr_text)
        self.assertIn("SHAPE:", ocr_text)
        self.assertIn("VALUES:", ocr_text)
        self.assertIn("MEAN:", ocr_text)
        self.assertIn("STD:", ocr_text)
    
    def test_text_to_ocr_image(self):
        """Test text to OCR image conversion"""
        text = "Test OCR text"
        image = self.encoder._text_to_ocr_image(text)
        
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (self.config.ocr_image_width, self.config.ocr_image_height))
        self.assertEqual(image.mode, 'L')  # Grayscale
    
    def test_ocr_image_optimization(self):
        """Test OCR image optimization"""
        # Create test image
        image = Image.new('RGB', (100, 100), color='white')
        
        # Optimize for OCR
        optimized = self.encoder._optimize_for_ocr(image)
        
        self.assertIsInstance(optimized, Image.Image)
        self.assertEqual(optimized.mode, 'L')  # Should be grayscale


class TestOCRMemoryBank(unittest.TestCase):
    """Test cases for OCR Memory Bank"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = OCRNativeConfig()
        self.memory_bank = OCRMemoryBank(self.config)
    
    def test_store_ocr_memory(self):
        """Test storing OCR memory"""
        data = torch.randn(5, 10)
        memory_type = "test_memory"
        importance = 0.8
        
        memory_id = self.memory_bank.store_ocr_memory(data, memory_type, importance)
        
        self.assertIsInstance(memory_id, str)
        self.assertIn(memory_type, memory_id)
        self.assertEqual(len(self.memory_bank.memory_images), 1)
        self.assertEqual(len(self.memory_bank.memory_metadata), 1)
    
    def test_retrieve_ocr_memory(self):
        """Test retrieving OCR memory"""
        # Store some memories
        data1 = torch.randn(5, 10)
        data2 = torch.randn(3, 8)
        
        self.memory_bank.store_ocr_memory(data1, "test_memory_1", 0.8)
        self.memory_bank.store_ocr_memory(data2, "test_memory_2", 0.6)
        
        # Retrieve memories
        memories = self.memory_bank.retrieve_ocr_memory("test", top_k=2)
        
        self.assertIsInstance(memories, list)
        self.assertLessEqual(len(memories), 2)
        
        # Check that all retrieved items are images
        for memory in memories:
            self.assertIsInstance(memory, Image.Image)


class TestOCRInputProcessor(unittest.TestCase):
    """Test cases for OCR Input Processor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = OCRNativeConfig()
        self.processor = OCRInputProcessor(self.config)
    
    def test_text_preprocessing(self):
        """Test text preprocessing for OCR"""
        text = "Hello, world! 123"
        processed = self.processor._preprocess_text_for_ocr(text)
        
        self.assertIsInstance(processed, str)
        self.assertEqual(processed, "HELLO,  WORLD!  I23")  # After replacements
    
    def test_process_text_to_ocr(self):
        """Test text to OCR processing"""
        text = "Hello, world!"
        ocr_image = self.processor.process_text_to_ocr(text)
        
        self.assertIsInstance(ocr_image, Image.Image)
        self.assertEqual(ocr_image.size, (self.config.ocr_image_width, self.config.ocr_image_height))
    
    def test_process_speech_to_ocr(self):
        """Test speech to OCR processing"""
        audio_data = np.random.randn(1600)
        ocr_image = self.processor.process_speech_to_ocr(audio_data)
        
        self.assertIsInstance(ocr_image, Image.Image)
        self.assertEqual(ocr_image.size, (self.config.ocr_image_width, self.config.ocr_image_height))
    
    def test_process_image_to_ocr(self):
        """Test image to OCR processing"""
        image = Image.new('RGB', (224, 224), color='white')
        ocr_image = self.processor.process_image_to_ocr(image)
        
        self.assertIsInstance(ocr_image, Image.Image)
        self.assertEqual(ocr_image.size, (self.config.ocr_image_width, self.config.ocr_image_height))


class TestIntegration(unittest.TestCase):
    """Integration tests for OCR-native LLM"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = OCRNativeConfig(
            d_model=128,  # Very small for testing
            n_layers=2,
            n_heads=2,
            d_ff=256,
            vocab_size=100,
            max_seq_length=256
        )
        self.model = OCRNativeLLM(self.config)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # 1. Process inputs
        inputs = {
            'text': "What is artificial intelligence?",
            'speech': np.random.randn(800),
            'image': Image.new('RGB', (112, 112), color='white')
        }
        
        # 2. Generate response
        response = self.model.generate_response(inputs, "Explain AI")
        self.assertIsInstance(response, str)
        
        # 3. Add to memory
        memory_id = self.model.add_to_memory("AI is artificial intelligence", "knowledge", 0.9)
        self.assertIsInstance(memory_id, str)
        
        # 4. Store weights as OCR
        ocr_weights = self.model.store_weights_as_ocr()
        self.assertIsInstance(ocr_weights, dict)
        self.assertGreater(len(ocr_weights), 0)
        
        # 5. Check conversation history
        history = self.model.get_conversation_history()
        self.assertIsInstance(history, list)
    
    def test_memory_persistence(self):
        """Test memory persistence across multiple interactions"""
        # Add multiple memories
        memories = []
        for i in range(5):
            memory_id = self.model.add_to_memory(f"Memory {i}", "test", 0.8)
            memories.append(memory_id)
        
        # Check that all memories are stored
        self.assertEqual(len(memories), 5)
        self.assertEqual(len(self.model.memory_bank.memory_images), 5)
    
    def test_ocr_weight_consistency(self):
        """Test OCR weight consistency"""
        # Store weights as OCR
        ocr_weights1 = self.model.store_weights_as_ocr()
        
        # Store again
        ocr_weights2 = self.model.store_weights_as_ocr()
        
        # Check that same layers are present
        self.assertEqual(set(ocr_weights1.keys()), set(ocr_weights2.keys()))
        
        # Check that images have same dimensions
        for layer_name in ocr_weights1.keys():
            img1 = ocr_weights1[layer_name]
            img2 = ocr_weights2[layer_name]
            self.assertEqual(img1.size, img2.size)
            self.assertEqual(img1.mode, img2.mode)


class TestPerformance(unittest.TestCase):
    """Performance tests for OCR-native LLM"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = OCRNativeConfig(
            d_model=256,
            n_layers=4,
            n_heads=4,
            d_ff=1024,
            vocab_size=1000,
            max_seq_length=512
        )
        self.model = OCRNativeLLM(self.config)
    
    def test_memory_usage(self):
        """Test memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate multiple responses
        for i in range(10):
            inputs = {'text': f"Message {i}"}
            response = self.model.generate_response(inputs, f"Response {i}")
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable
        self.assertLess(memory_increase, 100)  # Less than 100MB increase
    
    def test_response_time(self):
        """Test response generation time"""
        import time
        
        inputs = {
            'text': "Hello, world!",
            'speech': np.random.randn(1600),
            'image': Image.new('RGB', (224, 224), color='white')
        }
        
        start_time = time.time()
        response = self.model.generate_response(inputs, "Test prompt")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Response should be generated quickly
        self.assertLess(response_time, 5.0)  # Less than 5 seconds
    
    def test_ocr_processing_speed(self):
        """Test OCR processing speed"""
        import time
        
        text = "This is a test text for OCR processing speed testing."
        
        start_time = time.time()
        ocr_image = self.model.input_processor.process_text_to_ocr(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # OCR processing should be fast
        self.assertLess(processing_time, 1.0)  # Less than 1 second


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)