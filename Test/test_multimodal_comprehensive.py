"""
Comprehensive Multi-Modal Mamba 3 Test Suite
Tests all components: model, training, inference, API, and compression
"""

import torch
import torch.nn as nn
import numpy as np
import json
import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add Training directory to path
sys.path.append(str(Path(__file__).parent.parent / "Training"))

from model_mamba3_multimodal import (
    Mamba3MultiModal, Mamba3Config, DynamicVocabulary,
    QuantizedLinear, ExpertRouter, Mamba3Block,
    AudioEncoder, TextEncoder, VisionEncoder
)


class TestDynamicVocabulary(unittest.TestCase):
    """Test dynamic vocabulary functionality"""
    
    def setUp(self):
        self.vocab = DynamicVocabulary(initial_vocab_size=1000, max_vocab_size=5000)
    
    def test_initial_vocab_size(self):
        self.assertEqual(self.vocab.current_vocab_size, 1000)
    
    def test_frequency_update(self):
        tokens = [1, 2, 3, 1, 2, 1]
        self.vocab.update_frequencies(tokens)
        
        self.assertEqual(self.vocab.token_frequencies[1], 3)
        self.assertEqual(self.vocab.token_frequencies[2], 2)
        self.assertEqual(self.vocab.token_frequencies[3], 1)
    
    def test_vocab_growth_decision(self):
        # Test with low usage
        self.vocab.update_frequencies([1, 2, 3])
        self.assertFalse(self.vocab.should_grow_vocab())
        
        # Test with high usage
        tokens = list(range(800))  # Use 80% of vocabulary
        self.vocab.update_frequencies(tokens)
        self.assertTrue(self.vocab.should_grow_vocab())
    
    def test_vocab_growth(self):
        old_size = self.vocab.current_vocab_size
        growth = self.vocab.grow_vocabulary(500)
        
        self.assertEqual(growth, 500)
        self.assertEqual(self.vocab.current_vocab_size, old_size + 500)
    
    def test_max_vocab_limit(self):
        # Try to grow beyond max size
        self.vocab.current_vocab_size = 4500
        growth = self.vocab.grow_vocabulary(1000)
        
        self.assertEqual(growth, 500)  # Should be limited to max size
        self.assertEqual(self.vocab.current_vocab_size, 5000)


class TestQuantizedLinear(unittest.TestCase):
    """Test quantized linear layer"""
    
    def setUp(self):
        self.layer = QuantizedLinear(10, 5, bits=8)
    
    def test_initialization(self):
        self.assertEqual(self.layer.in_features, 10)
        self.assertEqual(self.layer.out_features, 5)
        self.assertEqual(self.layer.bits, 8)
    
    def test_forward_pass(self):
        x = torch.randn(3, 10)
        output = self.layer(x)
        
        self.assertEqual(output.shape, (3, 5))
    
    def test_quantization(self):
        # Test quantization doesn't break forward pass
        x = torch.randn(3, 10)
        
        # First forward pass (should quantize)
        output1 = self.layer(x)
        
        # Second forward pass (should use quantized weights)
        output2 = self.layer(x)
        
        # Outputs should be similar
        self.assertTrue(torch.allclose(output1, output2, atol=1e-3))


class TestExpertRouter(unittest.TestCase):
    """Test expert router functionality"""
    
    def setUp(self):
        self.router = ExpertRouter(
            d_model=64,
            num_experts=4,
            expert_categories=["audio", "text", "vision", "general"]
        )
    
    def test_initialization(self):
        self.assertEqual(self.router.d_model, 64)
        self.assertEqual(self.router.num_experts, 4)
        self.assertEqual(len(self.router.expert_categories), 4)
    
    def test_forward_pass(self):
        x = torch.randn(2, 10, 64)  # [batch, seq, d_model]
        expert_weights, expert_indices = self.router(x, modality="text")
        
        self.assertEqual(expert_weights.shape, (2, 10, 4))
        self.assertEqual(expert_indices.shape, (2, 10, 2))  # top-2 experts
        
        # Check that weights sum to 1
        self.assertTrue(torch.allclose(expert_weights.sum(dim=-1), torch.ones(2, 10)))
        
        # Check that indices are valid
        self.assertTrue(torch.all(expert_indices >= 0))
        self.assertTrue(torch.all(expert_indices < 4))


class TestMamba3Block(unittest.TestCase):
    """Test Mamba 3 block functionality"""
    
    def setUp(self):
        self.block = Mamba3Block(
            d_model=64,
            d_state=16,
            d_conv=4,
            dropout=0.1,
            use_quantization=False
        )
    
    def test_initialization(self):
        self.assertEqual(self.block.d_model, 64)
        self.assertEqual(self.block.d_state, 16)
        self.assertEqual(self.block.d_conv, 4)
    
    def test_forward_pass(self):
        x = torch.randn(2, 10, 64)
        output = self.block(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_residual_connection(self):
        x = torch.randn(2, 10, 64)
        output = self.block(x)
        
        # Output should be input + transformation
        # This is a basic check - in practice, the transformation might be small
        self.assertTrue(torch.allclose(output, x, atol=1e-1))


class TestModalityEncoders(unittest.TestCase):
    """Test modality-specific encoders"""
    
    def setUp(self):
        self.audio_encoder = AudioEncoder(input_dim=128, d_model=64)
        self.text_encoder = TextEncoder(vocab_size=1000, d_model=64, 
                                       dynamic_vocab=DynamicVocabulary())
        self.vision_encoder = VisionEncoder(input_dim=3, d_model=64)
    
    def test_audio_encoder(self):
        x = torch.randn(2, 100, 128)  # [batch, time, features]
        output = self.audio_encoder(x)
        
        self.assertEqual(output.shape, (2, 100, 64))
    
    def test_text_encoder(self):
        x = torch.randint(0, 1000, (2, 20))  # [batch, seq]
        output = self.text_encoder(x)
        
        self.assertEqual(output.shape, (2, 20, 64))
    
    def test_vision_encoder(self):
        x = torch.randn(2, 3, 224, 224)  # [batch, channels, height, width]
        output = self.vision_encoder(x)
        
        # Should output patches: 224/16 = 14, so 14*14 = 196 patches
        self.assertEqual(output.shape, (2, 196, 64))


class TestMamba3MultiModal(unittest.TestCase):
    """Test complete multi-modal model"""
    
    def setUp(self):
        self.config = Mamba3Config(
            d_model=64,
            n_layers=2,
            num_experts=4,
            quantization_bits=16,  # Disable quantization for testing
            pruning_ratio=0.0  # Disable pruning for testing
        )
        self.model = Mamba3MultiModal(self.config)
    
    def test_initialization(self):
        self.assertEqual(self.model.config.d_model, 64)
        self.assertEqual(self.model.config.n_layers, 2)
        self.assertEqual(self.model.config.num_experts, 4)
    
    def test_forward_pass_text_only(self):
        inputs = {
            "text": torch.randint(0, 1000, (2, 10))
        }
        outputs = self.model(inputs, modality_priority=["text"])
        
        self.assertIn("text", outputs)
        self.assertEqual(outputs["text"].shape, (2, 10, self.config.initial_vocab_size))
    
    def test_forward_pass_audio_only(self):
        inputs = {
            "audio": torch.randn(2, 100, 128)
        }
        outputs = self.model(inputs, modality_priority=["audio"])
        
        self.assertIn("audio", outputs)
        self.assertEqual(outputs["audio"].shape, (2, 100, self.config.audio_dim))
    
    def test_forward_pass_vision_only(self):
        inputs = {
            "vision": torch.randn(2, 3, 224, 224)
        }
        outputs = self.model(inputs, modality_priority=["vision"])
        
        self.assertIn("vision", outputs)
        self.assertEqual(outputs["vision"].shape, (2, 196, self.config.vision_dim))
    
    def test_forward_pass_multimodal(self):
        inputs = {
            "text": torch.randint(0, 1000, (2, 10)),
            "audio": torch.randn(2, 100, 128),
            "vision": torch.randn(2, 3, 224, 224)
        }
        outputs = self.model(inputs, modality_priority=["audio", "text", "vision"])
        
        self.assertIn("text", outputs)
        self.assertIn("audio", outputs)
        self.assertIn("vision", outputs)
    
    def test_compression(self):
        # Test compression methods don't break the model
        self.model.compress_model()
        
        # Model should still work after compression
        inputs = {"text": torch.randint(0, 1000, (1, 5))}
        outputs = self.model(inputs)
        
        self.assertIn("text", outputs)
    
    def test_vocabulary_update(self):
        # Test vocabulary update
        new_tokens = [1, 2, 3, 4, 5]
        self.model.update_vocabulary(new_tokens)
        
        # Should not raise an error
        self.assertTrue(True)


class TestTrainingPipeline(unittest.TestCase):
    """Test training pipeline components"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create necessary directories
        Path("Dataset").mkdir()
        Path("Model").mkdir()
        Path("Config").mkdir()
        Path("logs").mkdir()
    
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_config_loading(self):
        """Test configuration loading"""
        from training_multimodal import Mamba3Config
        
        config = Mamba3Config(d_model=128, n_layers=4)
        self.assertEqual(config.d_model, 128)
        self.assertEqual(config.n_layers, 4)
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        from training_multimodal import MultiModalDataset
        
        # Create sample data
        sample_data = {
            "audio": [{"audio_features": np.random.randn(128, 128).tolist()}],
            "text": [{"text": "Hello world"}],
            "vision": [{"image_features": np.random.randn(196, 512).tolist()}]
        }
        
        for modality, data in sample_data.items():
            file_path = f"Dataset/{modality}_data.jsonl"
            with open(file_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        
        # Test dataset loading
        config = Mamba3Config()
        dataset = MultiModalDataset(
            {"audio": "Dataset/audio_data.jsonl"},
            None,  # No tokenizer for this test
            config,
            max_samples=1
        )
        
        self.assertEqual(len(dataset), 1)
        
        # Test data loading
        inputs, targets = dataset[0]
        self.assertIn("audio", inputs)


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoint functionality"""
    
    def setUp(self):
        # This would require setting up a test client
        # For now, we'll test the data processing functions
        pass
    
    def test_audio_processing(self):
        """Test audio processing utilities"""
        from serve_multimodal_api import AudioProcessor
        
        # Create dummy audio data
        audio_data = np.random.randn(16000).astype(np.float32).tobytes()
        
        # Test processing
        features = AudioProcessor.process_audio(audio_data)
        
        self.assertEqual(features.shape, (128, 128))
    
    def test_vision_processing(self):
        """Test vision processing utilities"""
        from serve_multimodal_api import VisionProcessor
        
        # Create dummy image data
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image_data = image.tobytes()
        
        # Test processing
        features = VisionProcessor.process_image(image_data)
        
        self.assertEqual(features.shape, (196, 512))  # 14x14 patches
    
    def test_text_processing(self):
        """Test text processing utilities"""
        from serve_multimodal_api import TextProcessor
        
        # Test tokenization
        tokens = TextProcessor.process_text("Hello world", None)
        
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)


class TestCompression(unittest.TestCase):
    """Test compression functionality"""
    
    def setUp(self):
        self.config = Mamba3Config(
            d_model=64,
            n_layers=2,
            quantization_bits=8,
            pruning_ratio=0.1
        )
        self.model = Mamba3MultiModal(self.config)
    
    def test_quantization(self):
        """Test quantization functionality"""
        # Test quantized linear layer
        layer = QuantizedLinear(10, 5, bits=8)
        x = torch.randn(3, 10)
        
        # Should work without errors
        output = layer(x)
        self.assertEqual(output.shape, (3, 5))
    
    def test_pruning(self):
        """Test pruning functionality"""
        # Test pruning
        self.model.apply_pruning(0.1)
        
        # Model should still work
        inputs = {"text": torch.randint(0, 1000, (1, 5))}
        outputs = self.model(inputs)
        
        self.assertIn("text", outputs)
    
    def test_compression_integration(self):
        """Test integrated compression"""
        # Test full compression pipeline
        self.model.compress_model()
        
        # Model should still work
        inputs = {"text": torch.randint(0, 1000, (1, 5))}
        outputs = self.model(inputs)
        
        self.assertIn("text", outputs)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create necessary directories
        Path("Dataset").mkdir()
        Path("Model").mkdir()
        Path("Config").mkdir()
        Path("logs").mkdir()
    
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_full_training_cycle(self):
        """Test complete training cycle"""
        # This would test the full training pipeline
        # For now, we'll test individual components
        
        # Test model creation
        config = Mamba3Config(d_model=64, n_layers=2, num_experts=4)
        model = Mamba3MultiModal(config)
        
        # Test forward pass
        inputs = {
            "text": torch.randint(0, 1000, (1, 10)),
            "audio": torch.randn(1, 100, 128),
            "vision": torch.randn(1, 3, 224, 224)
        }
        
        outputs = model(inputs)
        
        # Check outputs
        self.assertIn("text", outputs)
        self.assertIn("audio", outputs)
        self.assertIn("vision", outputs)
        
        # Test compression
        model.compress_model()
        
        # Model should still work
        outputs_compressed = model(inputs)
        self.assertIn("text", outputs_compressed)


def run_performance_tests():
    """Run performance benchmarks"""
    print("\n🚀 Running Performance Tests...")
    
    # Test model creation speed
    import time
    
    start_time = time.time()
    config = Mamba3Config(d_model=256, n_layers=4, num_experts=4)
    model = Mamba3MultiModal(config)
    creation_time = time.time() - start_time
    
    print(f"Model creation time: {creation_time:.3f}s")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass speed
    inputs = {
        "text": torch.randint(0, 1000, (1, 50)),
        "audio": torch.randn(1, 100, 128),
        "vision": torch.randn(1, 3, 224, 224)
    }
    
    # Warmup
    for _ in range(5):
        _ = model(inputs)
    
    # Benchmark
    start_time = time.time()
    for _ in range(10):
        _ = model(inputs)
    inference_time = time.time() - start_time
    
    print(f"Average inference time: {inference_time/10:.3f}s")
    
    # Test memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")


def main():
    """Run all tests"""
    print("🧪 Running Multi-Modal Mamba 3 Test Suite")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_tests()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()