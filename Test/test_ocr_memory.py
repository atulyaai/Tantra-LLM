"""
Test script for OCR Memory System
Tests the OCR-based memory functionality
"""

import torch
import torch.nn as nn
import sys
import os

# Add the Training directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Training'))

from ocr_memory import OCRMemoryConfig, OCRMemoryBank, OCRContextualMemory
from ocr_native_model import OCRNativeModel, OCRNativeConfig
from ocr_implementation_guide import OCRImplementationGuide


def test_ocr_memory_basic():
    """Test basic OCR memory functionality"""
    print("Testing Basic OCR Memory Functionality")
    print("=" * 50)
    
    # Create OCR memory configuration
    config = OCRMemoryConfig(
        image_width=128,
        image_height=128,
        precision_digits=4,
        encoding_scheme="scientific"
    )
    
    # Create memory bank
    memory_bank = OCRMemoryBank(config)
    
    # Test storing weights
    print("1. Testing weight storage...")
    sample_weights = torch.randn(4, 4)
    image_id = memory_bank.store_memory("test_layer", sample_weights)
    print(f"   ✓ Stored weights with ID: {image_id}")
    
    # Test retrieving weights
    print("2. Testing weight retrieval...")
    memories = memory_bank.retrieve_memory("test_layer")
    if memories:
        retrieved_id, retrieved_weights, metadata = memories[0]
        print(f"   ✓ Retrieved weights shape: {retrieved_weights.shape}")
        print(f"   ✓ Original shape: {sample_weights.shape}")
        
        # Check reconstruction accuracy
        mse_error = torch.nn.functional.mse_loss(sample_weights, retrieved_weights)
        print(f"   ✓ Reconstruction MSE: {mse_error.item():.6f}")
    else:
        print("   ❌ No memories retrieved")
        return False
    
    # Test memory statistics
    print("3. Testing memory statistics...")
    stats = memory_bank.list_memories()
    print(f"   ✓ Total memories: {len(stats)}")
    
    print("✓ Basic OCR memory test passed!")
    return True


def test_ocr_native_model():
    """Test OCR-native model"""
    print("\nTesting OCR-Native Model")
    print("=" * 50)
    
    # Create model configuration
    config = OCRNativeConfig(
        d_model=256,
        n_layers=4,
        image_size=128,
        memory_capacity=100
    )
    
    # Create model
    model = OCRNativeModel(config)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    print("1. Testing forward pass...")
    batch_size = 2
    seq_len = 64
    
    inputs = {
        "audio": torch.randn(batch_size, seq_len, config.audio_dim),
        "text": torch.randint(0, 32000, (batch_size, seq_len)),
        "vision": torch.randn(batch_size, 3, config.image_size, config.image_size)
    }
    
    try:
        outputs = model(inputs)
        print(f"   ✓ Forward pass successful")
        print(f"   - Audio output: {outputs['audio'].shape}")
        print(f"   - Text output: {outputs['text'].shape}")
        print(f"   - Vision output: {outputs['vision'].shape}")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False
    
    # Test memory storage
    print("2. Testing memory storage...")
    try:
        model.store_memory(inputs, "test_memory")
        print("   ✓ Memory stored successfully")
    except Exception as e:
        print(f"   ❌ Memory storage failed: {e}")
        return False
    
    # Test memory retrieval
    print("3. Testing memory retrieval...")
    try:
        retrieved = model.retrieve_memory(inputs)
        print(f"   ✓ Retrieved {len(retrieved)} memories")
    except Exception as e:
        print(f"   ❌ Memory retrieval failed: {e}")
        return False
    
    # Test memory statistics
    print("4. Testing memory statistics...")
    try:
        stats = model.get_memory_statistics()
        print(f"   ✓ Memory statistics: {stats}")
    except Exception as e:
        print(f"   ❌ Memory statistics failed: {e}")
        return False
    
    print("✓ OCR-native model test passed!")
    return True


def test_contextual_memory():
    """Test contextual memory functionality"""
    print("\nTesting Contextual Memory")
    print("=" * 50)
    
    # Create configuration
    config = OCRMemoryConfig(
        image_width=128,
        image_height=128,
        precision_digits=4
    )
    
    # Create contextual memory
    contextual_memory = OCRContextualMemory(config, d_model=256)
    
    # Test storing contextual memory
    print("1. Testing contextual memory storage...")
    context = torch.randn(1, 10, 256)
    weights = torch.randn(4, 4)
    
    try:
        memory_id = contextual_memory.store_contextual_memory(
            context, weights, "test_context"
        )
        print(f"   ✓ Stored contextual memory with ID: {memory_id}")
    except Exception as e:
        print(f"   ❌ Contextual memory storage failed: {e}")
        return False
    
    # Test retrieving contextual memory
    print("2. Testing contextual memory retrieval...")
    try:
        retrieved = contextual_memory.retrieve_contextual_memory(context, "test_context")
        print(f"   ✓ Retrieved {len(retrieved)} contextual memories")
    except Exception as e:
        print(f"   ❌ Contextual memory retrieval failed: {e}")
        return False
    
    print("✓ Contextual memory test passed!")
    return True


def test_ocr_encoding_decoding():
    """Test OCR encoding and decoding accuracy"""
    print("\nTesting OCR Encoding/Decoding Accuracy")
    print("=" * 50)
    
    from ocr_memory import OCREncoder, OCRDecoder
    
    # Create encoder and decoder
    config = OCRMemoryConfig(
        image_width=128,
        image_height=128,
        precision_digits=4
    )
    
    encoder = OCREncoder(config)
    decoder = OCRDecoder(config)
    
    # Test with different weight tensors
    test_cases = [
        torch.randn(2, 2),
        torch.randn(3, 3, 3),
        torch.randn(1, 10),
        torch.ones(4, 4) * 0.5,
        torch.zeros(2, 3)
    ]
    
    total_error = 0
    num_tests = len(test_cases)
    
    for i, weights in enumerate(test_cases):
        print(f"   Test case {i+1}/{num_tests}: {weights.shape}")
        
        try:
            # Encode to image
            image = encoder.encode_weights_to_image(weights, f"test_{i}")
            
            # Decode back to weights
            text = decoder.decode_image_to_text(image)
            reconstructed_weights, layer_name = decoder.decode_text_to_weights(text)
            
            # Calculate error
            error = torch.nn.functional.mse_loss(weights, reconstructed_weights)
            total_error += error.item()
            
            print(f"     ✓ MSE Error: {error.item():.6f}")
            
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            return False
    
    avg_error = total_error / num_tests
    print(f"   ✓ Average reconstruction error: {avg_error:.6f}")
    
    if avg_error < 0.1:  # Threshold for acceptable error
        print("✓ OCR encoding/decoding test passed!")
        return True
    else:
        print("❌ OCR encoding/decoding error too high!")
        return False


def run_performance_benchmark():
    """Run performance benchmark"""
    print("\nRunning Performance Benchmark")
    print("=" * 50)
    
    import time
    
    # Create model
    config = OCRNativeConfig(d_model=256, n_layers=4)
    model = OCRNativeModel(config)
    
    # Create test data
    inputs = {
        "audio": torch.randn(4, 64, 128),
        "text": torch.randint(0, 32000, (4, 64)),
        "vision": torch.randn(4, 3, 128, 128)
    }
    
    # Benchmark inference time
    print("1. Benchmarking inference time...")
    model.eval()
    
    num_runs = 10
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    print(f"   ✓ Average inference time: {avg_time:.4f} seconds")
    
    # Benchmark memory operations
    print("2. Benchmarking memory operations...")
    
    memory_times = []
    for i in range(5):
        start_time = time.time()
        model.store_memory(inputs, f"benchmark_{i}")
        end_time = time.time()
        memory_times.append(end_time - start_time)
    
    avg_memory_time = sum(memory_times) / len(memory_times)
    print(f"   ✓ Average memory storage time: {avg_memory_time:.4f} seconds")
    
    # Memory usage
    print("3. Checking memory usage...")
    stats = model.get_memory_statistics()
    print(f"   ✓ Total memories: {stats['total_memories']}")
    print(f"   ✓ Memory types: {stats['memory_types']}")
    
    print("✓ Performance benchmark completed!")
    return True


def main():
    """Run all tests"""
    print("OCR Memory System Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic OCR Memory", test_ocr_memory_basic),
        ("OCR-Native Model", test_ocr_native_model),
        ("Contextual Memory", test_contextual_memory),
        ("OCR Encoding/Decoding", test_ocr_encoding_decoding),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*60)
    
    if passed == total:
        print("🎉 All tests passed! OCR Memory System is working correctly.")
    else:
        print("💥 Some tests failed. Check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)