#!/usr/bin/env python3
"""
OCR-Native LLM Experimentation Suite
Comprehensive testing and experimentation framework
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.interfaces.conversational import create_conversational_interface, quick_chat
from src.benchmarks.performance_benchmark import OCRNativeBenchmark, run_quick_benchmark, compare_variants
from src.architectures.transformer_variants import TransformerVariantConfig
from src.configs.ocr_config import ConfigManager
from src.utils.error_handler import logger


def experiment_1_conversational_testing():
    """Experiment 1: Test conversational capabilities across variants"""
    print("🧪 Experiment 1: Conversational Testing")
    print("=" * 60)
    
    variants = ["standard", "mamba", "hybrid", "memory_enhanced"]
    test_messages = [
        "Hello! How are you today?",
        "Explain what OCR-native processing means.",
        "What makes your architecture different from traditional LLMs?",
        "Can you process images and audio through OCR?",
        "Tell me about your memory system."
    ]
    
    results = {}
    
    for variant in variants:
        print(f"\n🔧 Testing {variant} variant...")
        variant_results = []
        
        for i, message in enumerate(test_messages, 1):
            print(f"  Test {i}/5: {message[:50]}...")
            
            try:
                response = quick_chat(message, variant)
                variant_results.append({
                    "message": message,
                    "response": response,
                    "success": True
                })
                print(f"    ✅ Response: {response[:100]}...")
            except Exception as e:
                variant_results.append({
                    "message": message,
                    "response": str(e),
                    "success": False
                })
                print(f"    ❌ Error: {e}")
        
        results[variant] = variant_results
        print(f"  📊 {variant}: {sum(1 for r in variant_results if r['success'])}/5 successful")
    
    # Save results
    import json
    with open("experiment_1_conversational_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to experiment_1_conversational_results.json")
    return results


def experiment_2_performance_benchmarking():
    """Experiment 2: Comprehensive performance benchmarking"""
    print("\n🧪 Experiment 2: Performance Benchmarking")
    print("=" * 60)
    
    # Run comprehensive benchmark
    benchmark = OCRNativeBenchmark("experiment_2_benchmark_results")
    
    print("Running comprehensive benchmark...")
    suite = benchmark.run_comprehensive_benchmark()
    
    # Print summary
    print(f"\n📊 Benchmark Summary:")
    print(f"   Total Tests: {suite.summary['total_tests']}")
    print(f"   Successful: {suite.summary['successful_tests']}")
    print(f"   Failed: {suite.summary['failed_tests']}")
    print(f"   Average Throughput: {suite.summary['average_throughput']:.2f} ops/s")
    print(f"   Average Duration: {suite.summary['average_duration']:.4f}s")
    print(f"   Average Memory Usage: {suite.summary['average_memory_usage']:.2f}%")
    
    if suite.summary.get('best_performing_model'):
        print(f"   Best Model: {suite.summary['best_performing_model']}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    benchmark.generate_visualizations(suite)
    
    print(f"\n📁 Results saved to experiment_2_benchmark_results/")
    return suite


def experiment_3_architecture_comparison():
    """Experiment 3: Compare different architectures"""
    print("\n🧪 Experiment 3: Architecture Comparison")
    print("=" * 60)
    
    # Quick comparison
    print("Running quick architecture comparison...")
    comparison = compare_variants()
    
    if not comparison:
        print("❌ No comparison data available")
        return {}
    
    print("\n📊 Architecture Performance Comparison:")
    print("-" * 40)
    
    # Sort by performance
    sorted_models = sorted(comparison.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model, throughput) in enumerate(sorted_models, 1):
        size, variant = model.split('_', 1)
        print(f"{i:2d}. {variant:15s} ({size:5s}): {throughput:6.2f} ops/s")
    
    print(f"\n🏆 Best Architecture: {sorted_models[0][0]}")
    
    # Save comparison
    import json
    with open("experiment_3_architecture_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n📁 Results saved to experiment_3_architecture_comparison.json")
    return comparison


def experiment_4_memory_analysis():
    """Experiment 4: Memory usage analysis"""
    print("\n🧪 Experiment 4: Memory Analysis")
    print("=" * 60)
    
    import psutil
    import torch
    
    # Test different model sizes
    sizes = ["small", "default", "large"]
    memory_results = {}
    
    for size in sizes:
        print(f"\n🔍 Testing {size} model memory usage...")
        
        try:
            # Get initial memory
            initial_memory = psutil.virtual_memory().used / (1024**3)
            
            # Create model
            interface = create_conversational_interface(size, "standard")
            session_id = interface.start_conversation("memory_test")
            
            # Get memory after model creation
            model_memory = psutil.virtual_memory().used / (1024**3)
            
            # Test with different inputs
            test_inputs = [
                "Short test message",
                "This is a longer test message that should use more memory for processing",
                " ".join(["Very long message"] * 50)
            ]
            
            input_memories = []
            for i, test_input in enumerate(test_inputs):
                # Get memory before processing
                before_memory = psutil.virtual_memory().used / (1024**3)
                
                # Process input
                response = interface.send_message(session_id, test_input)
                
                # Get memory after processing
                after_memory = psutil.virtual_memory().used / (1024**3)
                
                input_memories.append({
                    "input_length": len(test_input),
                    "before_memory": before_memory,
                    "after_memory": after_memory,
                    "memory_delta": after_memory - before_memory
                })
            
            # End session
            interface.end_conversation(session_id)
            
            memory_results[size] = {
                "initial_memory": initial_memory,
                "model_memory": model_memory,
                "model_memory_delta": model_memory - initial_memory,
                "input_processing": input_memories
            }
            
            print(f"  ✅ {size}: Model uses {memory_results[size]['model_memory_delta']:.2f} GB")
            
        except Exception as e:
            print(f"  ❌ Error testing {size}: {e}")
            memory_results[size] = {"error": str(e)}
    
    # Print summary
    print(f"\n📊 Memory Analysis Summary:")
    print("-" * 30)
    for size, results in memory_results.items():
        if "error" not in results:
            print(f"{size:8s}: Model {results['model_memory_delta']:5.2f} GB, "
                  f"Avg Input {np.mean([i['memory_delta'] for i in results['input_processing']]):5.3f} GB")
        else:
            print(f"{size:8s}: Error - {results['error']}")
    
    # Save results
    import json
    with open("experiment_4_memory_analysis.json", "w") as f:
        json.dump(memory_results, f, indent=2)
    
    print(f"\n📁 Results saved to experiment_4_memory_analysis.json")
    return memory_results


def experiment_5_ocr_processing_demo():
    """Experiment 5: OCR processing demonstration"""
    print("\n🧪 Experiment 5: OCR Processing Demonstration")
    print("=" * 60)
    
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Create test images
    test_images = []
    
    # Text image
    text_img = Image.new('RGB', (300, 100), 'white')
    draw = ImageDraw.Draw(text_img)
    draw.text((10, 10), "Hello OCR-Native World!", fill='black')
    test_images.append(("text", text_img))
    
    # Number image
    num_img = Image.new('RGB', (200, 100), 'white')
    draw = ImageDraw.Draw(num_img)
    draw.text((10, 10), "12345 67890", fill='black')
    test_images.append(("numbers", num_img))
    
    # Pattern image
    pattern_img = Image.new('RGB', (250, 150), 'white')
    draw = ImageDraw.Draw(pattern_img)
    for i in range(0, 250, 20):
        draw.line([(i, 0), (i, 150)], fill='black', width=1)
    for j in range(0, 150, 20):
        draw.line([(0, j), (250, j)], fill='black', width=1)
    test_images.append(("pattern", pattern_img))
    
    # Test with different variants
    variants = ["standard", "mamba", "hybrid"]
    results = {}
    
    for variant in variants:
        print(f"\n🔧 Testing {variant} variant with OCR processing...")
        variant_results = []
        
        try:
            interface = create_conversational_interface("small", variant)
            session_id = interface.start_conversation("ocr_test")
            
            for img_type, img in test_images:
                print(f"  Processing {img_type} image...")
                
                # Test image processing
                response = interface.send_message(
                    session_id, 
                    f"Analyze this {img_type} image", 
                    message_type="image",
                    media_data=img
                )
                
                variant_results.append({
                    "image_type": img_type,
                    "response": response['response'],
                    "success": True
                })
                
                print(f"    ✅ Response: {response['response'][:100]}...")
            
            interface.end_conversation(session_id)
            results[variant] = variant_results
            
        except Exception as e:
            print(f"  ❌ Error with {variant}: {e}")
            results[variant] = [{"error": str(e)}]
    
    # Save results
    import json
    with open("experiment_5_ocr_processing.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to experiment_5_ocr_processing.json")
    return results


def run_all_experiments():
    """Run all experiments"""
    print("🚀 OCR-Native LLM Comprehensive Experimentation Suite")
    print("=" * 80)
    print("This will run multiple experiments to test different aspects of the system.")
    print("Results will be saved to individual JSON files and benchmark directories.")
    print("=" * 80)
    
    all_results = {}
    
    try:
        # Experiment 1: Conversational Testing
        all_results["conversational"] = experiment_1_conversational_testing()
        
        # Experiment 2: Performance Benchmarking
        all_results["performance"] = experiment_2_performance_benchmarking()
        
        # Experiment 3: Architecture Comparison
        all_results["architecture"] = experiment_3_architecture_comparison()
        
        # Experiment 4: Memory Analysis
        all_results["memory"] = experiment_4_memory_analysis()
        
        # Experiment 5: OCR Processing Demo
        all_results["ocr_processing"] = experiment_5_ocr_processing_demo()
        
        # Save combined results
        import json
        with open("all_experiments_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n🎉 All experiments completed!")
        print(f"📁 Combined results saved to all_experiments_results.json")
        
        # Print final summary
        print(f"\n📊 Final Summary:")
        print(f"   Experiments Run: 5")
        print(f"   Conversational Tests: {len(all_results.get('conversational', {}))} variants")
        print(f"   Performance Tests: {all_results.get('performance', {}).get('summary', {}).get('total_tests', 0)} total")
        print(f"   Architecture Comparisons: {len(all_results.get('architecture', {}))} models")
        print(f"   Memory Analysis: {len(all_results.get('memory', {}))} sizes")
        print(f"   OCR Processing: {len(all_results.get('ocr_processing', {}))} variants")
        
    except Exception as e:
        logger.error(f"Experiment suite error: {e}")
        print(f"\n❌ Experiment suite failed: {e}")
    
    return all_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OCR-Native LLM Experimentation Suite")
    parser.add_argument("--experiment", choices=["1", "2", "3", "4", "5", "all"], 
                       default="all", help="Which experiment to run")
    parser.add_argument("--quick", action="store_true", help="Run quick versions")
    
    args = parser.parse_args()
    
    if args.experiment == "1":
        experiment_1_conversational_testing()
    elif args.experiment == "2":
        experiment_2_performance_benchmarking()
    elif args.experiment == "3":
        experiment_3_architecture_comparison()
    elif args.experiment == "4":
        experiment_4_memory_analysis()
    elif args.experiment == "5":
        experiment_5_ocr_processing_demo()
    elif args.experiment == "all":
        run_all_experiments()


if __name__ == "__main__":
    main()