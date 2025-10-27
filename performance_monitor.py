#!/usr/bin/env python3
"""
Performance Monitoring Script for OCR-Native LLM
Monitors system performance and model metrics
"""

import sys
import os
import time
import psutil
import torch
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.ocr_native_llm import OCRNativeLLM
from src.configs.ocr_config import ConfigManager
from src.utils.error_handler import logger, performance_monitor


class SystemMonitor:
    """System performance monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = []
    
    def get_system_metrics(self):
        """Get current system metrics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'uptime_seconds': time.time() - self.start_time
        }
    
    def get_pytorch_metrics(self):
        """Get PyTorch-specific metrics"""
        if torch.cuda.is_available():
            return {
                'cuda_available': True,
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'cuda_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'cuda_device_name': torch.cuda.get_device_name(0)
            }
        else:
            return {
                'cuda_available': False,
                'cuda_device_count': 0,
                'cuda_memory_allocated_gb': 0,
                'cuda_memory_reserved_gb': 0,
                'cuda_device_name': 'N/A'
            }
    
    def log_metrics(self):
        """Log current metrics"""
        system_metrics = self.get_system_metrics()
        pytorch_metrics = self.get_pytorch_metrics()
        
        combined_metrics = {**system_metrics, **pytorch_metrics}
        self.metrics.append(combined_metrics)
        
        logger.info(f"System Metrics: CPU={system_metrics['cpu_percent']:.1f}%, "
                   f"Memory={system_metrics['memory_percent']:.1f}%, "
                   f"Available={system_metrics['memory_available_gb']:.1f}GB")
        
        if pytorch_metrics['cuda_available']:
            logger.info(f"CUDA Metrics: Memory={pytorch_metrics['cuda_memory_allocated_gb']:.2f}GB, "
                       f"Reserved={pytorch_metrics['cuda_memory_reserved_gb']:.2f}GB")
    
    def save_metrics(self, filepath: str):
        """Save metrics to file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {filepath}")


def benchmark_model_performance():
    """Benchmark model performance with different configurations"""
    logger.info("🚀 Starting model performance benchmark...")
    
    # Test configurations
    configs = [
        ("tiny", ConfigManager.get_small_config()),
        ("default", ConfigManager.get_default_config())
    ]
    
    results = {}
    
    for config_name, config in configs:
        logger.info(f"Testing {config_name} configuration...")
        
        try:
            # Create model
            model = OCRNativeLLM(config)
            model_info = model.get_model_info()
            
            # Test inputs
            test_inputs = {
                'text': 'Hello, OCR-native world!',
                'speech': [0.1, 0.2, 0.3, 0.4, 0.5],
                'image': None
            }
            
            # Warmup
            for _ in range(3):
                _ = model(test_inputs)
            
            # Benchmark forward pass
            times = []
            for i in range(10):
                start_time = time.time()
                outputs = model(test_inputs)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            results[config_name] = {
                'config': {
                    'd_model': config.d_model,
                    'n_layers': config.n_layers,
                    'n_heads': config.n_heads,
                    'vocab_size': config.vocab_size
                },
                'model_info': model_info,
                'performance': {
                    'avg_forward_time': avg_time,
                    'min_forward_time': min_time,
                    'max_forward_time': max_time,
                    'throughput_per_second': 1.0 / avg_time
                }
            }
            
            logger.info(f"{config_name} - Avg: {avg_time:.4f}s, "
                       f"Min: {min_time:.4f}s, Max: {max_time:.4f}s, "
                       f"Throughput: {1.0/avg_time:.2f} ops/s")
            
        except Exception as e:
            logger.error(f"Benchmark failed for {config_name}", exception=e)
            results[config_name] = {'error': str(e)}
    
    return results


def monitor_system_continuously(duration_seconds: int = 60):
    """Monitor system continuously for specified duration"""
    logger.info(f"🔍 Starting continuous system monitoring for {duration_seconds} seconds...")
    
    monitor = SystemMonitor()
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration_seconds:
            monitor.log_metrics()
            time.sleep(5)  # Log every 5 seconds
        
        # Save final metrics
        monitor.save_metrics("logs/system_metrics.json")
        
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error("Monitoring failed", exception=e)
    finally:
        monitor.save_metrics("logs/system_metrics.json")


def main():
    """Main performance monitoring function"""
    print("📊 OCR-Native LLM Performance Monitor")
    print("=" * 50)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Run benchmark
    benchmark_results = benchmark_model_performance()
    
    # Save benchmark results
    with open("logs/benchmark_results.json", 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    # Print summary
    print("\n📈 Benchmark Results Summary:")
    print("-" * 30)
    for config_name, result in benchmark_results.items():
        if 'error' not in result:
            perf = result['performance']
            print(f"{config_name.upper()}:")
            print(f"  Parameters: {result['model_info']['total_parameters']:,}")
            print(f"  Model Size: {result['model_info']['model_size_mb']:.2f} MB")
            print(f"  Avg Time: {perf['avg_forward_time']:.4f}s")
            print(f"  Throughput: {perf['throughput_per_second']:.2f} ops/s")
            print()
        else:
            print(f"{config_name.upper()}: ERROR - {result['error']}")
    
    # Run continuous monitoring
    print("🔍 Starting continuous monitoring (60 seconds)...")
    monitor_system_continuously(60)
    
    print("\n✅ Performance monitoring completed!")
    print("Check logs/ directory for detailed results.")


if __name__ == "__main__":
    main()