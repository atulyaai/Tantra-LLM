"""
Performance Benchmarking for OCR-Native LLM
Comprehensive performance testing and analysis
"""

import torch
import time
import psutil
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import threading
import queue

from src.architectures.transformer_variants import OCRNativeTransformer, TransformerVariantConfig
from src.configs.ocr_config import ConfigManager
from src.utils.error_handler import logger


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run"""
    test_name: str
    model_variant: str
    model_size: str
    duration: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    accuracy: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""
    suite_name: str
    timestamp: float
    results: List[BenchmarkResult]
    system_info: Dict[str, Any]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self, duration: float = 1.0):
        self.duration = duration
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> List[Dict[str, float]]:
        """Stop monitoring and return metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.metrics
    
    def _monitor_loop(self):
        """Monitoring loop"""
        start_time = time.time()
        while self.monitoring and (time.time() - start_time) < self.duration:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            self.metrics.append({
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_used_gb': memory.used / (1024**3)
            })
            
            time.sleep(0.1)  # Sample every 100ms


class OCRNativeBenchmark:
    """Comprehensive benchmarking suite for OCR-Native LLM"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Benchmark configurations
        self.model_sizes = ["small", "default", "large"]
        self.variants = ["standard", "mamba", "hybrid", "memory_enhanced"]
        
        # Test datasets
        self.test_inputs = self._create_test_inputs()
        
        logger.info("Initialized OCR-Native Benchmark suite")
    
    def _create_test_inputs(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create test input datasets"""
        return {
            "text_only": [
                {"text": "Hello, how are you today?"},
                {"text": "Explain the concept of artificial intelligence."},
                {"text": "What is the capital of France?"},
                {"text": "Write a short story about a robot."},
                {"text": "Calculate 2 + 2 and explain the process."}
            ],
            "multimodal": [
                {"text": "Hello", "speech": np.random.randn(1600), "image": None},
                {"text": "Describe this image", "speech": None, "image": self._create_test_image()},
                {"text": "Process this audio", "speech": np.random.randn(3200), "image": None},
                {"text": "Analyze everything", "speech": np.random.randn(2400), "image": self._create_test_image()}
            ],
            "long_text": [
                {"text": " ".join(["This is a long text input"] * 100)},
                {"text": " ".join(["Another long text for testing"] * 200)},
                {"text": " ".join(["Even longer text input"] * 500)}
            ]
        }
    
    def _create_test_image(self) -> 'Image.Image':
        """Create a test image"""
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (224, 224), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Test Image", fill='black')
        return img
    
    def run_comprehensive_benchmark(self) -> BenchmarkSuite:
        """Run comprehensive benchmark suite"""
        logger.info("Starting comprehensive benchmark suite")
        
        results = []
        start_time = time.time()
        
        # Test all model sizes and variants
        for model_size in self.model_sizes:
            for variant in self.variants:
                try:
                    logger.info(f"Testing {model_size} model with {variant} variant")
                    
                    # Initialize model
                    model = self._create_model(model_size, variant)
                    
                    # Run performance tests
                    perf_results = self._run_performance_tests(model, model_size, variant)
                    results.extend(perf_results)
                    
                    # Run accuracy tests
                    acc_results = self._run_accuracy_tests(model, model_size, variant)
                    results.extend(acc_results)
                    
                    # Run memory tests
                    mem_results = self._run_memory_tests(model, model_size, variant)
                    results.extend(mem_results)
                    
                except Exception as e:
                    logger.error(f"Error testing {model_size}-{variant}: {e}")
                    results.append(BenchmarkResult(
                        test_name="initialization",
                        model_variant=variant,
                        model_size=model_size,
                        duration=0,
                        memory_usage=0,
                        cpu_usage=0,
                        throughput=0,
                        error=str(e)
                    ))
        
        # Generate summary
        summary = self._generate_summary(results)
        
        # Create benchmark suite
        suite = BenchmarkSuite(
            suite_name="OCR-Native Comprehensive Benchmark",
            timestamp=time.time(),
            results=results,
            system_info=self._get_system_info(),
            summary=summary
        )
        
        # Save results
        self._save_benchmark_suite(suite)
        
        logger.info(f"Benchmark suite completed in {time.time() - start_time:.2f}s")
        return suite
    
    def _create_model(self, model_size: str, variant: str) -> OCRNativeTransformer:
        """Create model with specified size and variant"""
        if model_size == "small":
            config = ConfigManager.get_small_config()
        elif model_size == "large":
            config = ConfigManager.get_large_config()
        else:
            config = ConfigManager.get_default_config()
        
        # Convert to transformer variant config
        variant_config = TransformerVariantConfig(**config.__dict__)
        variant_config.variant = variant
        
        return OCRNativeTransformer(variant_config)
    
    def _run_performance_tests(self, model: OCRNativeTransformer, model_size: str, variant: str) -> List[BenchmarkResult]:
        """Run performance tests"""
        results = []
        
        # Test different input types
        for test_type, inputs_list in self.test_inputs.items():
            for i, inputs in enumerate(inputs_list):
                try:
                    # Warmup
                    with torch.no_grad():
                        _ = model(inputs)
                    
                    # Performance test
                    start_time = time.time()
                    monitor = PerformanceMonitor(duration=1.0)
                    monitor.start_monitoring()
                    
                    with torch.no_grad():
                        outputs = model(inputs)
                    
                    duration = time.time() - start_time
                    metrics = monitor.stop_monitoring()
                    
                    # Calculate metrics
                    avg_cpu = np.mean([m['cpu_percent'] for m in metrics]) if metrics else 0
                    avg_memory = np.mean([m['memory_percent'] for m in metrics]) if metrics else 0
                    throughput = 1.0 / duration if duration > 0 else 0
                    
                    results.append(BenchmarkResult(
                        test_name=f"performance_{test_type}_{i}",
                        model_variant=variant,
                        model_size=model_size,
                        duration=duration,
                        memory_usage=avg_memory,
                        cpu_usage=avg_cpu,
                        throughput=throughput,
                        metadata={
                            "input_type": test_type,
                            "input_size": len(str(inputs)),
                            "output_shape": {k: list(v.shape) if hasattr(v, 'shape') else str(type(v)) for k, v in outputs.items()}
                        }
                    ))
                    
                except Exception as e:
                    results.append(BenchmarkResult(
                        test_name=f"performance_{test_type}_{i}",
                        model_variant=variant,
                        model_size=model_size,
                        duration=0,
                        memory_usage=0,
                        cpu_usage=0,
                        throughput=0,
                        error=str(e)
                    ))
        
        return results
    
    def _run_accuracy_tests(self, model: OCRNativeTransformer, model_size: str, variant: str) -> List[BenchmarkResult]:
        """Run accuracy tests (simplified)"""
        results = []
        
        # Test text generation consistency
        test_input = {"text": "Hello world"}
        
        try:
            with torch.no_grad():
                outputs = model(test_input)
            
            # Simple accuracy metric (would be more sophisticated in practice)
            text_logits = outputs.get('text_logits', torch.zeros(1, 1, 1000))
            confidence = torch.softmax(text_logits, dim=-1).max().item()
            
            results.append(BenchmarkResult(
                test_name="accuracy_consistency",
                model_variant=variant,
                model_size=model_size,
                duration=0,
                memory_usage=0,
                cpu_usage=0,
                throughput=0,
                accuracy=confidence,
                metadata={"test_type": "text_generation_consistency"}
            ))
            
        except Exception as e:
            results.append(BenchmarkResult(
                test_name="accuracy_consistency",
                model_variant=variant,
                model_size=model_size,
                duration=0,
                memory_usage=0,
                cpu_usage=0,
                throughput=0,
                accuracy=0,
                error=str(e)
            ))
        
        return results
    
    def _run_memory_tests(self, model: OCRNativeTransformer, model_size: str, variant: str) -> List[BenchmarkResult]:
        """Run memory usage tests"""
        results = []
        
        try:
            # Test memory usage during forward pass
            initial_memory = psutil.virtual_memory().used / (1024**3)
            
            with torch.no_grad():
                outputs = model({"text": "Memory test"})
            
            peak_memory = psutil.virtual_memory().used / (1024**3)
            memory_increase = peak_memory - initial_memory
            
            results.append(BenchmarkResult(
                test_name="memory_usage",
                model_variant=variant,
                model_size=model_size,
                duration=0,
                memory_usage=memory_increase,
                cpu_usage=0,
                throughput=0,
                metadata={
                    "initial_memory_gb": initial_memory,
                    "peak_memory_gb": peak_memory,
                    "memory_increase_gb": memory_increase
                }
            ))
            
        except Exception as e:
            results.append(BenchmarkResult(
                test_name="memory_usage",
                model_variant=variant,
                model_size=model_size,
                duration=0,
                memory_usage=0,
                cpu_usage=0,
                throughput=0,
                error=str(e)
            ))
        
        return results
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate benchmark summary"""
        successful_results = [r for r in results if r.error is None]
        
        if not successful_results:
            return {"error": "No successful benchmark results"}
        
        # Performance summary
        performance_results = [r for r in successful_results if r.test_name.startswith("performance")]
        if performance_results:
            avg_throughput = np.mean([r.throughput for r in performance_results])
            avg_duration = np.mean([r.duration for r in performance_results])
            avg_memory = np.mean([r.memory_usage for r in performance_results])
        else:
            avg_throughput = avg_duration = avg_memory = 0
        
        # Accuracy summary
        accuracy_results = [r for r in successful_results if r.accuracy is not None]
        avg_accuracy = np.mean([r.accuracy for r in accuracy_results]) if accuracy_results else 0
        
        # Model comparison
        model_stats = {}
        for result in successful_results:
            key = f"{result.model_size}_{result.model_variant}"
            if key not in model_stats:
                model_stats[key] = {"count": 0, "total_duration": 0, "total_throughput": 0}
            model_stats[key]["count"] += 1
            model_stats[key]["total_duration"] += result.duration
            model_stats[key]["total_throughput"] += result.throughput
        
        # Calculate averages
        for key in model_stats:
            stats = model_stats[key]
            stats["avg_duration"] = stats["total_duration"] / stats["count"]
            stats["avg_throughput"] = stats["total_throughput"] / stats["count"]
        
        return {
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "failed_tests": len(results) - len(successful_results),
            "average_throughput": avg_throughput,
            "average_duration": avg_duration,
            "average_memory_usage": avg_memory,
            "average_accuracy": avg_accuracy,
            "model_comparison": model_stats,
            "best_performing_model": max(model_stats.keys(), key=lambda k: model_stats[k]["avg_throughput"]) if model_stats else None
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import sys
        import platform
        return {
            "python_version": f"{sys.version}",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "platform": platform.platform(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _save_benchmark_suite(self, suite: BenchmarkSuite):
        """Save benchmark suite to files"""
        timestamp = datetime.fromtimestamp(suite.timestamp).strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.output_dir / f"benchmark_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(suite.to_dict(), f, indent=2)
        
        # Save summary report
        report_file = self.output_dir / f"benchmark_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(f"OCR-Native LLM Benchmark Report\n")
            f.write(f"Generated: {datetime.fromtimestamp(suite.timestamp)}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Tests: {suite.summary['total_tests']}\n")
            f.write(f"Successful: {suite.summary['successful_tests']}\n")
            f.write(f"Failed: {suite.summary['failed_tests']}\n")
            f.write(f"Average Throughput: {suite.summary['average_throughput']:.2f} ops/s\n")
            f.write(f"Average Duration: {suite.summary['average_duration']:.4f}s\n")
            f.write(f"Average Memory Usage: {suite.summary['average_memory_usage']:.2f}%\n")
            f.write(f"Average Accuracy: {suite.summary['average_accuracy']:.4f}\n\n")
            
            f.write("Model Comparison:\n")
            f.write("-" * 30 + "\n")
            for model, stats in suite.summary['model_comparison'].items():
                f.write(f"{model}:\n")
                f.write(f"  Tests: {stats['count']}\n")
                f.write(f"  Avg Duration: {stats['avg_duration']:.4f}s\n")
                f.write(f"  Avg Throughput: {stats['avg_throughput']:.2f} ops/s\n")
        
        logger.info(f"Benchmark results saved to {self.output_dir}")
    
    def generate_visualizations(self, suite: BenchmarkSuite):
        """Generate visualization plots"""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Performance comparison plot
            self._plot_performance_comparison(suite)
            
            # Memory usage plot
            self._plot_memory_usage(suite)
            
            # Throughput over time plot
            self._plot_throughput_trends(suite)
            
            logger.info("Visualizations generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _plot_performance_comparison(self, suite: BenchmarkSuite):
        """Plot performance comparison"""
        performance_results = [r for r in suite.results if r.test_name.startswith("performance") and r.error is None]
        
        if not performance_results:
            return
        
        # Group by model variant
        variants = {}
        for result in performance_results:
            if result.model_variant not in variants:
                variants[result.model_variant] = []
            variants[result.model_variant].append(result.throughput)
        
        # Create box plot
        plt.figure(figsize=(12, 8))
        data = [variants[v] for v in variants.keys()]
        labels = list(variants.keys())
        
        plt.boxplot(data, labels=labels)
        plt.title('Performance Comparison by Model Variant')
        plt.xlabel('Model Variant')
        plt.ylabel('Throughput (ops/s)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        timestamp = datetime.fromtimestamp(suite.timestamp).strftime("%Y%m%d_%H%M%S")
        plt.savefig(self.output_dir / f"performance_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage(self, suite: BenchmarkSuite):
        """Plot memory usage"""
        memory_results = [r for r in suite.results if r.test_name == "memory_usage" and r.error is None]
        
        if not memory_results:
            return
        
        # Group by model size and variant
        model_data = {}
        for result in memory_results:
            key = f"{result.model_size}_{result.model_variant}"
            if key not in model_data:
                model_data[key] = []
            model_data[key].append(result.memory_usage)
        
        # Create bar plot
        plt.figure(figsize=(14, 8))
        models = list(model_data.keys())
        memory_values = [np.mean(model_data[m]) for m in models]
        
        bars = plt.bar(models, memory_values)
        plt.title('Memory Usage by Model Configuration')
        plt.xlabel('Model Configuration')
        plt.ylabel('Memory Usage (%)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, memory_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        timestamp = datetime.fromtimestamp(suite.timestamp).strftime("%Y%m%d_%H%M%S")
        plt.savefig(self.output_dir / f"memory_usage_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_throughput_trends(self, suite: BenchmarkSuite):
        """Plot throughput trends over time"""
        performance_results = [r for r in suite.results if r.test_name.startswith("performance") and r.error is None]
        
        if not performance_results:
            return
        
        # Sort by timestamp (using test index as proxy)
        performance_results.sort(key=lambda x: x.test_name)
        
        # Group by variant
        variants = {}
        for i, result in enumerate(performance_results):
            if result.model_variant not in variants:
                variants[result.model_variant] = {'x': [], 'y': []}
            variants[result.model_variant]['x'].append(i)
            variants[result.model_variant]['y'].append(result.throughput)
        
        # Create line plot
        plt.figure(figsize=(12, 8))
        for variant, data in variants.items():
            plt.plot(data['x'], data['y'], marker='o', label=variant, linewidth=2)
        
        plt.title('Throughput Trends by Model Variant')
        plt.xlabel('Test Index')
        plt.ylabel('Throughput (ops/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        timestamp = datetime.fromtimestamp(suite.timestamp).strftime("%Y%m%d_%H%M%S")
        plt.savefig(self.output_dir / f"throughput_trends_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()


# Convenience functions
def run_quick_benchmark() -> BenchmarkSuite:
    """Run a quick benchmark with small models only"""
    benchmark = OCRNativeBenchmark()
    
    # Override model sizes for quick test
    benchmark.model_sizes = ["small"]
    benchmark.variants = ["standard", "mamba"]
    
    return benchmark.run_comprehensive_benchmark()


def compare_variants() -> Dict[str, Any]:
    """Quick comparison of different model variants"""
    benchmark = OCRNativeBenchmark()
    suite = run_quick_benchmark()
    
    # Extract comparison data
    comparison = {}
    for result in suite.results:
        if result.error is None and result.test_name.startswith("performance"):
            key = f"{result.model_size}_{result.model_variant}"
            if key not in comparison:
                comparison[key] = []
            comparison[key].append(result.throughput)
    
    # Calculate averages
    for key in comparison:
        comparison[key] = np.mean(comparison[key])
    
    return comparison