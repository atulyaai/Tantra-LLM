"""
Performance Optimization System
Provides async operations, caching, and performance monitoring.
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Union, Type
from functools import wraps, lru_cache
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import pickle

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


class PerformanceMonitor:
    """Monitors and tracks performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history = max_history
        self.lock = threading.Lock()
    
    def record_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric."""
        with self.lock:
            self.metrics_history.append(metric)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history = self.metrics_history[-self.max_history:]
    
    def get_average_duration(self, operation_name: str) -> float:
        """Get average duration for an operation."""
        with self.lock:
            durations = [
                m.duration for m in self.metrics_history
                if m.operation_name == operation_name and m.success
            ]
            return sum(durations) / len(durations) if durations else 0.0
    
    def get_success_rate(self, operation_name: str) -> float:
        """Get success rate for an operation."""
        with self.lock:
            operations = [m for m in self.metrics_history if m.operation_name == operation_name]
            if not operations:
                return 0.0
            successful = sum(1 for m in operations if m.success)
            return successful / len(operations)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self.lock:
            if not self.metrics_history:
                return {"status": "no_data"}
            
            total_operations = len(self.metrics_history)
            successful_operations = sum(1 for m in self.metrics_history if m.success)
            
            # Group by operation
            operation_stats = {}
            for metric in self.metrics_history:
                if metric.operation_name not in operation_stats:
                    operation_stats[metric.operation_name] = {
                        "count": 0,
                        "total_duration": 0.0,
                        "successful": 0,
                        "failed": 0
                    }
                
                stats = operation_stats[metric.operation_name]
                stats["count"] += 1
                stats["total_duration"] += metric.duration
                if metric.success:
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1
            
            # Calculate averages
            for stats in operation_stats.values():
                stats["average_duration"] = stats["total_duration"] / stats["count"]
                stats["success_rate"] = stats["successful"] / stats["count"]
            
            return {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "overall_success_rate": successful_operations / total_operations,
                "operation_stats": operation_stats
            }


class CacheManager:
    """Manages caching for expensive operations."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        key_data = {
            "func_name": func_name,
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_str = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(key_str).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            if time.time() - entry["timestamp"] > self.ttl_seconds:
                del self.cache[key]
                return None
            
            return entry["value"]
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self.lock:
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
                del self.cache[oldest_key]
            
            self.cache[key] = {
                "value": value,
                "timestamp": time.time()
            }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds
            }


class AsyncExecutor:
    """Manages async execution of operations."""
    
    def __init__(self, max_workers: int = 4):
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.loop = None
        self.loop_thread = None
    
    def start_event_loop(self) -> None:
        """Start the event loop in a separate thread."""
        if self.loop is None:
            def run_loop():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()
            
            self.loop_thread = threading.Thread(target=run_loop, daemon=True)
            self.loop_thread.start()
    
    def run_async(self, coro) -> Any:
        """Run an async coroutine."""
        if self.loop is None:
            self.start_event_loop()
        
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()
    
    def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Run a function in a thread pool."""
        future = self.thread_pool.submit(func, *args, **kwargs)
        return future.result()
    
    def run_in_process(self, func: Callable, *args, **kwargs) -> Any:
        """Run a function in a process pool."""
        future = self.process_pool.submit(func, *args, **kwargs)
        return future.result()
    
    def shutdown(self) -> None:
        """Shutdown all executors."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)


class PerformanceOptimizer:
    """Main performance optimization system."""
    
    def __init__(self, enable_caching: bool = True, enable_monitoring: bool = True):
        self.monitor = PerformanceMonitor() if enable_monitoring else None
        self.cache = CacheManager() if enable_caching else None
        self.executor = AsyncExecutor()
        self.enable_caching = enable_caching
        self.enable_monitoring = enable_monitoring
    
    def cached(self, ttl_seconds: int = 3600):
        """Decorator for caching function results."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_caching or self.cache is None:
                    return func(*args, **kwargs)
                
                # Generate cache key
                key = self.cache._generate_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_result = self.cache.get(key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.cache.set(key, result)
                logger.debug(f"Cached result for {func.__name__}")
                return result
            
            return wrapper
        return decorator
    
    def monitored(self, operation_name: str = None):
        """Decorator for monitoring function performance."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_monitoring or self.monitor is None:
                    return func(*args, **kwargs)
                
                op_name = operation_name or func.__name__
                start_time = time.time()
                success = True
                error_message = None
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error_message = str(e)
                    raise
                finally:
                    end_time = time.time()
                    metric = PerformanceMetrics(
                        operation_name=op_name,
                        start_time=start_time,
                        end_time=end_time,
                        duration=end_time - start_time,
                        success=success,
                        error_message=error_message
                    )
                    self.monitor.record_metric(metric)
            
            return wrapper
        return decorator
    
    def async_operation(self, operation_name: str = None):
        """Decorator for async operations."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or func.__name__
                
                # Run in thread pool for CPU-bound operations
                if self._is_cpu_bound(func):
                    return self.executor.run_in_thread(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def _is_cpu_bound(self, func: Callable) -> bool:
        """Determine if a function is CPU-bound."""
        # Simple heuristic based on function name
        cpu_bound_keywords = [
            "encode", "decode", "process", "compute", "calculate",
            "transform", "convert", "generate", "train", "optimize"
        ]
        return any(keyword in func.__name__.lower() for keyword in cpu_bound_keywords)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {}
        
        if self.monitor:
            report["monitoring"] = self.monitor.get_performance_summary()
        
        if self.cache:
            report["caching"] = self.cache.get_stats()
        
        return report
    
    def optimize_memory_usage(self) -> None:
        """Optimize memory usage by clearing caches and garbage collection."""
        import gc
        
        if self.cache:
            self.cache.clear()
        
        gc.collect()
        logger.info("Memory optimization completed")
    
    def shutdown(self) -> None:
        """Shutdown the performance optimizer."""
        self.executor.shutdown()
        logger.info("Performance optimizer shutdown completed")


# Global performance optimizer instance
global_optimizer = PerformanceOptimizer()


def performance_optimized(operation_name: str = None, enable_caching: bool = True, enable_monitoring: bool = True):
    """Convenience decorator for performance optimization."""
    def decorator(func: Callable) -> Callable:
        # Apply caching if enabled
        if enable_caching:
            func = global_optimizer.cached()(func)
        
        # Apply monitoring if enabled
        if enable_monitoring:
            func = global_optimizer.monitored(operation_name)(func)
        
        # Apply async optimization
        func = global_optimizer.async_operation(operation_name)(func)
        
        return func
    return decorator


class BatchProcessor:
    """Processes multiple items in batches for better performance."""
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def process_batch(self, items: List[Any], process_func: Callable) -> List[Any]:
        """Process items in batches."""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = self.executor.map(process_func, batch)
            results.extend(batch_results)
        
        return results
    
    def process_batch_async(self, items: List[Any], process_func: Callable) -> List[Any]:
        """Process items in batches asynchronously."""
        async def process_async():
            tasks = []
            for i in range(0, len(items), self.batch_size):
                batch = items[i:i + self.batch_size]
                task = asyncio.create_task(self._process_batch_async(batch, process_func))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return [item for batch_result in results for item in batch_result]
        
        return asyncio.run(process_async())
    
    async def _process_batch_async(self, batch: List[Any], process_func: Callable) -> List[Any]:
        """Process a single batch asynchronously."""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self.executor, 
            lambda: [process_func(item) for item in batch]
        )
        return results
    
    def shutdown(self) -> None:
        """Shutdown the batch processor."""
        self.executor.shutdown()


# Example usage and testing
if __name__ == "__main__":
    # Test performance optimization
    optimizer = PerformanceOptimizer()
    
    @optimizer.cached(ttl_seconds=60)
    @optimizer.monitored("test_operation")
    def expensive_operation(x: int) -> int:
        time.sleep(0.1)  # Simulate expensive operation
        return x * x
    
    # Test the operation
    result1 = expensive_operation(5)
    result2 = expensive_operation(5)  # Should be cached
    
    print(f"Result: {result1}")
    print(f"Cached result: {result2}")
    print(f"Performance report: {optimizer.get_performance_report()}")
    
    optimizer.shutdown()