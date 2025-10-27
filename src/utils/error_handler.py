"""
Comprehensive Error Handling and Logging System
For OCR-Native LLM
"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Optional, Dict
from pathlib import Path
import json


class OCRNativeLogger:
    """Centralized logging system for OCR-Native LLM"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger("ocr_native")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / "ocr_native.log")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.FileHandler(self.log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        self.logger.addHandler(error_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception details"""
        if exception:
            self.logger.error(f"{message}: {str(exception)}", extra=kwargs)
            self.logger.error(f"Traceback: {traceback.format_exc()}", extra=kwargs)
        else:
            self.logger.error(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)


# Global logger instance
logger = OCRNativeLogger()


class ErrorHandler:
    """Comprehensive error handling decorators and utilities"""
    
    @staticmethod
    def handle_errors(
        default_return: Any = None,
        log_error: bool = True,
        reraise: bool = False,
        error_message: str = "An error occurred"
    ):
        """Decorator to handle errors gracefully"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if log_error:
                        logger.error(f"{error_message} in {func.__name__}", exception=e)
                    
                    if reraise:
                        raise
                    
                    return default_return
            return wrapper
        return decorator
    
    @staticmethod
    def validate_inputs(**validators):
        """Decorator to validate function inputs"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Validate inputs based on validators
                for param_name, validator in validators.items():
                    if param_name in kwargs:
                        if not validator(kwargs[param_name]):
                            raise ValueError(f"Invalid input for {param_name}")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
        """Decorator to retry function on failure"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries:
                            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                            time.sleep(delay)
                        else:
                            logger.error(f"All {max_retries + 1} attempts failed", exception=e)
                
                raise last_exception
            return wrapper
        return decorator


class PerformanceMonitor:
    """Performance monitoring and profiling utilities"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        if operation not in self.start_times:
            logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        logger.info(f"Operation '{operation}' completed in {duration:.4f}s")
        
        return duration
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics summary"""
        summary = {}
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    'count': len(times),
                    'total': sum(times),
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
        return summary
    
    def save_metrics(self, filepath: str):
        """Save metrics to file"""
        metrics_data = {
            'timestamp': time.time(),
            'metrics': self.get_metrics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Performance metrics saved to {filepath}")


# Global performance monitor
performance_monitor = PerformanceMonitor()


def log_performance(operation_name: str):
    """Decorator to log performance of operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            performance_monitor.start_timer(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                performance_monitor.end_timer(operation_name)
        return wrapper
    return decorator


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class ModelError(Exception):
    """Custom exception for model-related errors"""
    pass


class OCRProcessingError(Exception):
    """Custom exception for OCR processing errors"""
    pass


def validate_model_config(config: Dict[str, Any]) -> bool:
    """Validate model configuration"""
    required_keys = ['d_model', 'n_layers', 'n_heads', 'vocab_size']
    
    for key in required_keys:
        if key not in config:
            raise ValidationError(f"Missing required config key: {key}")
        
        if not isinstance(config[key], (int, float)):
            raise ValidationError(f"Config key '{key}' must be numeric")
        
        if config[key] <= 0:
            raise ValidationError(f"Config key '{key}' must be positive")
    
    return True


def validate_input_data(inputs: Dict[str, Any]) -> bool:
    """Validate input data for OCR processing"""
    if not isinstance(inputs, dict):
        raise ValidationError("Inputs must be a dictionary")
    
    valid_types = ['text', 'speech', 'image']
    has_valid_input = False
    
    for input_type in valid_types:
        if input_type in inputs and inputs[input_type] is not None:
            has_valid_input = True
            break
    
    if not has_valid_input:
        raise ValidationError("At least one valid input type must be provided")
    
    return True


def safe_execute(func: Callable, *args, **kwargs) -> tuple[Any, bool]:
    """Safely execute a function and return result with success status"""
    try:
        result = func(*args, **kwargs)
        return result, True
    except Exception as e:
        logger.error(f"Safe execution failed for {func.__name__}", exception=e)
        return None, False


# Example usage and testing
if __name__ == "__main__":
    # Test the error handling system
    logger.info("Testing error handling system...")
    
    @ErrorHandler.handle_errors(default_return="Error occurred", log_error=True)
    def test_function(value: int):
        if value < 0:
            raise ValueError("Value must be positive")
        return value * 2
    
    # Test successful execution
    result, success = safe_execute(test_function, 5)
    print(f"Test 1 - Success: {success}, Result: {result}")
    
    # Test error handling
    result, success = safe_execute(test_function, -1)
    print(f"Test 2 - Success: {success}, Result: {result}")
    
    # Test performance monitoring
    @log_performance("test_operation")
    def slow_operation():
        time.sleep(0.1)
        return "completed"
    
    slow_operation()
    
    # Print metrics
    metrics = performance_monitor.get_metrics()
    print(f"Performance metrics: {metrics}")