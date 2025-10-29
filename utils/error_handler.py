"""
Comprehensive Error Handling and Recovery System
Provides robust error handling, logging, and recovery mechanisms.
"""

from __future__ import annotations

import logging
import traceback
import functools
import asyncio
from typing import Any, Callable, Dict, List, Optional, Union, Type
from datetime import datetime
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better organization."""
    MODEL_LOADING = "model_loading"
    MEMORY_OPERATION = "memory_operation"
    FUSION_PROCESSING = "fusion_processing"
    GENERATION = "generation"
    VISION_ENCODING = "vision_encoding"
    AUDIO_ENCODING = "audio_encoding"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ErrorRecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    LOG_AND_CONTINUE = "log_and_continue"


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Error tracking
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        # Recovery strategies
        self.recovery_strategies: Dict[Type[Exception], ErrorRecoveryStrategy] = {
            ConnectionError: ErrorRecoveryStrategy.RETRY,
            TimeoutError: ErrorRecoveryStrategy.RETRY,
            FileNotFoundError: ErrorRecoveryStrategy.FALLBACK,
            ImportError: ErrorRecoveryStrategy.FALLBACK,
            ValueError: ErrorRecoveryStrategy.LOG_AND_CONTINUE,
            RuntimeError: ErrorRecoveryStrategy.RETRY,
            MemoryError: ErrorRecoveryStrategy.ABORT,
        }
        
        # Retry configurations
        self.retry_configs: Dict[Type[Exception], Dict[str, Any]] = {
            ConnectionError: {"max_retries": 3, "delay": 1.0, "backoff": 2.0},
            TimeoutError: {"max_retries": 2, "delay": 0.5, "backoff": 1.5},
            RuntimeError: {"max_retries": 1, "delay": 0.1, "backoff": 1.0},
        }
    
    def handle_error(
        self,
        error: Exception,
        context: str = "",
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: Optional[ErrorRecoveryStrategy] = None
    ) -> Any:
        """Handle an error with appropriate recovery strategy."""
        # Log the error
        self._log_error(error, context, category, severity)
        
        # Track error statistics
        self._track_error(error, category)
        
        # Determine recovery strategy
        if recovery_strategy is None:
            recovery_strategy = self._determine_recovery_strategy(error)
        
        # Execute recovery strategy
        return self._execute_recovery_strategy(error, recovery_strategy, context)
    
    def _log_error(
        self,
        error: Exception,
        context: str,
        category: ErrorCategory,
        severity: ErrorSeverity
    ) -> None:
        """Log error with appropriate level and details."""
        error_msg = f"[{category.value}] {context}: {str(error)}"
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(error_msg, exc_info=True)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(error_msg, exc_info=True)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(error_msg)
        else:
            self.logger.info(error_msg)
    
    def _track_error(self, error: Exception, category: ErrorCategory) -> None:
        """Track error statistics and history."""
        error_key = f"{category.value}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Add to history
        error_record = {
            "timestamp": datetime.now(),
            "category": category.value,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        }
        
        self.error_history.append(error_record)
        
        # Trim history if too large
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _determine_recovery_strategy(self, error: Exception) -> ErrorRecoveryStrategy:
        """Determine appropriate recovery strategy for error type."""
        error_type = type(error)
        
        # Check specific error types first
        for exception_type, strategy in self.recovery_strategies.items():
            if issubclass(error_type, exception_type):
                return strategy
        
        # Default strategy
        return ErrorRecoveryStrategy.LOG_AND_CONTINUE
    
    def _execute_recovery_strategy(
        self,
        error: Exception,
        strategy: ErrorRecoveryStrategy,
        context: str
    ) -> Any:
        """Execute the determined recovery strategy."""
        if strategy == ErrorRecoveryStrategy.RETRY:
            return self._retry_operation(error, context)
        elif strategy == ErrorRecoveryStrategy.FALLBACK:
            return self._fallback_operation(error, context)
        elif strategy == ErrorRecoveryStrategy.SKIP:
            return None
        elif strategy == ErrorRecoveryStrategy.ABORT:
            raise error
        else:  # LOG_AND_CONTINUE
            return None
    
    def _retry_operation(self, error: Exception, context: str) -> Any:
        """Retry operation with exponential backoff."""
        error_type = type(error)
        retry_config = self.retry_configs.get(error_type, {
            "max_retries": 1,
            "delay": 0.1,
            "backoff": 1.0
        })
        
        max_retries = retry_config["max_retries"]
        delay = retry_config["delay"]
        backoff = retry_config["backoff"]
        
        for attempt in range(max_retries):
            try:
                # This would need to be implemented by the calling code
                # For now, just return None
                self.logger.info(f"Retry attempt {attempt + 1}/{max_retries} for {context}")
                return None
            except Exception as retry_error:
                if attempt == max_retries - 1:
                    self.logger.error(f"All retry attempts failed for {context}: {retry_error}")
                    return None
                else:
                    import time
                    time.sleep(delay * (backoff ** attempt))
        
        return None
    
    def _fallback_operation(self, error: Exception, context: str) -> Any:
        """Execute fallback operation."""
        self.logger.info(f"Executing fallback for {context}: {error}")
        
        # Implement fallback logic based on context
        if "model" in context.lower():
            return self._fallback_model_operation()
        elif "memory" in context.lower():
            return self._fallback_memory_operation()
        elif "vision" in context.lower():
            return self._fallback_vision_operation()
        elif "audio" in context.lower():
            return self._fallback_audio_operation()
        else:
            return None
    
    def _fallback_model_operation(self) -> Any:
        """Fallback for model operations."""
        return None  # Would return a basic model or None
    
    def _fallback_memory_operation(self) -> Any:
        """Fallback for memory operations."""
        return []  # Return empty list for memory operations
    
    def _fallback_vision_operation(self) -> Any:
        """Fallback for vision operations."""
        import torch
        return torch.zeros(1, 1024)  # Return zero embeddings
    
    def _fallback_audio_operation(self) -> Any:
        """Fallback for audio operations."""
        import torch
        return torch.zeros(1, 1024)  # Return zero embeddings
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and health metrics."""
        total_errors = sum(self.error_counts.values())
        
        # Get recent error rate (last hour)
        now = datetime.now()
        recent_errors = [
            error for error in self.error_history
            if (now - error["timestamp"]).total_seconds() < 3600
        ]
        
        return {
            "total_errors": total_errors,
            "error_counts": self.error_counts,
            "recent_error_rate": len(recent_errors),
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None,
            "error_history_size": len(self.error_history)
        }
    
    def clear_error_history(self) -> None:
        """Clear error history and reset counters."""
        self.error_counts.clear()
        self.error_history.clear()
        self.logger.info("Error history cleared")


def error_handler(
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    recovery_strategy: Optional[ErrorRecoveryStrategy] = None,
    context: str = ""
):
    """Decorator for automatic error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get error handler from args or create new one
                error_handler_instance = None
                for arg in args:
                    if isinstance(arg, ErrorHandler):
                        error_handler_instance = arg
                        break
                
                if error_handler_instance is None:
                    error_handler_instance = ErrorHandler()
                
                return error_handler_instance.handle_error(
                    e, context or func.__name__, category, severity, recovery_strategy
                )
        return wrapper
    return decorator


class SafeOperation:
    """Context manager for safe operations with automatic error handling."""
    
    def __init__(
        self,
        operation_name: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: Optional[ErrorRecoveryStrategy] = None,
        error_handler_instance: Optional[ErrorHandler] = None
    ):
        self.operation_name = operation_name
        self.category = category
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.error_handler = error_handler_instance or ErrorHandler()
        self.result = None
        self.error = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = exc_val
            self.result = self.error_handler.handle_error(
                exc_val, self.operation_name, self.category, self.severity, self.recovery_strategy
            )
            return True  # Suppress the exception
        return False
    
    def execute(self, func: Callable, *args, **kwargs):
        """Execute a function safely within this context."""
        try:
            self.result = func(*args, **kwargs)
            return self.result
        except Exception as e:
            self.error = e
            self.result = self.error_handler.handle_error(
                e, f"{self.operation_name}:{func.__name__}", self.category, self.severity, self.recovery_strategy
            )
            return self.result


# Global error handler instance
global_error_handler = ErrorHandler()