"""
OCR-Native LLM Configuration System
Clean, modular configuration management
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import yaml
import os


@dataclass
class OCRNativeConfig:
    """Configuration for OCR-Native LLM - Bigger Model"""
    # Model architecture - BIGGER MODEL
    d_model: int = 1024  # Increased from 512
    n_layers: int = 24   # Increased from 12
    n_heads: int = 16    # Increased from 8
    d_ff: int = 4096     # Increased from 2048
    vocab_size: int = 100000  # Increased from 50000
    max_seq_length: int = 16384  # Increased from 8192
    
    # OCR-specific settings
    ocr_image_width: int = 1024
    ocr_image_height: int = 1024
    ocr_font_size: int = 14
    ocr_precision: int = 8
    ocr_compression_ratio: float = 0.7
    
    # Memory and context - BIGGER
    memory_window_size: int = 100000  # Increased from 50000
    ocr_memory_bank_size: int = 2000  # Increased from 1000
    context_retention: float = 0.95
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 4  # Reduced due to bigger model
    gradient_accumulation_steps: int = 8
    
    # OCR processing
    text_to_ocr_enabled: bool = True
    speech_to_ocr_enabled: bool = True
    image_ocr_enabled: bool = True
    
    # Performance
    use_cuda: bool = False
    mixed_precision: bool = False
    gradient_checkpointing: bool = True


class ConfigManager:
    """Manages configuration loading and saving"""
    
    @staticmethod
    def load_config(config_path: str) -> OCRNativeConfig:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return OCRNativeConfig(**config_dict)
    
    @staticmethod
    def save_config(config: OCRNativeConfig, config_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'd_ff': config.d_ff,
            'vocab_size': config.vocab_size,
            'max_seq_length': config.max_seq_length,
            'ocr_image_width': config.ocr_image_width,
            'ocr_image_height': config.ocr_image_height,
            'ocr_font_size': config.ocr_font_size,
            'ocr_precision': config.ocr_precision,
            'ocr_compression_ratio': config.ocr_compression_ratio,
            'memory_window_size': config.memory_window_size,
            'ocr_memory_bank_size': config.ocr_memory_bank_size,
            'context_retention': config.context_retention,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'gradient_accumulation_steps': config.gradient_accumulation_steps,
            'text_to_ocr_enabled': config.text_to_ocr_enabled,
            'speech_to_ocr_enabled': config.speech_to_ocr_enabled,
            'image_ocr_enabled': config.image_ocr_enabled,
            'use_cuda': config.use_cuda,
            'mixed_precision': config.mixed_precision,
            'gradient_checkpointing': config.gradient_checkpointing
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @staticmethod
    def get_default_config() -> OCRNativeConfig:
        """Get default configuration"""
        return OCRNativeConfig()
    
    @staticmethod
    def get_small_config() -> OCRNativeConfig:
        """Get small configuration for testing"""
        return OCRNativeConfig(
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=1024,
            vocab_size=10000,
            max_seq_length=2048,
            memory_window_size=10000,
            ocr_memory_bank_size=100
        )
    
    @staticmethod
    def get_large_config() -> OCRNativeConfig:
        """Get large configuration for production"""
        return OCRNativeConfig(
            d_model=2048,
            n_layers=48,
            n_heads=32,
            d_ff=8192,
            vocab_size=200000,
            max_seq_length=32768,
            memory_window_size=200000,
            ocr_memory_bank_size=5000
        )