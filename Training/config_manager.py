"""
Dynamic Configuration Management System for Tantra LLM
Handles model configurations, paths, and settings dynamically
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str
    architecture: str
    d_model: int
    n_layers: int
    d_state: int
    d_conv: int
    dropout: float
    vocab_size: int = 50000
    max_seq_len: int = 2048
    
    # Additional model-specific parameters
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}

class ConfigManager:
    """Dynamic configuration management system"""
    
    def __init__(self, config_dir: str = "Config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configurations
        self.default_configs = {
            "mamba": {
                "architecture": "mamba",
                "d_model": 512,
                "n_layers": 8,
                "d_state": 32,
                "d_conv": 4,
                "dropout": 0.1,
                "vocab_size": 50000,
                "max_seq_len": 2048
            },
            "mamba_multimodal": {
                "architecture": "mamba_multimodal",
                "d_model": 768,
                "n_layers": 16,
                "d_state": 64,
                "d_conv": 4,
                "dropout": 0.1,
                "vocab_size": 100000,
                "max_seq_len": 4096,
                "extra_params": {
                    "num_experts": 8,
                    "expert_capacity": 64,
                    "quantization_bits": 8,
                    "pruning_ratio": 0.1,
                    "audio_dim": 128,
                    "vision_dim": 512
                }
            },
            "ocr_native": {
                "architecture": "ocr_native",
                "d_model": 512,
                "n_layers": 12,
                "d_state": 64,
                "d_conv": 4,
                "dropout": 0.1,
                "vocab_size": 50000,
                "max_seq_len": 2048,
                "extra_params": {
                    "ocr_loss_weight": 0.3,
                    "text_loss_weight": 0.7,
                    "ocr_precision": 8
                }
            }
        }
        
        # Load existing configurations
        self.configs = self._load_configs()
    
    def _load_configs(self) -> Dict[str, ModelConfig]:
        """Load configurations from files"""
        configs = {}
        
        # Load from YAML files
        for yaml_file in self.config_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'model' in data:
                        model_data = data['model']
                        model_type = yaml_file.stem
                        
                        # Extract extra parameters
                        extra_params = {}
                        for key, value in model_data.items():
                            if key not in ['d_model', 'n_layers', 'd_state', 'd_conv', 'dropout', 'vocab_size', 'max_seq_len']:
                                extra_params[key] = value
                        
                        config = ModelConfig(
                            model_type=model_type,
                            architecture=model_data.get('architecture', 'mamba'),
                            d_model=model_data.get('d_model', 512),
                            n_layers=model_data.get('n_layers', 8),
                            d_state=model_data.get('d_state', 32),
                            d_conv=model_data.get('d_conv', 4),
                            dropout=model_data.get('dropout', 0.1),
                            vocab_size=model_data.get('vocab_size', 50000),
                            max_seq_len=model_data.get('max_seq_len', 2048),
                            extra_params=extra_params
                        )
                        
                        configs[model_type] = config
                        logger.info(f"Loaded config: {model_type}")
                        
            except Exception as e:
                logger.warning(f"Could not load config from {yaml_file}: {e}")
        
        # Add default configs for missing model types
        for model_type, default_config in self.default_configs.items():
            if model_type not in configs:
                configs[model_type] = ModelConfig(
                    model_type=model_type,
                    **default_config
                )
        
        return configs
    
    def get_config(self, model_type: str) -> ModelConfig:
        """Get configuration for a model type"""
        if model_type not in self.configs:
            logger.warning(f"Unknown model type: {model_type}, using default")
            return self.default_configs.get(model_type, self.default_configs["mamba"])
        
        return self.configs[model_type]
    
    def update_config(self, model_type: str, **kwargs) -> ModelConfig:
        """Update configuration for a model type"""
        if model_type not in self.configs:
            # Create new config
            config = ModelConfig(model_type=model_type, **self.default_configs.get(model_type, {}))
            self.configs[model_type] = config
        
        config = self.configs[model_type]
        
        # Update parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                if config.extra_params is None:
                    config.extra_params = {}
                config.extra_params[key] = value
        
        # Save updated config
        self._save_config(model_type, config)
        
        logger.info(f"Updated config: {model_type}")
        return config
    
    def _save_config(self, model_type: str, config: ModelConfig):
        """Save configuration to file"""
        config_file = self.config_dir / f"{model_type}.yaml"
        
        # Convert to dict for YAML serialization
        config_dict = asdict(config)
        
        # Create YAML structure
        yaml_data = {
            'model': {
                'architecture': config_dict['architecture'],
                'd_model': config_dict['d_model'],
                'n_layers': config_dict['n_layers'],
                'd_state': config_dict['d_state'],
                'd_conv': config_dict['d_conv'],
                'dropout': config_dict['dropout'],
                'vocab_size': config_dict['vocab_size'],
                'max_seq_len': config_dict['max_seq_len']
            }
        }
        
        # Add extra parameters
        if config_dict['extra_params']:
            yaml_data['model'].update(config_dict['extra_params'])
        
        # Add paths section
        yaml_data['paths'] = {
            'tokenizer': f"Model/tokenizer.json",
            'weights': f"Model/weights/{model_type}_weights.safetensors",
            'weights_backup': f"Model/backups/{model_type}_weights.bak"
        }
        
        # Add training section
        yaml_data['train'] = {
            'data_glob': "Dataset/*.jsonl",
            'seq_len': config_dict['max_seq_len'],
            'batch_size': 8,
            'lr': 0.002,
            'epochs': 5,
            'log_dir': "logs",
            'init_from_existing': False,
            'save_every': 500,
            'eval_every': 500
        }
        
        # Save to file
        with open(config_file, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved config: {config_file}")
    
    def get_paths(self, model_type: str) -> Dict[str, str]:
        """Get paths for a model type"""
        config = self.get_config(model_type)
        
        return {
            'tokenizer': f"Model/tokenizer.json",
            'weights': f"Model/weights/{model_type}_weights.safetensors",
            'weights_backup': f"Model/backups/{model_type}_weights.bak",
            'checkpoints': f"Model/checkpoints/{model_type}",
            'logs': f"logs/{model_type}"
        }
    
    def get_training_config(self, model_type: str) -> Dict[str, Any]:
        """Get training configuration for a model type"""
        config = self.get_config(model_type)
        
        return {
            'model': asdict(config),
            'paths': self.get_paths(model_type),
            'train': {
                'data_glob': "Dataset/*.jsonl",
                'seq_len': config.max_seq_len,
                'batch_size': 8,
                'lr': 0.002,
                'epochs': 5,
                'log_dir': f"logs/{model_type}",
                'init_from_existing': False,
                'save_every': 500,
                'eval_every': 500
            },
            'distill': {
                'enabled': True,
                'alpha': 0.3,
                'ema_decay': 0.999
            }
        }
    
    def get_serve_config(self, model_type: str) -> Dict[str, Any]:
        """Get serving configuration for a model type"""
        config = self.get_config(model_type)
        paths = self.get_paths(model_type)
        
        return {
            'server': {
                'host': "0.0.0.0",
                'port': 8000
            },
            'inference': {
                'cpu_only': True,
                'quantization': "int8_dynamic",
                'architecture': config.architecture,
                'max_tokens': 256,
                'temperature': 0.7,
                'top_p': 0.9,
                'batching': {
                    'continuous': True,
                    'max_batch_size': 4
                }
            },
            'model': asdict(config),
            'paths': paths,
            'timeouts': {
                'generate_ms': 20000
            },
            'telemetry': {
                'prometheus_port': 9090
            }
        }
    
    def list_model_types(self) -> list:
        """List available model types"""
        return list(self.configs.keys())
    
    def create_model_config(self, model_type: str, **kwargs) -> ModelConfig:
        """Create a new model configuration"""
        if model_type in self.configs:
            logger.warning(f"Model type {model_type} already exists, updating instead")
            return self.update_config(model_type, **kwargs)
        
        # Create new config
        config = ModelConfig(
            model_type=model_type,
            **self.default_configs.get(model_type, self.default_configs["mamba"]),
            **kwargs
        )
        
        self.configs[model_type] = config
        self._save_config(model_type, config)
        
        logger.info(f"Created new config: {model_type}")
        return config

# Global config manager instance
config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    """Get the global config manager instance"""
    return config_manager

def get_model_config(model_type: str) -> ModelConfig:
    """Convenience function to get model configuration"""
    return config_manager.get_config(model_type)

def get_model_paths(model_type: str) -> Dict[str, str]:
    """Convenience function to get model paths"""
    return config_manager.get_paths(model_type)