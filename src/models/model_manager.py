"""
OCR-Native LLM Model Manager
Clean model loading, saving, and management
"""

import torch
import os
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from src.core.ocr_native_llm import OCRNativeLLM
from src.configs.ocr_config import OCRNativeConfig, ConfigManager

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages OCR-Native LLM models"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.weights_dir = self.model_dir / "weights"
        self.weights_dir.mkdir(exist_ok=True)
        self.configs_dir = self.model_dir / "configs"
        self.configs_dir.mkdir(exist_ok=True)
        
        # Model registry
        self.registry_file = self.model_dir / "model_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save model registry"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def create_model(self, config: OCRNativeConfig, model_name: str = "ocr_native") -> OCRNativeLLM:
        """Create a new OCR-Native LLM model"""
        model = OCRNativeLLM(config)
        
        # Save model info
        model_info = {
            'name': model_name,
            'config': {
                'd_model': config.d_model,
                'n_layers': config.n_layers,
                'n_heads': config.n_heads,
                'd_ff': config.d_ff,
                'vocab_size': config.vocab_size,
                'max_seq_length': config.max_seq_length
            },
            'parameters': model.get_model_info(),
            'created_at': str(time.time())
        }
        
        self.registry[model_name] = model_info
        self._save_registry()
        
        return model
    
    def save_model(self, model: OCRNativeLLM, model_name: str, version: str = "v1.0"):
        """Save model weights and configuration"""
        # Save weights
        weights_path = self.weights_dir / f"{model_name}_{version}.pt"
        torch.save(model.state_dict(), weights_path)
        
        # Save config
        config_path = self.configs_dir / f"{model_name}_{version}.yaml"
        ConfigManager.save_config(model.config, str(config_path))
        
        # Update registry
        if model_name not in self.registry:
            self.registry[model_name] = {}
        
        if 'versions' not in self.registry[model_name]:
            self.registry[model_name]['versions'] = {}
        
        self.registry[model_name]['versions'][version] = {
            'weights_path': str(weights_path),
            'config_path': str(config_path),
            'parameters': model.get_model_info()
        }
        
        self.registry[model_name]['active_version'] = version
        self._save_registry()
        
        logger.info(f"Model {model_name} v{version} saved successfully")
        return str(weights_path)
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> OCRNativeLLM:
        """Load model from saved weights"""
        if model_name not in self.registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if version is None:
            version = self.registry[model_name].get('active_version', 'v1.0')
        
        if version not in self.registry[model_name]['versions']:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        model_info = self.registry[model_name]['versions'][version]
        
        # Load config
        config_path = model_info['config_path']
        config = ConfigManager.load_config(config_path)
        
        # Create model
        model = OCRNativeLLM(config)
        
        # Load weights
        weights_path = model_info['weights_path']
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info(f"Model {model_name} v{version} loaded successfully")
        else:
            logger.warning(f"Weights file not found: {weights_path}")
        
        return model
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.registry.keys())
    
    def list_versions(self, model_name: str) -> List[str]:
        """List all versions for a model"""
        if model_name not in self.registry:
            return []
        
        return list(self.registry[model_name].get('versions', {}).keys())
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get model information"""
        if model_name not in self.registry:
            raise ValueError(f"Model {model_name} not found")
        
        if version is None:
            version = self.registry[model_name].get('active_version', 'v1.0')
        
        if version not in self.registry[model_name]['versions']:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        return self.registry[model_name]['versions'][version]
    
    def delete_model(self, model_name: str, version: Optional[str] = None):
        """Delete model or specific version"""
        if model_name not in self.registry:
            raise ValueError(f"Model {model_name} not found")
        
        if version is None:
            # Delete entire model
            for ver, info in self.registry[model_name]['versions'].items():
                if os.path.exists(info['weights_path']):
                    os.remove(info['weights_path'])
                if os.path.exists(info['config_path']):
                    os.remove(info['config_path'])
            del self.registry[model_name]
        else:
            # Delete specific version
            if version in self.registry[model_name]['versions']:
                info = self.registry[model_name]['versions'][version]
                if os.path.exists(info['weights_path']):
                    os.remove(info['weights_path'])
                if os.path.exists(info['config_path']):
                    os.remove(info['config_path'])
                del self.registry[model_name]['versions'][version]
        
        self._save_registry()
        logger.info(f"Model {model_name} v{version or 'all'} deleted successfully")
    
    def generate_weights(self, model_name: str = "ocr_native", version: str = "v1.0"):
        """Generate and save model weights"""
        # Create model with default config
        config = ConfigManager.get_default_config()
        model = self.create_model(config, model_name)
        
        # Save model
        weights_path = self.save_model(model, model_name, version)
        
        logger.info(f"Generated weights for {model_name} v{version}: {weights_path}")
        return weights_path