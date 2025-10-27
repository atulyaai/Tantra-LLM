"""
Dynamic Weight Management System for Tantra LLM
Handles model weight loading, saving, and versioning dynamically
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import torch
import safetensors.torch
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class WeightInfo:
    """Information about a weight file"""
    name: str
    path: str
    size_mb: float
    created_at: str
    model_type: str
    version: str
    checksum: str
    is_active: bool = False

class WeightManager:
    """Dynamic weight management system"""
    
    def __init__(self, base_dir: str = "Model", config_file: str = "Model/weight_config.json"):
        self.base_dir = Path(base_dir)
        self.config_file = Path(config_file)
        self.weights_dir = self.base_dir / "weights"
        self.backup_dir = self.base_dir / "backups"
        
        # Create directories
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create weight registry
        self.weight_registry = self._load_registry()
        
    def _load_registry(self) -> Dict[str, WeightInfo]:
        """Load weight registry from config file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    return {
                        name: WeightInfo(**info) 
                        for name, info in data.get('weights', {}).items()
                    }
            except Exception as e:
                logger.warning(f"Could not load weight registry: {e}")
        
        return {}
    
    def _save_registry(self):
        """Save weight registry to config file"""
        data = {
            'weights': {name: asdict(info) for name, info in self.weight_registry.items()},
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate checksum for a file"""
        import hashlib
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB"""
        return file_path.stat().st_size / (1024 * 1024)
    
    def register_weight(self, name: str, path: str, model_type: str, 
                       version: str = None, is_active: bool = False) -> WeightInfo:
        """Register a new weight file"""
        weight_path = Path(path)
        
        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {path}")
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        weight_info = WeightInfo(
            name=name,
            path=str(weight_path.absolute()),
            size_mb=self._get_file_size_mb(weight_path),
            created_at=datetime.now().isoformat(),
            model_type=model_type,
            version=version,
            checksum=self._calculate_checksum(weight_path),
            is_active=is_active
        )
        
        # If this is set as active, deactivate others of the same model type
        if is_active:
            for existing_name, existing_info in self.weight_registry.items():
                if existing_info.model_type == model_type:
                    existing_info.is_active = False
        
        self.weight_registry[name] = weight_info
        self._save_registry()
        
        logger.info(f"Registered weight: {name} ({model_type})")
        return weight_info
    
    def get_active_weight(self, model_type: str) -> Optional[WeightInfo]:
        """Get the active weight for a model type"""
        for weight_info in self.weight_registry.values():
            if weight_info.model_type == model_type and weight_info.is_active:
                return weight_info
        return None
    
    def get_weight_path(self, model_type: str, version: str = None) -> Optional[str]:
        """Get weight path for a model type and optional version"""
        if version:
            # Look for specific version
            for weight_info in self.weight_registry.values():
                if weight_info.model_type == model_type and weight_info.version == version:
                    return weight_info.path
        else:
            # Get active weight
            active_weight = self.get_active_weight(model_type)
            if active_weight:
                return active_weight.path
        
        return None
    
    def list_weights(self, model_type: str = None) -> List[WeightInfo]:
        """List all weights, optionally filtered by model type"""
        weights = list(self.weight_registry.values())
        if model_type:
            weights = [w for w in weights if w.model_type == model_type]
        return sorted(weights, key=lambda x: x.created_at, reverse=True)
    
    def set_active_weight(self, name: str) -> bool:
        """Set a weight as active for its model type"""
        if name not in self.weight_registry:
            return False
        
        weight_info = self.weight_registry[name]
        
        # Deactivate others of the same model type
        for existing_name, existing_info in self.weight_registry.items():
            if existing_info.model_type == weight_info.model_type:
                existing_info.is_active = False
        
        # Activate this weight
        weight_info.is_active = True
        self._save_registry()
        
        logger.info(f"Set active weight: {name} ({weight_info.model_type})")
        return True
    
    def backup_weight(self, name: str) -> bool:
        """Create a backup of a weight file"""
        if name not in self.weight_registry:
            return False
        
        weight_info = self.weight_registry[name]
        source_path = Path(weight_info.path)
        
        if not source_path.exists():
            return False
        
        # Create backup with timestamp
        backup_name = f"{name}_{weight_info.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / f"{backup_name}.safetensors"
        
        shutil.copy2(source_path, backup_path)
        
        logger.info(f"Created backup: {backup_path}")
        return True
    
    def load_weights(self, model_type: str, version: str = None) -> Optional[Dict[str, torch.Tensor]]:
        """Load weights for a model type"""
        weight_path = self.get_weight_path(model_type, version)
        
        if not weight_path or not Path(weight_path).exists():
            logger.warning(f"No weights found for {model_type}")
            return None
        
        try:
            if weight_path.endswith('.safetensors'):
                state_dict = safetensors.torch.load_file(weight_path)
            else:
                state_dict = torch.load(weight_path, map_location='cpu')
            
            logger.info(f"Loaded weights from: {weight_path}")
            return state_dict
            
        except Exception as e:
            logger.error(f"Failed to load weights from {weight_path}: {e}")
            return None
    
    def save_weights(self, state_dict: Dict[str, torch.Tensor], model_type: str, 
                    name: str = None, version: str = None, is_active: bool = True) -> str:
        """Save weights for a model type"""
        if name is None:
            name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create weight file path
        weight_path = self.weights_dir / f"{name}.safetensors"
        
        # Save weights
        safetensors.torch.save_file(state_dict, weight_path)
        
        # Register the weight
        self.register_weight(name, str(weight_path), model_type, version, is_active)
        
        logger.info(f"Saved weights: {weight_path}")
        return str(weight_path)
    
    def cleanup_old_weights(self, model_type: str, keep_count: int = 5):
        """Clean up old weight files, keeping only the most recent ones"""
        weights = self.list_weights(model_type)
        
        if len(weights) <= keep_count:
            return
        
        # Sort by creation time (oldest first)
        weights_to_remove = weights[keep_count:]
        
        for weight_info in weights_to_remove:
            try:
                # Remove from registry
                if weight_info.name in self.weight_registry:
                    del self.weight_registry[weight_info.name]
                
                # Remove file
                weight_path = Path(weight_info.path)
                if weight_path.exists():
                    weight_path.unlink()
                
                logger.info(f"Cleaned up old weight: {weight_info.name}")
                
            except Exception as e:
                logger.warning(f"Failed to cleanup weight {weight_info.name}: {e}")
        
        self._save_registry()
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get model configuration for a specific model type"""
        configs = {
            "mamba": {
                "d_model": 512,
                "n_layers": 8,
                "d_state": 32,
                "d_conv": 4,
                "dropout": 0.1
            },
            "mamba_multimodal": {
                "d_model": 768,
                "n_layers": 16,
                "d_state": 64,
                "d_conv": 4,
                "dropout": 0.1,
                "num_experts": 8,
                "quantization_bits": 8
            },
            "ocr_native": {
                "d_model": 512,
                "n_layers": 12,
                "d_state": 64,
                "d_conv": 4,
                "dropout": 0.1,
                "ocr_loss_weight": 0.3,
                "text_loss_weight": 0.7
            }
        }
        
        return configs.get(model_type, {})
    
    def validate_weights(self, model_type: str, version: str = None) -> bool:
        """Validate that weights exist and are loadable"""
        weight_path = self.get_weight_path(model_type, version)
        
        if not weight_path:
            return False
        
        try:
            # Try to load the weights
            state_dict = self.load_weights(model_type, version)
            return state_dict is not None
        except Exception as e:
            logger.error(f"Weight validation failed: {e}")
            return False

# Global weight manager instance
weight_manager = WeightManager()

def get_weight_manager() -> WeightManager:
    """Get the global weight manager instance"""
    return weight_manager

def load_model_weights(model_type: str, version: str = None) -> Optional[Dict[str, torch.Tensor]]:
    """Convenience function to load model weights"""
    return weight_manager.load_weights(model_type, version)

def save_model_weights(state_dict: Dict[str, torch.Tensor], model_type: str, 
                      name: str = None, version: str = None, is_active: bool = True) -> str:
    """Convenience function to save model weights"""
    return weight_manager.save_weights(state_dict, model_type, name, version, is_active)

def get_model_weight_path(model_type: str, version: str = None) -> Optional[str]:
    """Convenience function to get model weight path"""
    return weight_manager.get_weight_path(model_type, version)