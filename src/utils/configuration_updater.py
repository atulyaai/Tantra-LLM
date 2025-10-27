"""
Configuration Updater - Syncs config files with actual model data
Ensures consistency between model files and configuration
"""

import json
import os
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigurationUpdater:
    """Updates configuration files with real model data"""
    
    def __init__(self, model_dir: str = "Model"):
        self.model_dir = Path(model_dir)
        self.weights_dir = self.model_dir / "weights"
        self.config_dir = self.model_dir
        
        # Configuration file templates
        self.config_templates = {
            'weight_config': {
                'weights': {},
                'last_updated': None
            },
            'model_config': {
                'tantra_config': {},
                'model_info': {},
                'layer_breakdown': {},
                'vocabulary_info': {},
                'analysis_timestamp': None
            },
            'training_config': {
                'model_architecture': {},
                'training_parameters': {},
                'data_config': {},
                'optimization': {}
            }
        }
    
    def update_all_configurations(self, model_analysis: Dict[str, Any], vocab_analysis: Dict[str, Any], size_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Update all configuration files with real data"""
        try:
            results = {
                'updated_files': [],
                'created_files': [],
                'errors': [],
                'backup_created': False
            }
            
            # Create backup of existing configs
            backup_path = self._create_config_backup()
            if backup_path:
                results['backup_created'] = True
                results['backup_path'] = str(backup_path)
            
            # Update weight configuration
            weight_config_result = self._update_weight_config(model_analysis, size_analysis)
            if weight_config_result['success']:
                results['updated_files'].append(weight_config_result['file_path'])
            else:
                results['errors'].append(weight_config_result['error'])
            
            # Update model configuration
            model_config_result = self._update_model_config(model_analysis, vocab_analysis, size_analysis)
            if model_config_result['success']:
                results['updated_files'].append(model_config_result['file_path'])
            else:
                results['errors'].append(model_config_result['error'])
            
            # Update training configuration
            training_config_result = self._update_training_config(model_analysis)
            if training_config_result['success']:
                results['updated_files'].append(training_config_result['file_path'])
            else:
                results['errors'].append(training_config_result['error'])
            
            # Create consolidated configuration
            consolidated_result = self._create_consolidated_config(model_analysis, vocab_analysis, size_analysis)
            if consolidated_result['success']:
                results['created_files'].append(consolidated_result['file_path'])
            else:
                results['errors'].append(consolidated_result['error'])
            
            # Validate all configurations
            validation_result = self._validate_all_configurations()
            results['validation'] = validation_result
            
            logger.info(f"Configuration update completed: {len(results['updated_files'])} updated, {len(results['created_files'])} created")
            return results
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return {'error': str(e), 'updated_files': [], 'created_files': [], 'errors': [str(e)]}
    
    def _create_config_backup(self) -> Optional[Path]:
        """Create backup of existing configuration files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.model_dir / f"config_backup_{timestamp}"
            backup_dir.mkdir(exist_ok=True)
            
            # Backup existing config files
            config_files = [
                "weight_config.json",
                "Tantra_fine_tuned_config.json",
                "Tantra_medium_config.json",
                "tokenizer.json",
                "tokenizer_vocab.json"
            ]
            
            for config_file in config_files:
                source_path = self.model_dir / config_file
                if source_path.exists():
                    shutil.copy2(source_path, backup_dir / config_file)
            
            # Backup weights directory configs
            weights_config_dir = backup_dir / "weights"
            weights_config_dir.mkdir(exist_ok=True)
            
            for config_file in self.weights_dir.glob("*.json"):
                shutil.copy2(config_file, weights_config_dir / config_file.name)
            
            logger.info(f"Configuration backup created: {backup_dir}")
            return backup_dir
            
        except Exception as e:
            logger.error(f"Failed to create config backup: {e}")
            return None
    
    def _update_weight_config(self, model_analysis: Dict, size_analysis: Dict) -> Dict[str, Any]:
        """Update weight configuration file"""
        try:
            weight_config_path = self.weights_dir / "weight_config.json"
            
            # Extract model info
            model_name = "Tantra_v1.0"
            file_path = model_analysis.get('file_path', '')
            file_size_mb = size_analysis['file_analysis']['size_mb']
            memory_size_mb = size_analysis['memory_analysis']['memory_mb']
            parameter_count = model_analysis.get('total_parameters', 0)
            checksum = size_analysis['file_analysis']['checksum_md5']
            created_at = model_analysis.get('created_at', datetime.now().isoformat())
            
            weight_config = {
                "weights": {
                    model_name: {
                        "name": model_name,
                        "path": file_path,
                        "size_mb": file_size_mb,
                        "memory_mb": memory_size_mb,
                        "parameters": parameter_count,
                        "created_at": created_at,
                        "model_type": model_analysis.get('model_type', 'tantra_multimodal'),
                        "version": "v1.0",
                        "checksum": checksum,
                        "is_active": True,
                        "validation_status": "valid" if model_analysis.get('is_valid', False) else "invalid",
                        "compression_ratio": size_analysis['compression_analysis']['compression_ratio'],
                        "efficiency_rating": size_analysis['summary']['efficiency_rating']
                    }
                },
                "last_updated": datetime.now().isoformat(),
                "total_models": 1,
                "total_size_mb": file_size_mb,
                "total_memory_mb": memory_size_mb
            }
            
            # Write to file
            with open(weight_config_path, 'w') as f:
                json.dump(weight_config, f, indent=2)
            
            return {'success': True, 'file_path': str(weight_config_path)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _update_model_config(self, model_analysis: Dict, vocab_analysis: Dict, size_analysis: Dict) -> Dict[str, Any]:
        """Update model configuration file"""
        try:
            model_config_path = self.weights_dir / "Tantra_real_config.json"
            
            # Extract architecture info
            tantra_config = {
                "d_model": model_analysis.get('actual_d_model', 0),
                "n_layers": model_analysis.get('actual_n_layers', 0),
                "n_heads": model_analysis.get('actual_n_heads', 0),
                "d_ff": model_analysis.get('actual_d_model', 0) * 4,  # Typical FFN ratio
                "vocab_size": model_analysis.get('actual_vocab_size', 0),
                "max_seq_length": model_analysis.get('actual_max_seq_length', 0)
            }
            
            # Extract model info
            model_info = {
                "name": "Tantra v1.0 (Real)",
                "total_parameters": model_analysis.get('total_parameters', 0),
                "trainable_parameters": model_analysis.get('trainable_parameters', 0),
                "model_size_mb": size_analysis['memory_analysis']['memory_mb'],
                "file_size_mb": size_analysis['file_analysis']['size_mb'],
                "d_model": model_analysis.get('actual_d_model', 0),
                "n_layers": model_analysis.get('actual_n_layers', 0),
                "n_heads": model_analysis.get('actual_n_heads', 0),
                "vocab_size": model_analysis.get('actual_vocab_size', 0),
                "max_seq_length": model_analysis.get('actual_max_seq_length', 0),
                "model_type": model_analysis.get('model_type', 'tantra_multimodal'),
                "validation_status": "valid" if model_analysis.get('is_valid', False) else "invalid",
                "validation_errors": model_analysis.get('validation_errors', []),
                "compression_ratio": size_analysis['compression_analysis']['compression_ratio'],
                "efficiency_rating": size_analysis['summary']['efficiency_rating'],
                "size_category": size_analysis['summary']['size_category']
            }
            
            # Layer breakdown
            layer_breakdown = model_analysis.get('layer_breakdown', {})
            
            # Vocabulary info
            vocabulary_info = {
                "size": vocab_analysis.get('size', 0),
                "sample_tokens": vocab_analysis.get('tokens', [])[:20],
                "token_types": vocab_analysis.get('token_types', {}),
                "language_hints": vocab_analysis.get('language_hints', {}),
                "coverage_analysis": vocab_analysis.get('coverage_analysis', {})
            }
            
            model_config = {
                "tantra_config": tantra_config,
                "model_info": model_info,
                "layer_breakdown": layer_breakdown,
                "vocabulary_info": vocabulary_info,
                "size_analysis": {
                    "file_analysis": size_analysis['file_analysis'],
                    "memory_analysis": size_analysis['memory_analysis'],
                    "compression_analysis": size_analysis['compression_analysis']
                },
                "analysis_timestamp": datetime.now().isoformat(),
                "config_version": "2.0",
                "auto_generated": True
            }
            
            # Write to file
            with open(model_config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            return {'success': True, 'file_path': str(model_config_path)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _update_training_config(self, model_analysis: Dict) -> Dict[str, Any]:
        """Update training configuration file"""
        try:
            training_config_path = self.weights_dir / "training_config.json"
            
            # Model architecture for training
            model_architecture = {
                "d_model": model_analysis.get('actual_d_model', 0),
                "n_layers": model_analysis.get('actual_n_layers', 0),
                "n_heads": model_analysis.get('actual_n_heads', 0),
                "d_ff": model_analysis.get('actual_d_model', 0) * 4,
                "vocab_size": model_analysis.get('actual_vocab_size', 0),
                "max_seq_length": model_analysis.get('actual_max_seq_length', 0),
                "total_parameters": model_analysis.get('total_parameters', 0)
            }
            
            # Training parameters optimized for this model size
            training_parameters = self._calculate_optimal_training_params(model_analysis)
            
            # Data configuration
            data_config = {
                "batch_size": self._calculate_optimal_batch_size(model_analysis),
                "sequence_length": model_analysis.get('actual_max_seq_length', 512),
                "vocab_size": model_analysis.get('actual_vocab_size', 10000),
                "data_loading_workers": 4,
                "pin_memory": True
            }
            
            # Optimization settings
            optimization = {
                "learning_rate": training_parameters['learning_rate'],
                "weight_decay": training_parameters['weight_decay'],
                "gradient_clipping": training_parameters['gradient_clipping'],
                "optimizer": "AdamW",
                "scheduler": "cosine",
                "warmup_steps": training_parameters['warmup_steps']
            }
            
            training_config = {
                "model_architecture": model_architecture,
                "training_parameters": training_parameters,
                "data_config": data_config,
                "optimization": optimization,
                "generated_at": datetime.now().isoformat(),
                "model_size_category": self._categorize_model_size(model_analysis),
                "recommended_hardware": self._recommend_hardware(model_analysis)
            }
            
            # Write to file
            with open(training_config_path, 'w') as f:
                json.dump(training_config, f, indent=2)
            
            return {'success': True, 'file_path': str(training_config_path)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_consolidated_config(self, model_analysis: Dict, vocab_analysis: Dict, size_analysis: Dict) -> Dict[str, Any]:
        """Create consolidated configuration file"""
        try:
            consolidated_path = self.model_dir / "consolidated_config.json"
            
            consolidated_config = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "description": "Consolidated configuration with real model data"
                },
                "model_analysis": model_analysis,
                "vocabulary_analysis": vocab_analysis,
                "size_analysis": size_analysis,
                "summary": {
                    "total_parameters": model_analysis.get('total_parameters', 0),
                    "model_size_mb": size_analysis['memory_analysis']['memory_mb'],
                    "file_size_mb": size_analysis['file_analysis']['size_mb'],
                    "vocabulary_size": vocab_analysis.get('size', 0),
                    "compression_ratio": size_analysis['compression_analysis']['compression_ratio'],
                    "validation_status": "valid" if model_analysis.get('is_valid', False) else "invalid"
                }
            }
            
            # Write to file
            with open(consolidated_path, 'w') as f:
                json.dump(consolidated_config, f, indent=2)
            
            return {'success': True, 'file_path': str(consolidated_path)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_optimal_training_params(self, model_analysis: Dict) -> Dict[str, Any]:
        """Calculate optimal training parameters based on model size"""
        param_count = model_analysis.get('total_parameters', 0)
        
        # Learning rate based on model size
        if param_count < 1_000_000:  # < 1M params
            learning_rate = 1e-3
            batch_size = 32
        elif param_count < 10_000_000:  # < 10M params
            learning_rate = 5e-4
            batch_size = 16
        elif param_count < 100_000_000:  # < 100M params
            learning_rate = 1e-4
            batch_size = 8
        else:  # > 100M params
            learning_rate = 5e-5
            batch_size = 4
        
        return {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": 0.01,
            "gradient_clipping": 1.0,
            "warmup_steps": max(100, param_count // 10000),
            "max_epochs": 50,
            "patience": 10
        }
    
    def _calculate_optimal_batch_size(self, model_analysis: Dict) -> int:
        """Calculate optimal batch size based on model size"""
        param_count = model_analysis.get('total_parameters', 0)
        
        if param_count < 1_000_000:
            return 32
        elif param_count < 10_000_000:
            return 16
        elif param_count < 100_000_000:
            return 8
        else:
            return 4
    
    def _categorize_model_size(self, model_analysis: Dict) -> str:
        """Categorize model size"""
        param_count = model_analysis.get('total_parameters', 0)
        
        if param_count < 1_000_000:
            return "tiny"
        elif param_count < 10_000_000:
            return "small"
        elif param_count < 100_000_000:
            return "medium"
        elif param_count < 1_000_000_000:
            return "large"
        else:
            return "huge"
    
    def _recommend_hardware(self, model_analysis: Dict) -> Dict[str, Any]:
        """Recommend hardware based on model size"""
        param_count = model_analysis.get('total_parameters', 0)
        memory_mb = model_analysis.get('memory_size_mb', 0)
        
        if param_count < 10_000_000:  # < 10M params
            return {
                "min_ram_gb": 4,
                "recommended_ram_gb": 8,
                "min_vram_gb": 2,
                "recommended_vram_gb": 4,
                "cpu_cores": 4,
                "storage_gb": 10
            }
        elif param_count < 100_000_000:  # < 100M params
            return {
                "min_ram_gb": 8,
                "recommended_ram_gb": 16,
                "min_vram_gb": 4,
                "recommended_vram_gb": 8,
                "cpu_cores": 8,
                "storage_gb": 20
            }
        else:  # > 100M params
            return {
                "min_ram_gb": 16,
                "recommended_ram_gb": 32,
                "min_vram_gb": 8,
                "recommended_vram_gb": 16,
                "cpu_cores": 16,
                "storage_gb": 50
            }
    
    def _validate_all_configurations(self) -> Dict[str, Any]:
        """Validate all configuration files"""
        validation_results = {
            'valid_files': [],
            'invalid_files': [],
            'errors': []
        }
        
        config_files = [
            self.weights_dir / "weight_config.json",
            self.weights_dir / "Tantra_real_config.json",
            self.weights_dir / "training_config.json",
            self.model_dir / "consolidated_config.json"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        json.load(f)
                    validation_results['valid_files'].append(str(config_file))
                except Exception as e:
                    validation_results['invalid_files'].append(str(config_file))
                    validation_results['errors'].append(f"{config_file}: {e}")
            else:
                validation_results['errors'].append(f"File not found: {config_file}")
        
        validation_results['is_valid'] = len(validation_results['invalid_files']) == 0
        return validation_results

def main():
    """Main function for testing"""
    updater = ConfigurationUpdater()
    
    # Mock data for testing
    mock_model_analysis = {
        'file_path': 'Model/weights/Tantra_v1.0.pt',
        'total_parameters': 5002432,
        'actual_d_model': 128,
        'actual_n_layers': 4,
        'actual_n_heads': 8,
        'actual_vocab_size': 10000,
        'actual_max_seq_length': 512,
        'model_type': 'tantra_multimodal',
        'is_valid': True,
        'validation_errors': [],
        'created_at': datetime.now().isoformat()
    }
    
    mock_vocab_analysis = {
        'size': 10000,
        'tokens': [f'token_{i}' for i in range(100)],
        'token_types': {'letters': 8000, 'numbers': 1000, 'special': 1000},
        'language_hints': {'english_indicators': 5000},
        'coverage_analysis': {'character_diversity': 0.8}
    }
    
    mock_size_analysis = {
        'file_analysis': {'size_mb': 19.1, 'checksum_md5': 'abc123'},
        'memory_analysis': {'memory_mb': 19.08, 'parameter_count': 5002432},
        'compression_analysis': {'compression_ratio': 1.0},
        'summary': {'efficiency_rating': 'Good', 'size_category': 'Small'}
    }
    
    print("Updating configurations...")
    results = updater.update_all_configurations(mock_model_analysis, mock_vocab_analysis, mock_size_analysis)
    
    print(f"Configuration update results:")
    print(f"- Updated files: {len(results['updated_files'])}")
    print(f"- Created files: {len(results['created_files'])}")
    print(f"- Errors: {len(results['errors'])}")
    print(f"- Backup created: {results['backup_created']}")

if __name__ == "__main__":
    main()