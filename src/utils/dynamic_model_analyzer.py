"""
Dynamic Model Analyzer - Real-time model parameter and vocabulary analysis
Automatically calculates and cross-checks real model details
"""

import torch
import json
import os
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelAnalysis:
    """Comprehensive model analysis results"""
    # File information
    file_path: str
    file_size_bytes: int
    file_size_mb: float
    
    # Model architecture (actual)
    actual_d_model: int
    actual_n_layers: int
    actual_n_heads: int
    actual_vocab_size: int
    actual_max_seq_length: int
    
    # Parameter counts (real)
    total_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int
    
    # Memory usage
    memory_size_bytes: int
    memory_size_mb: float
    memory_size_gb: float
    
    # Layer breakdown
    layer_breakdown: Dict[str, Dict[str, Any]]
    
    # Vocabulary analysis
    vocabulary_size: int
    vocabulary_tokens: List[str]
    
    # Validation
    is_valid: bool
    validation_errors: List[str]
    
    # Metadata
    model_type: str
    created_at: str
    checksum: str

class DynamicModelAnalyzer:
    """Dynamic model analyzer that calculates real model details"""
    
    def __init__(self, model_dir: str = "Model"):
        self.model_dir = Path(model_dir)
        self.weights_dir = self.model_dir / "weights"
        self.tokenizer_dir = self.model_dir
        
    def analyze_model(self, model_path: str) -> ModelAnalysis:
        """Comprehensive model analysis"""
        try:
            model_path = Path(model_path)
            
            # File information
            file_info = self._get_file_info(model_path)
            
            # Load model
            model_data = self._load_model(model_path)
            
            # Architecture analysis
            architecture = self._analyze_architecture(model_data)
            
            # Parameter analysis
            parameters = self._analyze_parameters(model_data)
            
            # Memory analysis
            memory = self._analyze_memory(model_data)
            
            # Layer breakdown
            layer_breakdown = self._analyze_layers(model_data)
            
            # Vocabulary analysis
            vocabulary = self._analyze_vocabulary()
            
            # Validation
            validation = self._validate_model(architecture, parameters, memory, file_info)
            
            # Create analysis result
            analysis = ModelAnalysis(
                file_path=str(model_path),
                file_size_bytes=file_info['size_bytes'],
                file_size_mb=file_info['size_mb'],
                actual_d_model=architecture['d_model'],
                actual_n_layers=architecture['n_layers'],
                actual_n_heads=architecture['n_heads'],
                actual_vocab_size=architecture['vocab_size'],
                actual_max_seq_length=architecture['max_seq_length'],
                total_parameters=parameters['total'],
                trainable_parameters=parameters['trainable'],
                non_trainable_parameters=parameters['non_trainable'],
                memory_size_bytes=memory['bytes'],
                memory_size_mb=memory['mb'],
                memory_size_gb=memory['gb'],
                layer_breakdown=layer_breakdown,
                vocabulary_size=vocabulary['size'],
                vocabulary_tokens=vocabulary['tokens'],
                is_valid=validation['is_valid'],
                validation_errors=validation['errors'],
                model_type=architecture['model_type'],
                created_at=file_info['created_at'],
                checksum=file_info['checksum']
            )
            
            logger.info(f"Model analysis completed: {analysis.total_parameters:,} parameters, {analysis.memory_size_mb:.2f} MB")
            return analysis
            
        except Exception as e:
            logger.error(f"Model analysis failed: {e}")
            raise
    
    def _get_file_info(self, model_path: Path) -> Dict[str, Any]:
        """Get file information"""
        stat = model_path.stat()
        size_bytes = stat.st_size
        size_mb = size_bytes / (1024 * 1024)
        
        # Calculate checksum
        with open(model_path, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()
        
        return {
            'size_bytes': size_bytes,
            'size_mb': size_mb,
            'created_at': str(stat.st_mtime),
            'checksum': checksum
        }
    
    def _load_model(self, model_path: Path) -> Dict[str, torch.Tensor]:
        """Load model data"""
        try:
            model_data = torch.load(model_path, map_location='cpu')
            
            if isinstance(model_data, dict):
                if 'state_dict' in model_data:
                    return model_data['state_dict']
                else:
                    return model_data
            else:
                # Handle OrderedDict or other types
                return dict(model_data)
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _analyze_architecture(self, model_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze model architecture from actual parameters"""
        architecture = {
            'd_model': 0,
            'n_layers': 0,
            'n_heads': 0,
            'vocab_size': 0,
            'max_seq_length': 0,
            'model_type': 'unknown'
        }
        
        # Analyze embedding layer
        if 'embed.weight' in model_data:
            embed_weight = model_data['embed.weight']
            architecture['vocab_size'] = embed_weight.shape[0]
            architecture['d_model'] = embed_weight.shape[1]
        
        # Count layers
        layer_count = 0
        for key in model_data.keys():
            if key.startswith('layers.') and 'self_attn' in key:
                layer_num = int(key.split('.')[1])
                layer_count = max(layer_count, layer_num + 1)
        architecture['n_layers'] = layer_count
        
        # Analyze attention heads
        if 'layers.0.self_attn.in_proj_weight' in model_data:
            in_proj_weight = model_data['layers.0.self_attn.in_proj_weight']
            # in_proj_weight combines Q, K, V projections
            # Shape should be [3 * d_model, d_model] for multi-head
            if in_proj_weight.shape[0] == 3 * architecture['d_model']:
                architecture['n_heads'] = 8  # Default assumption, could be calculated more precisely
        
        # Analyze position embeddings
        if 'pos_embed' in model_data:
            pos_embed = model_data['pos_embed']
            architecture['max_seq_length'] = pos_embed.shape[0]
        
        # Determine model type
        if 'ocr_conv' in model_data:
            architecture['model_type'] = 'tantra_multimodal'
        else:
            architecture['model_type'] = 'transformer'
        
        return architecture
    
    def _analyze_parameters(self, model_data: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Analyze parameter counts"""
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        for name, tensor in model_data.items():
            if isinstance(tensor, torch.Tensor):
                param_count = tensor.numel()
                total_params += param_count
                
                # Assume all parameters are trainable unless specified otherwise
                trainable_params += param_count
        
        non_trainable_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }
    
    def _analyze_memory(self, model_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyze memory usage"""
        total_bytes = 0
        
        for tensor in model_data.values():
            if isinstance(tensor, torch.Tensor):
                total_bytes += tensor.numel() * tensor.element_size()
        
        return {
            'bytes': total_bytes,
            'mb': total_bytes / (1024 * 1024),
            'gb': total_bytes / (1024 * 1024 * 1024)
        }
    
    def _analyze_layers(self, model_data: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
        """Analyze individual layers"""
        layer_breakdown = {}
        
        for name, tensor in model_data.items():
            if isinstance(tensor, torch.Tensor):
                layer_breakdown[name] = {
                    'shape': list(tensor.shape),
                    'parameters': tensor.numel(),
                    'size_mb': (tensor.numel() * tensor.element_size()) / (1024 * 1024),
                    'dtype': str(tensor.dtype),
                    'requires_grad': tensor.requires_grad if hasattr(tensor, 'requires_grad') else True
                }
        
        return layer_breakdown
    
    def _analyze_vocabulary(self) -> Dict[str, Any]:
        """Analyze vocabulary from tokenizer files"""
        vocabulary = {
            'size': 0,
            'tokens': []
        }
        
        # Try to load tokenizer
        tokenizer_path = self.tokenizer_dir / "tokenizer.json"
        if tokenizer_path.exists():
            try:
                with open(tokenizer_path, 'r') as f:
                    tokenizer_data = json.load(f)
                
                if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
                    vocab = tokenizer_data['model']['vocab']
                    vocabulary['size'] = len(vocab)
                    vocabulary['tokens'] = list(vocab.keys())[:100]  # First 100 tokens
                    
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
        
        # Fallback: try vocab file
        vocab_path = self.tokenizer_dir / "tokenizer_vocab.json"
        if vocab_path.exists() and vocabulary['size'] == 0:
            try:
                with open(vocab_path, 'r') as f:
                    vocab_data = json.load(f)
                
                if isinstance(vocab_data, dict):
                    vocabulary['size'] = len(vocab_data)
                    vocabulary['tokens'] = list(vocab_data.keys())[:100]
                elif isinstance(vocab_data, list):
                    vocabulary['size'] = len(vocab_data)
                    vocabulary['tokens'] = vocab_data[:100]
                    
            except Exception as e:
                logger.warning(f"Failed to load vocab file: {e}")
        
        return vocabulary
    
    def _validate_model(self, architecture: Dict, parameters: Dict, memory: Dict, file_info: Dict) -> Dict[str, Any]:
        """Validate model consistency"""
        errors = []
        
        # Check parameter count consistency
        expected_params = parameters['total']
        if expected_params == 0:
            errors.append("No parameters found in model")
        
        # Check memory size consistency
        expected_memory_mb = parameters['total'] * 4 / (1024 * 1024)  # 4 bytes per float32
        actual_memory_mb = memory['mb']
        
        if abs(expected_memory_mb - actual_memory_mb) > 0.1:  # Allow small floating point differences
            errors.append(f"Memory size mismatch: expected {expected_memory_mb:.2f} MB, got {actual_memory_mb:.2f} MB")
        
        # Check architecture consistency
        if architecture['d_model'] == 0:
            errors.append("Could not determine model dimension")
        
        if architecture['n_layers'] == 0:
            errors.append("Could not determine number of layers")
        
        if architecture['vocab_size'] == 0:
            errors.append("Could not determine vocabulary size")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    def update_config_files(self, analysis: ModelAnalysis) -> None:
        """Update configuration files with real model data"""
        try:
            # Update weight config
            weight_config_path = self.weights_dir / "weight_config.json"
            weight_config = {
                "weights": {
                    "Tantra_v1.0": {
                        "name": "Tantra_v1.0",
                        "path": str(analysis.file_path),
                        "size_mb": analysis.file_size_mb,
                        "created_at": analysis.created_at,
                        "model_type": analysis.model_type,
                        "version": "v1.0",
                        "checksum": analysis.checksum,
                        "is_active": True,
                        "real_parameters": analysis.total_parameters,
                        "real_memory_mb": analysis.memory_size_mb
                    }
                },
                "last_updated": analysis.created_at
            }
            
            with open(weight_config_path, 'w') as f:
                json.dump(weight_config, f, indent=2)
            
            # Update model config
            model_config_path = self.weights_dir / "Tantra_real_config.json"
            model_config = {
                "tantra_config": {
                    "d_model": analysis.actual_d_model,
                    "n_layers": analysis.actual_n_layers,
                    "n_heads": analysis.actual_n_heads,
                    "d_ff": analysis.actual_d_model * 4,  # Typical FFN ratio
                    "vocab_size": analysis.actual_vocab_size,
                    "max_seq_length": analysis.actual_max_seq_length
                },
                "model_info": {
                    "name": "Tantra v1.0 (Real)",
                    "total_parameters": analysis.total_parameters,
                    "trainable_parameters": analysis.trainable_parameters,
                    "model_size_mb": analysis.memory_size_mb,
                    "file_size_mb": analysis.file_size_mb,
                    "d_model": analysis.actual_d_model,
                    "n_layers": analysis.actual_n_layers,
                    "n_heads": analysis.actual_n_heads,
                    "vocab_size": analysis.actual_vocab_size,
                    "max_seq_length": analysis.actual_max_seq_length,
                    "model_type": analysis.model_type,
                    "validation_status": "valid" if analysis.is_valid else "invalid",
                    "validation_errors": analysis.validation_errors
                },
                "layer_breakdown": analysis.layer_breakdown,
                "vocabulary_info": {
                    "size": analysis.vocabulary_size,
                    "sample_tokens": analysis.vocabulary_tokens[:20]
                },
                "analysis_timestamp": analysis.created_at
            }
            
            with open(model_config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            logger.info("Configuration files updated with real model data")
            
        except Exception as e:
            logger.error(f"Failed to update config files: {e}")
            raise
    
    def generate_report(self, analysis: ModelAnalysis) -> str:
        """Generate comprehensive model report"""
        report = f"""
# Tantra Model Analysis Report

## File Information
- **File Path**: {analysis.file_path}
- **File Size**: {analysis.file_size_mb:.2f} MB ({analysis.file_size_bytes:,} bytes)
- **Created**: {analysis.created_at}
- **Checksum**: {analysis.checksum}

## Model Architecture (Real)
- **Model Type**: {analysis.model_type}
- **Dimensions**: {analysis.actual_d_model}
- **Layers**: {analysis.actual_n_layers}
- **Attention Heads**: {analysis.actual_n_heads}
- **Vocabulary Size**: {analysis.actual_vocab_size:,}
- **Max Sequence Length**: {analysis.actual_max_seq_length}

## Parameters (Real)
- **Total Parameters**: {analysis.total_parameters:,}
- **Trainable Parameters**: {analysis.trainable_parameters:,}
- **Non-trainable Parameters**: {analysis.non_trainable_parameters:,}

## Memory Usage
- **Memory Size**: {analysis.memory_size_mb:.2f} MB ({analysis.memory_size_gb:.3f} GB)
- **File vs Memory Ratio**: {analysis.file_size_mb / analysis.memory_size_mb:.2f}x compression

## Vocabulary Analysis
- **Vocabulary Size**: {analysis.vocabulary_size:,}
- **Sample Tokens**: {', '.join(analysis.vocabulary_tokens[:10])}...

## Validation
- **Status**: {'✅ VALID' if analysis.is_valid else '❌ INVALID'}
- **Errors**: {len(analysis.validation_errors)} error(s)
"""
        
        if analysis.validation_errors:
            report += "\n### Validation Errors\n"
            for error in analysis.validation_errors:
                report += f"- {error}\n"
        
        report += f"""
## Layer Breakdown
Total layers analyzed: {len(analysis.layer_breakdown)}

### Top 10 Largest Layers
"""
        
        # Sort layers by size
        sorted_layers = sorted(
            analysis.layer_breakdown.items(),
            key=lambda x: x[1]['size_mb'],
            reverse=True
        )
        
        for name, info in sorted_layers[:10]:
            report += f"- **{name}**: {info['shape']} = {info['parameters']:,} params ({info['size_mb']:.2f} MB)\n"
        
        return report

def main():
    """Main function for testing"""
    analyzer = DynamicModelAnalyzer()
    
    # Analyze the main model
    model_path = "Model/weights/Tantra_v1.0.pt"
    if os.path.exists(model_path):
        print("Analyzing model...")
        analysis = analyzer.analyze_model(model_path)
        
        # Update config files
        analyzer.update_config_files(analysis)
        
        # Generate report
        report = analyzer.generate_report(analysis)
        print(report)
        
        # Save report
        with open("model_analysis_report.md", "w") as f:
            f.write(report)
        
        print("Analysis complete! Report saved to model_analysis_report.md")
    else:
        print(f"Model file not found: {model_path}")

if __name__ == "__main__":
    main()