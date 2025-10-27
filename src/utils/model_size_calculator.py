"""
Model Size Calculator - Cross-checks file size vs memory size
Provides accurate size calculations and validation
"""

import os
import torch
import json
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelSizeCalculator:
    """Calculates and validates model sizes"""
    
    def __init__(self, model_dir: str = "Model"):
        self.model_dir = Path(model_dir)
        self.weights_dir = self.model_dir / "weights"
    
    def calculate_model_size(self, model_path: str) -> Dict[str, Any]:
        """Calculate comprehensive model size information"""
        try:
            model_path = Path(model_path)
            
            # File size analysis
            file_analysis = self._analyze_file_size(model_path)
            
            # Memory size analysis
            memory_analysis = self._analyze_memory_size(model_path)
            
            # Compression analysis
            compression_analysis = self._analyze_compression(file_analysis, memory_analysis)
            
            # Validation
            validation = self._validate_size_consistency(file_analysis, memory_analysis)
            
            # Size breakdown by component
            component_breakdown = self._analyze_component_sizes(model_path)
            
            return {
                'file_analysis': file_analysis,
                'memory_analysis': memory_analysis,
                'compression_analysis': compression_analysis,
                'validation': validation,
                'component_breakdown': component_breakdown,
                'summary': self._generate_size_summary(file_analysis, memory_analysis, compression_analysis)
            }
            
        except Exception as e:
            logger.error(f"Model size calculation failed: {e}")
            raise
    
    def _analyze_file_size(self, model_path: Path) -> Dict[str, Any]:
        """Analyze file size on disk"""
        stat = model_path.stat()
        size_bytes = stat.st_size
        
        return {
            'size_bytes': size_bytes,
            'size_kb': size_bytes / 1024,
            'size_mb': size_bytes / (1024 * 1024),
            'size_gb': size_bytes / (1024 * 1024 * 1024),
            'created_at': stat.st_mtime,
            'modified_at': stat.st_mtime,
            'file_type': self._detect_file_type(model_path),
            'checksum_md5': self._calculate_checksum(model_path, 'md5'),
            'checksum_sha256': self._calculate_checksum(model_path, 'sha256')
        }
    
    def _analyze_memory_size(self, model_path: Path) -> Dict[str, Any]:
        """Analyze memory size when loaded"""
        try:
            # Load model
            model_data = torch.load(model_path, map_location='cpu')
            
            # Calculate memory usage
            memory_bytes = 0
            parameter_count = 0
            tensor_info = []
            
            if isinstance(model_data, dict):
                if 'state_dict' in model_data:
                    state_dict = model_data['state_dict']
                else:
                    state_dict = model_data
                
                for name, tensor in state_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        tensor_bytes = tensor.numel() * tensor.element_size()
                        memory_bytes += tensor_bytes
                        parameter_count += tensor.numel()
                        
                        tensor_info.append({
                            'name': name,
                            'shape': list(tensor.shape),
                            'dtype': str(tensor.dtype),
                            'parameters': tensor.numel(),
                            'bytes': tensor_bytes,
                            'mb': tensor_bytes / (1024 * 1024)
                        })
            
            return {
                'memory_bytes': memory_bytes,
                'memory_kb': memory_bytes / 1024,
                'memory_mb': memory_bytes / (1024 * 1024),
                'memory_gb': memory_bytes / (1024 * 1024 * 1024),
                'parameter_count': parameter_count,
                'tensor_count': len(tensor_info),
                'tensor_info': tensor_info,
                'estimated_parameters': parameter_count,
                'bytes_per_parameter': memory_bytes / parameter_count if parameter_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze memory size: {e}")
            return {
                'memory_bytes': 0,
                'memory_kb': 0,
                'memory_mb': 0,
                'memory_gb': 0,
                'parameter_count': 0,
                'tensor_count': 0,
                'tensor_info': [],
                'estimated_parameters': 0,
                'bytes_per_parameter': 0,
                'error': str(e)
            }
    
    def _analyze_compression(self, file_analysis: Dict, memory_analysis: Dict) -> Dict[str, Any]:
        """Analyze compression ratio and efficiency"""
        file_bytes = file_analysis['size_bytes']
        memory_bytes = memory_analysis['memory_bytes']
        
        if memory_bytes == 0:
            return {
                'compression_ratio': 0,
                'compression_efficiency': 0,
                'space_saved_bytes': 0,
                'space_saved_mb': 0,
                'space_saved_percentage': 0,
                'compression_type': 'unknown'
            }
        
        compression_ratio = memory_bytes / file_bytes
        space_saved_bytes = memory_bytes - file_bytes
        space_saved_percentage = (space_saved_bytes / memory_bytes) * 100
        
        # Determine compression type
        compression_type = 'none'
        if compression_ratio > 10:
            compression_type = 'high'
        elif compression_ratio > 2:
            compression_type = 'medium'
        elif compression_ratio > 1.1:
            compression_type = 'low'
        elif compression_ratio < 1:
            compression_type = 'expansion'
        
        return {
            'compression_ratio': compression_ratio,
            'compression_efficiency': space_saved_percentage,
            'space_saved_bytes': space_saved_bytes,
            'space_saved_mb': space_saved_bytes / (1024 * 1024),
            'space_saved_percentage': space_saved_percentage,
            'compression_type': compression_type
        }
    
    def _validate_size_consistency(self, file_analysis: Dict, memory_analysis: Dict) -> Dict[str, Any]:
        """Validate size consistency and detect issues"""
        issues = []
        warnings = []
        
        file_bytes = file_analysis['size_bytes']
        memory_bytes = memory_analysis['memory_bytes']
        parameter_count = memory_analysis['parameter_count']
        
        # Check for zero sizes
        if file_bytes == 0:
            issues.append("File size is zero")
        
        if memory_bytes == 0:
            issues.append("Memory size is zero")
        
        if parameter_count == 0:
            issues.append("Parameter count is zero")
        
        # Check for reasonable compression ratio
        if memory_bytes > 0 and file_bytes > 0:
            compression_ratio = memory_bytes / file_bytes
            
            if compression_ratio < 0.5:
                warnings.append(f"Very high compression ratio: {compression_ratio:.2f}x")
            elif compression_ratio > 20:
                warnings.append(f"Very low compression ratio: {compression_ratio:.2f}x")
        
        # Check parameter count vs memory size consistency
        if parameter_count > 0:
            expected_bytes = parameter_count * 4  # Assuming float32
            actual_bytes = memory_bytes
            
            if abs(expected_bytes - actual_bytes) > expected_bytes * 0.1:  # 10% tolerance
                warnings.append(f"Parameter count vs memory size mismatch: expected {expected_bytes:,} bytes, got {actual_bytes:,} bytes")
        
        # Check for suspiciously large files
        if file_bytes > 10 * 1024 * 1024 * 1024:  # 10GB
            warnings.append("File size exceeds 10GB")
        
        if memory_bytes > 50 * 1024 * 1024 * 1024:  # 50GB
            warnings.append("Memory size exceeds 50GB")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'validation_score': max(0, 100 - len(issues) * 20 - len(warnings) * 5)
        }
    
    def _analyze_component_sizes(self, model_path: Path) -> Dict[str, Any]:
        """Analyze size breakdown by model components"""
        try:
            model_data = torch.load(model_path, map_location='cpu')
            
            if isinstance(model_data, dict):
                if 'state_dict' in model_data:
                    state_dict = model_data['state_dict']
                else:
                    state_dict = model_data
                
                components = {
                    'embeddings': {'bytes': 0, 'parameters': 0, 'tensors': []},
                    'attention': {'bytes': 0, 'parameters': 0, 'tensors': []},
                    'feedforward': {'bytes': 0, 'parameters': 0, 'tensors': []},
                    'normalization': {'bytes': 0, 'parameters': 0, 'tensors': []},
                    'output': {'bytes': 0, 'parameters': 0, 'tensors': []},
                    'other': {'bytes': 0, 'parameters': 0, 'tensors': []}
                }
                
                for name, tensor in state_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        tensor_bytes = tensor.numel() * tensor.element_size()
                        tensor_params = tensor.numel()
                        
                        # Categorize by layer type
                        if 'embed' in name.lower():
                            components['embeddings']['bytes'] += tensor_bytes
                            components['embeddings']['parameters'] += tensor_params
                            components['embeddings']['tensors'].append(name)
                        elif 'attn' in name.lower() or 'attention' in name.lower():
                            components['attention']['bytes'] += tensor_bytes
                            components['attention']['parameters'] += tensor_params
                            components['attention']['tensors'].append(name)
                        elif 'linear' in name.lower() or 'ffn' in name.lower() or 'mlp' in name.lower():
                            components['feedforward']['bytes'] += tensor_bytes
                            components['feedforward']['parameters'] += tensor_params
                            components['feedforward']['tensors'].append(name)
                        elif 'norm' in name.lower() or 'layer_norm' in name.lower():
                            components['normalization']['bytes'] += tensor_bytes
                            components['normalization']['parameters'] += tensor_params
                            components['normalization']['tensors'].append(name)
                        elif 'out' in name.lower() or 'head' in name.lower():
                            components['output']['bytes'] += tensor_bytes
                            components['output']['parameters'] += tensor_params
                            components['output']['tensors'].append(name)
                        else:
                            components['other']['bytes'] += tensor_bytes
                            components['other']['parameters'] += tensor_params
                            components['other']['tensors'].append(name)
                
                # Convert bytes to MB
                for component in components.values():
                    component['mb'] = component['bytes'] / (1024 * 1024)
                
                return components
                
        except Exception as e:
            logger.error(f"Failed to analyze component sizes: {e}")
            return {}
    
    def _detect_file_type(self, model_path: Path) -> str:
        """Detect file type"""
        suffix = model_path.suffix.lower()
        
        if suffix == '.pt':
            return 'pytorch'
        elif suffix == '.pth':
            return 'pytorch'
        elif suffix == '.bin':
            return 'binary'
        elif suffix == '.json':
            return 'json'
        elif suffix == '.pkl':
            return 'pickle'
        else:
            return 'unknown'
    
    def _calculate_checksum(self, model_path: Path, algorithm: str = 'md5') -> str:
        """Calculate file checksum"""
        hash_func = hashlib.md5() if algorithm == 'md5' else hashlib.sha256()
        
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def _generate_size_summary(self, file_analysis: Dict, memory_analysis: Dict, compression_analysis: Dict) -> Dict[str, Any]:
        """Generate size summary"""
        return {
            'file_size_mb': file_analysis['size_mb'],
            'memory_size_mb': memory_analysis['memory_mb'],
            'compression_ratio': compression_analysis['compression_ratio'],
            'parameter_count': memory_analysis['parameter_count'],
            'bytes_per_parameter': memory_analysis['bytes_per_parameter'],
            'efficiency_rating': self._calculate_efficiency_rating(compression_analysis),
            'size_category': self._categorize_size(memory_analysis['memory_mb'])
        }
    
    def _calculate_efficiency_rating(self, compression_analysis: Dict) -> str:
        """Calculate efficiency rating"""
        ratio = compression_analysis['compression_ratio']
        
        if ratio > 10:
            return 'Excellent'
        elif ratio > 5:
            return 'Good'
        elif ratio > 2:
            return 'Fair'
        elif ratio > 1:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _categorize_size(self, size_mb: float) -> str:
        """Categorize model size"""
        if size_mb < 10:
            return 'Tiny'
        elif size_mb < 100:
            return 'Small'
        elif size_mb < 1000:
            return 'Medium'
        elif size_mb < 10000:
            return 'Large'
        else:
            return 'Huge'
    
    def generate_size_report(self, size_data: Dict[str, Any]) -> str:
        """Generate comprehensive size report"""
        file_analysis = size_data['file_analysis']
        memory_analysis = size_data['memory_analysis']
        compression_analysis = size_data['compression_analysis']
        validation = size_data['validation']
        component_breakdown = size_data['component_breakdown']
        summary = size_data['summary']
        
        report = f"""
# Model Size Analysis Report

## File Information
- **File Size**: {file_analysis['size_mb']:.2f} MB ({file_analysis['size_bytes']:,} bytes)
- **File Type**: {file_analysis['file_type']}
- **Created**: {file_analysis['created_at']}
- **MD5 Checksum**: {file_analysis['checksum_md5']}
- **SHA256 Checksum**: {file_analysis['checksum_sha256']}

## Memory Information
- **Memory Size**: {memory_analysis['memory_mb']:.2f} MB ({memory_analysis['memory_bytes']:,} bytes)
- **Parameter Count**: {memory_analysis['parameter_count']:,}
- **Tensor Count**: {memory_analysis['tensor_count']}
- **Bytes per Parameter**: {memory_analysis['bytes_per_parameter']:.2f}

## Compression Analysis
- **Compression Ratio**: {compression_analysis['compression_ratio']:.2f}x
- **Compression Type**: {compression_analysis['compression_type']}
- **Space Saved**: {compression_analysis['space_saved_mb']:.2f} MB ({compression_analysis['space_saved_percentage']:.1f}%)
- **Efficiency Rating**: {summary['efficiency_rating']}

## Size Summary
- **Size Category**: {summary['size_category']}
- **File vs Memory**: {file_analysis['size_mb']:.2f} MB → {memory_analysis['memory_mb']:.2f} MB
- **Compression Factor**: {compression_analysis['compression_ratio']:.2f}x

## Validation
- **Status**: {'✅ Valid' if validation['is_valid'] else '❌ Invalid'}
- **Validation Score**: {validation['validation_score']}/100
"""
        
        if validation['issues']:
            report += "\n### Issues\n"
            for issue in validation['issues']:
                report += f"- ❌ {issue}\n"
        
        if validation['warnings']:
            report += "\n### Warnings\n"
            for warning in validation['warnings']:
                report += f"- ⚠️ {warning}\n"
        
        if component_breakdown:
            report += "\n## Component Size Breakdown\n"
            total_mb = sum(comp['mb'] for comp in component_breakdown.values())
            
            for component_name, component_data in component_breakdown.items():
                if component_data['mb'] > 0:
                    percentage = (component_data['mb'] / total_mb) * 100
                    report += f"- **{component_name.title()}**: {component_data['mb']:.2f} MB ({percentage:.1f}%) - {component_data['parameters']:,} parameters\n"
        
        # Top 10 largest tensors
        if memory_analysis['tensor_info']:
            report += "\n## Top 10 Largest Tensors\n"
            sorted_tensors = sorted(memory_analysis['tensor_info'], key=lambda x: x['mb'], reverse=True)
            
            for i, tensor in enumerate(sorted_tensors[:10]):
                report += f"{i+1:2d}. **{tensor['name']}**: {tensor['shape']} = {tensor['mb']:.2f} MB\n"
        
        return report

def main():
    """Main function for testing"""
    calculator = ModelSizeCalculator()
    
    # Calculate size for the main model
    model_path = "Model/weights/Tantra_v1.0.pt"
    if os.path.exists(model_path):
        print("Calculating model size...")
        size_data = calculator.calculate_model_size(model_path)
        
        # Generate report
        report = calculator.generate_size_report(size_data)
        print(report)
        
        # Save report
        with open("model_size_report.md", "w") as f:
            f.write(report)
        
        print("Size calculation complete! Report saved to model_size_report.md")
    else:
        print(f"Model file not found: {model_path}")

if __name__ == "__main__":
    main()