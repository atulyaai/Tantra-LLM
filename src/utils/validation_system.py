"""
Validation System - Ensures consistency across all model files
Comprehensive validation and error detection
"""

import json
import os
import torch
import hashlib
from typing import Dict, Any, List, Tuple, Optional, Set
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ValidationSystem:
    """Comprehensive validation system for model files"""
    
    def __init__(self, model_dir: str = "Model"):
        self.model_dir = Path(model_dir)
        self.weights_dir = self.model_dir / "weights"
        self.validation_rules = self._load_validation_rules()
    
    def validate_all(self) -> Dict[str, Any]:
        """Comprehensive validation of all model files"""
        validation_results = {
            'overall_status': 'unknown',
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'warnings': 0,
            'errors': [],
            'warnings_list': [],
            'file_validations': {},
            'cross_file_consistency': {},
            'recommendations': [],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Validate individual files
            file_validations = self._validate_individual_files()
            validation_results['file_validations'] = file_validations
            
            # Cross-file consistency checks
            consistency_checks = self._validate_cross_file_consistency()
            validation_results['cross_file_consistency'] = consistency_checks
            
            # Calculate overall statistics
            self._calculate_validation_stats(validation_results)
            
            # Generate recommendations
            validation_results['recommendations'] = self._generate_recommendations(validation_results)
            
            logger.info(f"Validation completed: {validation_results['passed_checks']}/{validation_results['total_checks']} checks passed")
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results['errors'].append(f"Validation system error: {e}")
            validation_results['overall_status'] = 'error'
            return validation_results
    
    def _validate_individual_files(self) -> Dict[str, Any]:
        """Validate individual model files"""
        file_validations = {}
        
        # Validate model weight files
        for weight_file in self.weights_dir.glob("*.pt"):
            file_validations[str(weight_file)] = self._validate_model_file(weight_file)
        
        # Validate configuration files
        config_files = [
            "weight_config.json",
            "Tantra_real_config.json",
            "training_config.json",
            "consolidated_config.json"
        ]
        
        for config_file in config_files:
            config_path = self.model_dir / config_file
            if config_path.exists():
                file_validations[str(config_path)] = self._validate_config_file(config_path)
        
        # Validate tokenizer files
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_vocab.json"
        ]
        
        for tokenizer_file in tokenizer_files:
            tokenizer_path = self.model_dir / tokenizer_file
            if tokenizer_path.exists():
                file_validations[str(tokenizer_path)] = self._validate_tokenizer_file(tokenizer_path)
        
        return file_validations
    
    def _validate_model_file(self, model_path: Path) -> Dict[str, Any]:
        """Validate individual model file"""
        validation = {
            'file_type': 'model',
            'is_valid': False,
            'checks_passed': 0,
            'total_checks': 0,
            'errors': [],
            'warnings': [],
            'file_info': {},
            'model_info': {}
        }
        
        try:
            # File existence and basic info
            validation['total_checks'] += 1
            if model_path.exists():
                validation['checks_passed'] += 1
                stat = model_path.stat()
                validation['file_info'] = {
                    'size_bytes': stat.st_size,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'created_at': stat.st_mtime,
                    'is_readable': os.access(model_path, os.R_OK)
                }
            else:
                validation['errors'].append("File does not exist")
                return validation
            
            # File readability
            validation['total_checks'] += 1
            if os.access(model_path, os.R_OK):
                validation['checks_passed'] += 1
            else:
                validation['errors'].append("File is not readable")
            
            # File size validation
            validation['total_checks'] += 1
            file_size_mb = stat.st_size / (1024 * 1024)
            if 0 < file_size_mb < 10000:  # Between 0 and 10GB
                validation['checks_passed'] += 1
            else:
                validation['warnings'].append(f"Unusual file size: {file_size_mb:.2f} MB")
            
            # Model loading validation
            validation['total_checks'] += 1
            try:
                model_data = torch.load(model_path, map_location='cpu')
                validation['checks_passed'] += 1
                
                # Model structure validation
                if isinstance(model_data, dict):
                    if 'state_dict' in model_data:
                        state_dict = model_data['state_dict']
                    else:
                        state_dict = model_data
                    
                    # Count parameters
                    param_count = sum(tensor.numel() for tensor in state_dict.values() if isinstance(tensor, torch.Tensor))
                    validation['model_info'] = {
                        'parameter_count': param_count,
                        'tensor_count': len([t for t in state_dict.values() if isinstance(t, torch.Tensor)]),
                        'has_embeddings': any('embed' in name.lower() for name in state_dict.keys()),
                        'has_attention': any('attn' in name.lower() for name in state_dict.keys()),
                        'has_output': any('out' in name.lower() or 'head' in name.lower() for name in state_dict.keys())
                    }
                    
                    # Parameter count validation
                    validation['total_checks'] += 1
                    if 0 < param_count < 1_000_000_000:  # Between 0 and 1B parameters
                        validation['checks_passed'] += 1
                    else:
                        validation['warnings'].append(f"Unusual parameter count: {param_count:,}")
                    
                    # Architecture validation
                    validation['total_checks'] += 1
                    if validation['model_info']['has_embeddings'] and validation['model_info']['has_attention']:
                        validation['checks_passed'] += 1
                    else:
                        validation['errors'].append("Missing essential model components (embeddings or attention)")
                
            except Exception as e:
                validation['errors'].append(f"Failed to load model: {e}")
            
            # Checksum validation
            validation['total_checks'] += 1
            try:
                with open(model_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                validation['file_info']['md5_checksum'] = file_hash
                validation['checks_passed'] += 1
            except Exception as e:
                validation['errors'].append(f"Failed to calculate checksum: {e}")
            
            validation['is_valid'] = len(validation['errors']) == 0
            
        except Exception as e:
            validation['errors'].append(f"Validation error: {e}")
        
        return validation
    
    def _validate_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Validate configuration file"""
        validation = {
            'file_type': 'config',
            'is_valid': False,
            'checks_passed': 0,
            'total_checks': 0,
            'errors': [],
            'warnings': [],
            'config_info': {}
        }
        
        try:
            # File existence
            validation['total_checks'] += 1
            if config_path.exists():
                validation['checks_passed'] += 1
            else:
                validation['errors'].append("Config file does not exist")
                return validation
            
            # JSON validity
            validation['total_checks'] += 1
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                validation['checks_passed'] += 1
                validation['config_info'] = {
                    'keys': list(config_data.keys()) if isinstance(config_data, dict) else [],
                    'is_dict': isinstance(config_data, dict),
                    'size': len(str(config_data))
                }
            except json.JSONDecodeError as e:
                validation['errors'].append(f"Invalid JSON: {e}")
                return validation
            except Exception as e:
                validation['errors'].append(f"Failed to read config: {e}")
                return validation
            
            # Required fields validation
            if isinstance(config_data, dict):
                required_fields = self._get_required_config_fields(config_path.name)
                for field in required_fields:
                    validation['total_checks'] += 1
                    if field in config_data:
                        validation['checks_passed'] += 1
                    else:
                        validation['warnings'].append(f"Missing recommended field: {field}")
                
                # Type validation
                validation['total_checks'] += 1
                if self._validate_config_types(config_data, config_path.name):
                    validation['checks_passed'] += 1
                else:
                    validation['warnings'].append("Some config values have unexpected types")
            
            validation['is_valid'] = len(validation['errors']) == 0
            
        except Exception as e:
            validation['errors'].append(f"Config validation error: {e}")
        
        return validation
    
    def _validate_tokenizer_file(self, tokenizer_path: Path) -> Dict[str, Any]:
        """Validate tokenizer file"""
        validation = {
            'file_type': 'tokenizer',
            'is_valid': False,
            'checks_passed': 0,
            'total_checks': 0,
            'errors': [],
            'warnings': [],
            'tokenizer_info': {}
        }
        
        try:
            # File existence
            validation['total_checks'] += 1
            if tokenizer_path.exists():
                validation['checks_passed'] += 1
            else:
                validation['errors'].append("Tokenizer file does not exist")
                return validation
            
            # JSON validity
            validation['total_checks'] += 1
            try:
                with open(tokenizer_path, 'r', encoding='utf-8') as f:
                    tokenizer_data = json.load(f)
                validation['checks_passed'] += 1
            except Exception as e:
                validation['errors'].append(f"Failed to load tokenizer: {e}")
                return validation
            
            # Vocabulary validation
            vocab_size = 0
            if isinstance(tokenizer_data, dict):
                if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
                    vocab = tokenizer_data['model']['vocab']
                    vocab_size = len(vocab)
                elif 'vocab' in tokenizer_data:
                    vocab = tokenizer_data['vocab']
                    vocab_size = len(vocab)
            elif isinstance(tokenizer_data, list):
                vocab_size = len(tokenizer_data)
            
            validation['tokenizer_info'] = {
                'vocab_size': vocab_size,
                'has_vocab': vocab_size > 0,
                'file_type': tokenizer_path.name
            }
            
            # Vocabulary size validation
            validation['total_checks'] += 1
            if 0 < vocab_size < 1_000_000:  # Between 0 and 1M tokens
                validation['checks_passed'] += 1
            else:
                validation['warnings'].append(f"Unusual vocabulary size: {vocab_size:,}")
            
            validation['is_valid'] = len(validation['errors']) == 0
            
        except Exception as e:
            validation['errors'].append(f"Tokenizer validation error: {e}")
        
        return validation
    
    def _validate_cross_file_consistency(self) -> Dict[str, Any]:
        """Validate consistency across files"""
        consistency = {
            'parameter_consistency': {'is_consistent': False, 'details': {}},
            'vocabulary_consistency': {'is_consistent': False, 'details': {}},
            'size_consistency': {'is_consistent': False, 'details': {}},
            'config_consistency': {'is_consistent': False, 'details': {}},
            'overall_consistency': False
        }
        
        try:
            # Parameter consistency
            param_consistency = self._check_parameter_consistency()
            consistency['parameter_consistency'] = param_consistency
            
            # Vocabulary consistency
            vocab_consistency = self._check_vocabulary_consistency()
            consistency['vocabulary_consistency'] = vocab_consistency
            
            # Size consistency
            size_consistency = self._check_size_consistency()
            consistency['size_consistency'] = size_consistency
            
            # Config consistency
            config_consistency = self._check_config_consistency()
            consistency['config_consistency'] = config_consistency
            
            # Overall consistency
            consistency['overall_consistency'] = all([
                param_consistency['is_consistent'],
                vocab_consistency['is_consistent'],
                size_consistency['is_consistent'],
                config_consistency['is_consistent']
            ])
            
        except Exception as e:
            logger.error(f"Cross-file consistency check failed: {e}")
            consistency['error'] = str(e)
        
        return consistency
    
    def _check_parameter_consistency(self) -> Dict[str, Any]:
        """Check parameter count consistency across files"""
        param_counts = {}
        
        # Get parameter count from model files
        for weight_file in self.weights_dir.glob("*.pt"):
            try:
                model_data = torch.load(weight_file, map_location='cpu')
                if isinstance(model_data, dict):
                    if 'state_dict' in model_data:
                        state_dict = model_data['state_dict']
                    else:
                        state_dict = model_data
                    
                    param_count = sum(tensor.numel() for tensor in state_dict.values() if isinstance(tensor, torch.Tensor))
                    param_counts[str(weight_file)] = param_count
            except Exception as e:
                logger.warning(f"Failed to load {weight_file}: {e}")
        
        # Get parameter count from config files
        config_files = [
            "Tantra_real_config.json",
            "consolidated_config.json"
        ]
        
        for config_file in config_files:
            config_path = self.model_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    if 'model_info' in config_data and 'total_parameters' in config_data['model_info']:
                        param_counts[str(config_path)] = config_data['model_info']['total_parameters']
                except Exception as e:
                    logger.warning(f"Failed to load {config_path}: {e}")
        
        # Check consistency
        unique_counts = set(param_counts.values())
        is_consistent = len(unique_counts) <= 1
        
        return {
            'is_consistent': is_consistent,
            'details': param_counts,
            'unique_counts': list(unique_counts),
            'inconsistency_reason': 'Multiple different parameter counts found' if not is_consistent else None
        }
    
    def _check_vocabulary_consistency(self) -> Dict[str, Any]:
        """Check vocabulary size consistency across files"""
        vocab_sizes = {}
        
        # Get vocabulary size from tokenizer files
        tokenizer_files = ["tokenizer.json", "tokenizer_vocab.json"]
        for tokenizer_file in tokenizer_files:
            tokenizer_path = self.model_dir / tokenizer_file
            if tokenizer_path.exists():
                try:
                    with open(tokenizer_path, 'r', encoding='utf-8') as f:
                        tokenizer_data = json.load(f)
                    
                    vocab_size = 0
                    if isinstance(tokenizer_data, dict):
                        if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
                            vocab_size = len(tokenizer_data['model']['vocab'])
                        elif 'vocab' in tokenizer_data:
                            vocab_size = len(tokenizer_data['vocab'])
                    elif isinstance(tokenizer_data, list):
                        vocab_size = len(tokenizer_data)
                    
                    if vocab_size > 0:
                        vocab_sizes[str(tokenizer_path)] = vocab_size
                except Exception as e:
                    logger.warning(f"Failed to load {tokenizer_path}: {e}")
        
        # Get vocabulary size from model files
        for weight_file in self.weights_dir.glob("*.pt"):
            try:
                model_data = torch.load(weight_file, map_location='cpu')
                if isinstance(model_data, dict):
                    if 'state_dict' in model_data:
                        state_dict = model_data['state_dict']
                    else:
                        state_dict = model_data
                    
                    # Look for embedding layer
                    if 'embed.weight' in state_dict:
                        embed_weight = state_dict['embed.weight']
                        vocab_size = embed_weight.shape[0]
                        vocab_sizes[str(weight_file)] = vocab_size
            except Exception as e:
                logger.warning(f"Failed to load {weight_file}: {e}")
        
        # Check consistency
        unique_sizes = set(vocab_sizes.values())
        is_consistent = len(unique_sizes) <= 1
        
        return {
            'is_consistent': is_consistent,
            'details': vocab_sizes,
            'unique_sizes': list(unique_sizes),
            'inconsistency_reason': 'Multiple different vocabulary sizes found' if not is_consistent else None
        }
    
    def _check_size_consistency(self) -> Dict[str, Any]:
        """Check file size consistency"""
        file_sizes = {}
        
        # Get file sizes
        for weight_file in self.weights_dir.glob("*.pt"):
            if weight_file.exists():
                file_sizes[str(weight_file)] = weight_file.stat().st_size
        
        # Check for reasonable size differences
        if len(file_sizes) > 1:
            sizes = list(file_sizes.values())
            max_size = max(sizes)
            min_size = min(sizes)
            size_ratio = max_size / min_size if min_size > 0 else float('inf')
            
            is_consistent = size_ratio < 10  # Less than 10x difference
        else:
            is_consistent = True
        
        return {
            'is_consistent': is_consistent,
            'details': file_sizes,
            'size_ratio': size_ratio if len(file_sizes) > 1 else 1.0,
            'inconsistency_reason': 'Large size differences between model files' if not is_consistent else None
        }
    
    def _check_config_consistency(self) -> Dict[str, Any]:
        """Check configuration file consistency"""
        config_data = {}
        
        # Load all config files
        config_files = [
            "weight_config.json",
            "Tantra_real_config.json",
            "training_config.json",
            "consolidated_config.json"
        ]
        
        for config_file in config_files:
            config_path = self.model_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data[str(config_path)] = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load {config_path}: {e}")
        
        # Check for conflicting values
        conflicts = []
        
        # Check parameter counts
        param_counts = []
        for path, data in config_data.items():
            if 'model_info' in data and 'total_parameters' in data['model_info']:
                param_counts.append((path, data['model_info']['total_parameters']))
        
        if len(set(count for _, count in param_counts)) > 1:
            conflicts.append("Inconsistent parameter counts across config files")
        
        # Check vocabulary sizes
        vocab_sizes = []
        for path, data in config_data.items():
            if 'model_info' in data and 'vocab_size' in data['model_info']:
                vocab_sizes.append((path, data['model_info']['vocab_size']))
        
        if len(set(size for _, size in vocab_sizes)) > 1:
            conflicts.append("Inconsistent vocabulary sizes across config files")
        
        return {
            'is_consistent': len(conflicts) == 0,
            'details': config_data,
            'conflicts': conflicts,
            'inconsistency_reason': '; '.join(conflicts) if conflicts else None
        }
    
    def _get_required_config_fields(self, config_name: str) -> List[str]:
        """Get required fields for specific config file"""
        field_map = {
            'weight_config.json': ['weights', 'last_updated'],
            'Tantra_real_config.json': ['tantra_config', 'model_info'],
            'training_config.json': ['model_architecture', 'training_parameters'],
            'consolidated_config.json': ['metadata', 'summary']
        }
        return field_map.get(config_name, [])
    
    def _validate_config_types(self, config_data: Dict, config_name: str) -> bool:
        """Validate config value types"""
        # Basic type validation
        if config_name == 'weight_config.json':
            return isinstance(config_data.get('weights'), dict)
        elif config_name == 'Tantra_real_config.json':
            return isinstance(config_data.get('tantra_config'), dict) and isinstance(config_data.get('model_info'), dict)
        elif config_name == 'training_config.json':
            return isinstance(config_data.get('model_architecture'), dict)
        elif config_name == 'consolidated_config.json':
            return isinstance(config_data.get('metadata'), dict)
        
        return True
    
    def _calculate_validation_stats(self, validation_results: Dict[str, Any]) -> None:
        """Calculate validation statistics"""
        total_checks = 0
        passed_checks = 0
        errors = 0
        warnings = 0
        
        # Count from file validations
        for file_path, file_validation in validation_results['file_validations'].items():
            total_checks += file_validation['total_checks']
            passed_checks += file_validation['checks_passed']
            errors += len(file_validation['errors'])
            warnings += len(file_validation['warnings'])
        
        # Count from cross-file consistency
        consistency = validation_results['cross_file_consistency']
        for check_name, check_result in consistency.items():
            if isinstance(check_result, dict) and 'is_consistent' in check_result:
                total_checks += 1
                if check_result['is_consistent']:
                    passed_checks += 1
                else:
                    warnings += 1
        
        validation_results['total_checks'] = total_checks
        validation_results['passed_checks'] = passed_checks
        validation_results['failed_checks'] = total_checks - passed_checks
        validation_results['warnings'] = warnings
        validation_results['errors'] = errors
        
        # Determine overall status
        if errors > 0:
            validation_results['overall_status'] = 'error'
        elif warnings > 0:
            validation_results['overall_status'] = 'warning'
        elif passed_checks == total_checks:
            validation_results['overall_status'] = 'valid'
        else:
            validation_results['overall_status'] = 'unknown'
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # File validation recommendations
        for file_path, file_validation in validation_results['file_validations'].items():
            if not file_validation['is_valid']:
                recommendations.append(f"Fix errors in {Path(file_path).name}: {', '.join(file_validation['errors'])}")
            
            if file_validation['warnings']:
                recommendations.append(f"Review warnings in {Path(file_path).name}: {', '.join(file_validation['warnings'])}")
        
        # Consistency recommendations
        consistency = validation_results['cross_file_consistency']
        for check_name, check_result in consistency.items():
            if isinstance(check_result, dict) and not check_result.get('is_consistent', True):
                recommendations.append(f"Resolve {check_name}: {check_result.get('inconsistency_reason', 'Inconsistency detected')}")
        
        # General recommendations
        if validation_results['overall_status'] == 'error':
            recommendations.append("Address all errors before using the model")
        elif validation_results['overall_status'] == 'warning':
            recommendations.append("Review warnings and consider addressing them")
        
        return recommendations
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules configuration"""
        return {
            'max_file_size_mb': 10000,
            'max_parameter_count': 1_000_000_000,
            'max_vocab_size': 1_000_000,
            'min_parameter_count': 1,
            'min_vocab_size': 1,
            'required_model_components': ['embeddings', 'attention', 'output'],
            'allowed_file_types': ['.pt', '.pth', '.json'],
            'max_size_ratio': 10
        }
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report"""
        report = f"""
# Model Validation Report

## Overall Status
- **Status**: {validation_results['overall_status'].upper()}
- **Total Checks**: {validation_results['total_checks']}
- **Passed**: {validation_results['passed_checks']}
- **Failed**: {validation_results['failed_checks']}
- **Warnings**: {validation_results['warnings']}
- **Errors**: {validation_results['errors']}

## File Validations
"""
        
        for file_path, file_validation in validation_results['file_validations'].items():
            file_name = Path(file_path).name
            status = "✅ VALID" if file_validation['is_valid'] else "❌ INVALID"
            
            report += f"\n### {file_name}\n"
            report += f"- **Status**: {status}\n"
            report += f"- **Checks**: {file_validation['checks_passed']}/{file_validation['total_checks']} passed\n"
            
            if file_validation['errors']:
                report += "- **Errors**:\n"
                for error in file_validation['errors']:
                    report += f"  - ❌ {error}\n"
            
            if file_validation['warnings']:
                report += "- **Warnings**:\n"
                for warning in file_validation['warnings']:
                    report += f"  - ⚠️ {warning}\n"
        
        # Cross-file consistency
        report += "\n## Cross-File Consistency\n"
        consistency = validation_results['cross_file_consistency']
        
        for check_name, check_result in consistency.items():
            if isinstance(check_result, dict) and 'is_consistent' in check_result:
                status = "✅ CONSISTENT" if check_result['is_consistent'] else "❌ INCONSISTENT"
                report += f"- **{check_name.replace('_', ' ').title()}**: {status}\n"
                
                if not check_result['is_consistent'] and check_result.get('inconsistency_reason'):
                    report += f"  - Reason: {check_result['inconsistency_reason']}\n"
        
        # Recommendations
        if validation_results['recommendations']:
            report += "\n## Recommendations\n"
            for i, recommendation in enumerate(validation_results['recommendations'], 1):
                report += f"{i}. {recommendation}\n"
        
        report += f"\n---\n*Validation completed at {validation_results['validation_timestamp']}*"
        
        return report

def main():
    """Main function for testing"""
    validator = ValidationSystem()
    
    print("Running comprehensive validation...")
    results = validator.validate_all()
    
    # Generate report
    report = validator.generate_validation_report(results)
    print(report)
    
    # Save report
    with open("validation_report.md", "w") as f:
        f.write(report)
    
    print(f"\nValidation complete! Status: {results['overall_status']}")
    print(f"Report saved to validation_report.md")

if __name__ == "__main__":
    main()