"""
Dynamic Model Orchestrator - Main orchestrator for all dynamic model analysis
Coordinates all analysis tools and provides unified interface
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from .dynamic_model_analyzer import DynamicModelAnalyzer, ModelAnalysis
from .vocabulary_analyzer import VocabularyAnalyzer
from .model_size_calculator import ModelSizeCalculator
from .configuration_updater import ConfigurationUpdater
from .validation_system import ValidationSystem

logger = logging.getLogger(__name__)

class DynamicModelOrchestrator:
    """Main orchestrator for dynamic model analysis"""
    
    def __init__(self, model_dir: str = "Model"):
        self.model_dir = Path(model_dir)
        self.weights_dir = self.model_dir / "weights"
        
        # Initialize all analysis tools
        self.model_analyzer = DynamicModelAnalyzer(model_dir)
        self.vocab_analyzer = VocabularyAnalyzer(model_dir)
        self.size_calculator = ModelSizeCalculator(model_dir)
        self.config_updater = ConfigurationUpdater(model_dir)
        self.validator = ValidationSystem(model_dir)
        
        # Results storage
        self.analysis_results = {}
        self.last_analysis_time = None
    
    def run_complete_analysis(self, model_path: str = None) -> Dict[str, Any]:
        """Run complete dynamic model analysis"""
        try:
            logger.info("Starting complete dynamic model analysis...")
            
            # Determine model path
            if model_path is None:
                model_path = self._find_primary_model()
            
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Run all analyses
            results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'model_path': model_path,
                'model_analysis': None,
                'vocabulary_analysis': None,
                'size_analysis': None,
                'configuration_update': None,
                'validation': None,
                'summary': {},
                'success': False,
                'errors': []
            }
            
            # 1. Model Analysis
            logger.info("Running model analysis...")
            try:
                model_analysis = self.model_analyzer.analyze_model(model_path)
                results['model_analysis'] = model_analysis
                logger.info(f"Model analysis completed: {model_analysis.total_parameters:,} parameters")
            except Exception as e:
                error_msg = f"Model analysis failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
                return results
            
            # 2. Vocabulary Analysis
            logger.info("Running vocabulary analysis...")
            try:
                vocab_analysis = self.vocab_analyzer.analyze_vocabulary()
                results['vocabulary_analysis'] = vocab_analysis
                logger.info(f"Vocabulary analysis completed: {vocab_analysis['size']:,} tokens")
            except Exception as e:
                error_msg = f"Vocabulary analysis failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
            
            # 3. Size Analysis
            logger.info("Running size analysis...")
            try:
                size_analysis = self.size_calculator.calculate_model_size(model_path)
                results['size_analysis'] = size_analysis
                logger.info(f"Size analysis completed: {size_analysis['file_analysis']['size_mb']:.2f} MB file, {size_analysis['memory_analysis']['memory_mb']:.2f} MB memory")
            except Exception as e:
                error_msg = f"Size analysis failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
            
            # 4. Configuration Update
            logger.info("Updating configurations...")
            try:
                config_update = self.config_updater.update_all_configurations(
                    results['model_analysis'],
                    results['vocabulary_analysis'],
                    results['size_analysis']
                )
                results['configuration_update'] = config_update
                logger.info(f"Configuration update completed: {len(config_update['updated_files'])} files updated")
            except Exception as e:
                error_msg = f"Configuration update failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
            
            # 5. Validation
            logger.info("Running validation...")
            try:
                validation = self.validator.validate_all()
                results['validation'] = validation
                logger.info(f"Validation completed: {validation['overall_status']}")
            except Exception as e:
                error_msg = f"Validation failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
            
            # 6. Generate Summary
            results['summary'] = self._generate_analysis_summary(results)
            results['success'] = len(results['errors']) == 0
            
            # Store results
            self.analysis_results = results
            self.last_analysis_time = datetime.now()
            
            logger.info("Complete dynamic model analysis finished")
            return results
            
        except Exception as e:
            logger.error(f"Complete analysis failed: {e}")
            return {
                'analysis_timestamp': datetime.now().isoformat(),
                'model_path': model_path,
                'success': False,
                'errors': [str(e)],
                'summary': {}
            }
    
    def run_quick_analysis(self, model_path: str = None) -> Dict[str, Any]:
        """Run quick analysis (model + size only)"""
        try:
            logger.info("Starting quick model analysis...")
            
            if model_path is None:
                model_path = self._find_primary_model()
            
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'model_path': model_path,
                'analysis_type': 'quick',
                'success': False,
                'errors': []
            }
            
            # Model Analysis
            try:
                model_analysis = self.model_analyzer.analyze_model(model_path)
                results['model_analysis'] = model_analysis
            except Exception as e:
                results['errors'].append(f"Model analysis failed: {e}")
                return results
            
            # Size Analysis
            try:
                size_analysis = self.size_calculator.calculate_model_size(model_path)
                results['size_analysis'] = size_analysis
            except Exception as e:
                results['errors'].append(f"Size analysis failed: {e}")
                return results
            
            # Quick Summary
            results['summary'] = {
                'total_parameters': model_analysis.total_parameters,
                'file_size_mb': size_analysis['file_analysis']['size_mb'],
                'memory_size_mb': size_analysis['memory_analysis']['memory_mb'],
                'compression_ratio': size_analysis['compression_analysis']['compression_ratio'],
                'model_type': model_analysis.model_type,
                'is_valid': model_analysis.is_valid
            }
            
            results['success'] = len(results['errors']) == 0
            logger.info("Quick analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Quick analysis failed: {e}")
            return {
                'analysis_timestamp': datetime.now().isoformat(),
                'model_path': model_path,
                'analysis_type': 'quick',
                'success': False,
                'errors': [str(e)],
                'summary': {}
            }
    
    def update_model_info(self, model_path: str = None) -> Dict[str, Any]:
        """Update model information files only"""
        try:
            logger.info("Updating model information...")
            
            if model_path is None:
                model_path = self._find_primary_model()
            
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Run minimal analysis
            model_analysis = self.model_analyzer.analyze_model(model_path)
            size_analysis = self.size_calculator.calculate_model_size(model_path)
            
            # Update configurations
            config_update = self.config_updater.update_all_configurations(
                model_analysis,
                {'size': 0, 'tokens': [], 'token_types': {}, 'language_hints': {}, 'coverage_analysis': {}},  # Empty vocab
                size_analysis
            )
            
            logger.info("Model information updated")
            return {
                'success': True,
                'model_analysis': model_analysis,
                'size_analysis': size_analysis,
                'configuration_update': config_update
            }
            
        except Exception as e:
            logger.error(f"Model info update failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def validate_model(self) -> Dict[str, Any]:
        """Run validation only"""
        try:
            logger.info("Running model validation...")
            validation = self.validator.validate_all()
            logger.info(f"Validation completed: {validation['overall_status']}")
            return validation
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_comprehensive_report(self, results: Dict[str, Any] = None) -> str:
        """Generate comprehensive analysis report"""
        if results is None:
            results = self.analysis_results
        
        if not results:
            return "No analysis results available. Run analysis first."
        
        report = f"""
# Dynamic Model Analysis Report

## Analysis Overview
- **Timestamp**: {results['analysis_timestamp']}
- **Model Path**: {results['model_path']}
- **Status**: {'✅ SUCCESS' if results['success'] else '❌ FAILED'}

## Summary
"""
        
        if 'summary' in results and results['summary']:
            summary = results['summary']
            report += f"""
- **Total Parameters**: {summary.get('total_parameters', 'N/A'):,}
- **File Size**: {summary.get('file_size_mb', 'N/A'):.2f} MB
- **Memory Size**: {summary.get('memory_size_mb', 'N/A'):.2f} MB
- **Compression Ratio**: {summary.get('compression_ratio', 'N/A'):.2f}x
- **Model Type**: {summary.get('model_type', 'N/A')}
- **Validation Status**: {'✅ Valid' if summary.get('is_valid', False) else '❌ Invalid'}
"""
        
        # Model Analysis Section
        if 'model_analysis' in results and results['model_analysis']:
            model_analysis = results['model_analysis']
            report += f"""
## Model Analysis
- **Architecture**: {model_analysis.actual_d_model}D, {model_analysis.actual_n_layers} layers, {model_analysis.actual_n_heads} heads
- **Vocabulary**: {model_analysis.actual_vocab_size:,} tokens
- **Max Sequence Length**: {model_analysis.actual_max_seq_length}
- **Total Parameters**: {model_analysis.total_parameters:,}
- **Memory Usage**: {model_analysis.memory_size_mb:.2f} MB
- **Validation**: {'✅ Valid' if model_analysis.is_valid else '❌ Invalid'}
"""
            
            if model_analysis.validation_errors:
                report += "\n### Validation Errors\n"
                for error in model_analysis.validation_errors:
                    report += f"- ❌ {error}\n"
        
        # Size Analysis Section
        if 'size_analysis' in results and results['size_analysis']:
            size_analysis = results['size_analysis']
            file_analysis = size_analysis['file_analysis']
            memory_analysis = size_analysis['memory_analysis']
            compression_analysis = size_analysis['compression_analysis']
            
            report += f"""
## Size Analysis
- **File Size**: {file_analysis['size_mb']:.2f} MB ({file_analysis['size_bytes']:,} bytes)
- **Memory Size**: {memory_analysis['memory_mb']:.2f} MB ({memory_analysis['memory_bytes']:,} bytes)
- **Compression Ratio**: {compression_analysis['compression_ratio']:.2f}x
- **Compression Type**: {compression_analysis['compression_type']}
- **Efficiency Rating**: {size_analysis['summary']['efficiency_rating']}
- **Size Category**: {size_analysis['summary']['size_category']}
"""
        
        # Vocabulary Analysis Section
        if 'vocabulary_analysis' in results and results['vocabulary_analysis']:
            vocab_analysis = results['vocabulary_analysis']
            report += f"""
## Vocabulary Analysis
- **Vocabulary Size**: {vocab_analysis['size']:,} tokens
- **Source**: {vocab_analysis.get('file_source', 'N/A')}
- **Status**: {'✅ Valid' if vocab_analysis['is_valid'] else '❌ Invalid'}
"""
            
            if vocab_analysis.get('token_types'):
                report += "\n### Token Types\n"
                for token_type, count in vocab_analysis['token_types'].items():
                    percentage = (count / vocab_analysis['size']) * 100 if vocab_analysis['size'] > 0 else 0
                    report += f"- **{token_type.title()}**: {count:,} ({percentage:.1f}%)\n"
        
        # Configuration Update Section
        if 'configuration_update' in results and results['configuration_update']:
            config_update = results['configuration_update']
            report += f"""
## Configuration Update
- **Updated Files**: {len(config_update['updated_files'])}
- **Created Files**: {len(config_update['created_files'])}
- **Backup Created**: {'✅ Yes' if config_update['backup_created'] else '❌ No'}
"""
            
            if config_update['updated_files']:
                report += "\n### Updated Files\n"
                for file_path in config_update['updated_files']:
                    report += f"- {Path(file_path).name}\n"
            
            if config_update['created_files']:
                report += "\n### Created Files\n"
                for file_path in config_update['created_files']:
                    report += f"- {Path(file_path).name}\n"
        
        # Validation Section
        if 'validation' in results and results['validation']:
            validation = results['validation']
            report += f"""
## Validation Results
- **Overall Status**: {validation['overall_status'].upper()}
- **Total Checks**: {validation['total_checks']}
- **Passed**: {validation['passed_checks']}
- **Failed**: {validation['failed_checks']}
- **Warnings**: {validation['warnings']}
- **Errors**: {validation['errors']}
"""
            
            if validation['recommendations']:
                report += "\n### Recommendations\n"
                for i, recommendation in enumerate(validation['recommendations'], 1):
                    report += f"{i}. {recommendation}\n"
        
        # Errors Section
        if results['errors']:
            report += "\n## Errors\n"
            for i, error in enumerate(results['errors'], 1):
                report += f"{i}. {error}\n"
        
        report += f"\n---\n*Report generated at {datetime.now().isoformat()}*"
        return report
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary"""
        summary = {}
        
        # Extract key information from results
        if 'model_analysis' in results and results['model_analysis']:
            model_analysis = results['model_analysis']
            summary.update({
                'total_parameters': model_analysis.total_parameters,
                'model_type': model_analysis.model_type,
                'is_valid': model_analysis.is_valid,
                'd_model': model_analysis.actual_d_model,
                'n_layers': model_analysis.actual_n_layers,
                'n_heads': model_analysis.actual_n_heads,
                'vocab_size': model_analysis.actual_vocab_size,
                'max_seq_length': model_analysis.actual_max_seq_length
            })
        
        if 'size_analysis' in results and results['size_analysis']:
            size_analysis = results['size_analysis']
            summary.update({
                'file_size_mb': size_analysis['file_analysis']['size_mb'],
                'memory_size_mb': size_analysis['memory_analysis']['memory_mb'],
                'compression_ratio': size_analysis['compression_analysis']['compression_ratio'],
                'efficiency_rating': size_analysis['summary']['efficiency_rating'],
                'size_category': size_analysis['summary']['size_category']
            })
        
        if 'vocabulary_analysis' in results and results['vocabulary_analysis']:
            vocab_analysis = results['vocabulary_analysis']
            summary.update({
                'vocabulary_size': vocab_analysis['size'],
                'vocabulary_valid': vocab_analysis['is_valid']
            })
        
        if 'validation' in results and results['validation']:
            validation = results['validation']
            summary.update({
                'validation_status': validation['overall_status'],
                'validation_score': validation.get('validation_score', 0),
                'checks_passed': validation['passed_checks'],
                'total_checks': validation['total_checks']
            })
        
        return summary
    
    def _find_primary_model(self) -> Optional[str]:
        """Find the primary model file"""
        # Look for common model file names
        model_candidates = [
            "Tantra_v1.0.pt",
            "model.pt",
            "weights.pt",
            "pytorch_model.bin"
        ]
        
        for candidate in model_candidates:
            model_path = self.weights_dir / candidate
            if model_path.exists():
                return str(model_path)
        
        # Look for any .pt file
        pt_files = list(self.weights_dir.glob("*.pt"))
        if pt_files:
            return str(pt_files[0])
        
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        if not self.analysis_results:
            return {'error': 'No analysis results available. Run analysis first.'}
        
        return {
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'model_path': self.analysis_results.get('model_path'),
            'summary': self.analysis_results.get('summary', {}),
            'success': self.analysis_results.get('success', False)
        }
    
    def save_analysis_results(self, file_path: str = None) -> str:
        """Save analysis results to file"""
        if not self.analysis_results:
            raise ValueError("No analysis results to save")
        
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"model_analysis_results_{timestamp}.json"
        
        # Convert ModelAnalysis objects to dict for JSON serialization
        results_to_save = self.analysis_results.copy()
        
        if 'model_analysis' in results_to_save and hasattr(results_to_save['model_analysis'], '__dict__'):
            results_to_save['model_analysis'] = results_to_save['model_analysis'].__dict__
        
        with open(file_path, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        logger.info(f"Analysis results saved to {file_path}")
        return file_path

def main():
    """Main function for testing"""
    orchestrator = DynamicModelOrchestrator()
    
    print("Running complete dynamic model analysis...")
    results = orchestrator.run_complete_analysis()
    
    if results['success']:
        print("✅ Analysis completed successfully!")
        
        # Generate and save report
        report = orchestrator.generate_comprehensive_report(results)
        print("\n" + "="*50)
        print(report)
        print("="*50)
        
        # Save report to file
        with open("comprehensive_model_analysis_report.md", "w") as f:
            f.write(report)
        
        # Save results to JSON
        results_file = orchestrator.save_analysis_results()
        
        print(f"\nReports saved:")
        print(f"- Comprehensive report: comprehensive_model_analysis_report.md")
        print(f"- Results JSON: {results_file}")
    else:
        print("❌ Analysis failed!")
        print("Errors:")
        for error in results['errors']:
            print(f"- {error}")

if __name__ == "__main__":
    main()