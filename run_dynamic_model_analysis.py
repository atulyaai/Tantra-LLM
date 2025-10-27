#!/usr/bin/env python3
"""
Dynamic Model Analysis Runner
Main script to run the complete dynamic model analysis system
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.dynamic_model_orchestrator import DynamicModelOrchestrator

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dynamic_model_analysis.log')
        ]
    )

def main():
    """Main function"""
    print("🚀 Dynamic Model Analysis System")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize orchestrator
        print("🔧 Initializing dynamic model orchestrator...")
        orchestrator = DynamicModelOrchestrator()
        
        # Check if model exists
        model_path = "Model/weights/Tantra_v1.0.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("💡 Please ensure the model file exists before running analysis")
            return 1
        
        print(f"✅ Model found: {model_path}")
        
        # Run complete analysis
        print("\n🔍 Running complete dynamic model analysis...")
        print("This includes:")
        print("  - Model parameter analysis")
        print("  - Vocabulary analysis")
        print("  - Size calculation and validation")
        print("  - Configuration file updates")
        print("  - Cross-file consistency validation")
        print("  - Comprehensive reporting")
        
        results = orchestrator.run_complete_analysis(model_path)
        
        if results['success']:
            print("\n✅ Analysis completed successfully!")
            
            # Display summary
            if 'summary' in results and results['summary']:
                summary = results['summary']
                print(f"\n📊 Analysis Summary:")
                print(f"  🧠 Total Parameters: {summary.get('total_parameters', 'N/A'):,}")
                print(f"  📁 File Size: {summary.get('file_size_mb', 'N/A'):.2f} MB")
                print(f"  💾 Memory Size: {summary.get('memory_size_mb', 'N/A'):.2f} MB")
                print(f"  📦 Compression Ratio: {summary.get('compression_ratio', 'N/A'):.2f}x")
                print(f"  🤖 Model Type: {summary.get('model_type', 'N/A')}")
                print(f"  ✅ Validation: {'Valid' if summary.get('is_valid', False) else 'Invalid'}")
            
            # Generate comprehensive report
            print("\n📄 Generating comprehensive report...")
            report = orchestrator.generate_comprehensive_report(results)
            
            # Save report
            report_file = "dynamic_model_analysis_report.md"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"📄 Report saved: {report_file}")
            
            # Save results
            results_file = orchestrator.save_analysis_results()
            print(f"💾 Results saved: {results_file}")
            
            # Show validation status
            if 'validation' in results and results['validation']:
                validation = results['validation']
                print(f"\n🔍 Validation Status:")
                print(f"  Status: {validation['overall_status'].upper()}")
                print(f"  Checks: {validation['passed_checks']}/{validation['total_checks']} passed")
                print(f"  Warnings: {validation['warnings']}")
                print(f"  Errors: {validation['errors']}")
                
                if validation['recommendations']:
                    print(f"\n💡 Recommendations:")
                    for i, rec in enumerate(validation['recommendations'], 1):
                        print(f"  {i}. {rec}")
            
            # Show configuration update status
            if 'configuration_update' in results and results['configuration_update']:
                config_update = results['configuration_update']
                print(f"\n⚙️  Configuration Updates:")
                print(f"  Updated Files: {len(config_update['updated_files'])}")
                print(f"  Created Files: {len(config_update['created_files'])}")
                print(f"  Backup Created: {'Yes' if config_update['backup_created'] else 'No'}")
                
                if config_update['updated_files']:
                    print(f"  Updated:")
                    for file_path in config_update['updated_files']:
                        print(f"    - {Path(file_path).name}")
                
                if config_update['created_files']:
                    print(f"  Created:")
                    for file_path in config_update['created_files']:
                        print(f"    - {Path(file_path).name}")
            
            print(f"\n🎉 Dynamic model analysis completed successfully!")
            print(f"📁 Check the following files:")
            print(f"  - {report_file} (comprehensive report)")
            print(f"  - {results_file} (detailed results)")
            print(f"  - dynamic_model_analysis.log (analysis log)")
            
            return 0
            
        else:
            print("\n❌ Analysis failed!")
            if 'errors' in results:
                print("Errors:")
                for error in results['errors']:
                    print(f"  - {error}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Analysis cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Analysis failed with error: {e}")
        print("Check dynamic_model_analysis.log for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())