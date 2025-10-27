"""
Dynamic Model CLI - Command-line interface for dynamic model analysis
Easy access to all dynamic model analysis tools
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from .dynamic_model_orchestrator import DynamicModelOrchestrator

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Dynamic Model Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis
  python -m src.utils.dynamic_model_cli analyze --complete
  
  # Run quick analysis
  python -m src.utils.dynamic_model_cli analyze --quick
  
  # Analyze specific model
  python -m src.utils.dynamic_model_cli analyze --model Model/weights/Tantra_v1.0.pt
  
  # Update model info only
  python -m src.utils.dynamic_model_cli update
  
  # Validate model
  python -m src.utils.dynamic_model_cli validate
  
  # Generate report from existing results
  python -m src.utils.dynamic_model_cli report
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run model analysis')
    analyze_parser.add_argument('--complete', action='store_true', help='Run complete analysis')
    analyze_parser.add_argument('--quick', action='store_true', help='Run quick analysis')
    analyze_parser.add_argument('--model', type=str, help='Path to model file')
    analyze_parser.add_argument('--output', type=str, help='Output file for results')
    analyze_parser.add_argument('--report', action='store_true', help='Generate report after analysis')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update model information')
    update_parser.add_argument('--model', type=str, help='Path to model file')
    update_parser.add_argument('--force', action='store_true', help='Force update even if recent analysis exists')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate model files')
    validate_parser.add_argument('--output', type=str, help='Output file for validation results')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate analysis report')
    report_parser.add_argument('--input', type=str, help='Input JSON file with analysis results')
    report_parser.add_argument('--output', type=str, help='Output file for report')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show analysis status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Initialize orchestrator
        orchestrator = DynamicModelOrchestrator()
        
        if args.command == 'analyze':
            return handle_analyze(orchestrator, args)
        elif args.command == 'update':
            return handle_update(orchestrator, args)
        elif args.command == 'validate':
            return handle_validate(orchestrator, args)
        elif args.command == 'report':
            return handle_report(orchestrator, args)
        elif args.command == 'info':
            return handle_info(orchestrator, args)
        elif args.command == 'status':
            return handle_status(orchestrator, args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

def handle_analyze(orchestrator: DynamicModelOrchestrator, args) -> int:
    """Handle analyze command"""
    if not args.complete and not args.quick:
        print("Please specify --complete or --quick")
        return 1
    
    print("🔍 Starting model analysis...")
    
    if args.complete:
        print("Running complete analysis...")
        results = orchestrator.run_complete_analysis(args.model)
    else:
        print("Running quick analysis...")
        results = orchestrator.run_quick_analysis(args.model)
    
    if results['success']:
        print("✅ Analysis completed successfully!")
        
        # Show summary
        if 'summary' in results and results['summary']:
            summary = results['summary']
            print(f"\n📊 Summary:")
            print(f"  Parameters: {summary.get('total_parameters', 'N/A'):,}")
            print(f"  File Size: {summary.get('file_size_mb', 'N/A'):.2f} MB")
            print(f"  Memory Size: {summary.get('memory_size_mb', 'N/A'):.2f} MB")
            print(f"  Compression: {summary.get('compression_ratio', 'N/A'):.2f}x")
            print(f"  Model Type: {summary.get('model_type', 'N/A')}")
            print(f"  Valid: {'✅' if summary.get('is_valid', False) else '❌'}")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"📁 Results saved to {args.output}")
        
        # Generate report if requested
        if args.report:
            report = orchestrator.generate_comprehensive_report(results)
            report_file = args.output.replace('.json', '.md') if args.output else 'analysis_report.md'
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to {report_file}")
        
        return 0
    else:
        print("❌ Analysis failed!")
        if 'errors' in results:
            for error in results['errors']:
                print(f"  Error: {error}")
        return 1

def handle_update(orchestrator: DynamicModelOrchestrator, args) -> int:
    """Handle update command"""
    print("🔄 Updating model information...")
    
    result = orchestrator.update_model_info(args.model)
    
    if result['success']:
        print("✅ Model information updated successfully!")
        
        # Show updated info
        if 'model_analysis' in result:
            model_analysis = result['model_analysis']
            print(f"\n📊 Updated Information:")
            print(f"  Parameters: {model_analysis.total_parameters:,}")
            print(f"  Architecture: {model_analysis.actual_d_model}D, {model_analysis.actual_n_layers} layers")
            print(f"  Vocabulary: {model_analysis.actual_vocab_size:,} tokens")
            print(f"  Memory: {model_analysis.memory_size_mb:.2f} MB")
        
        if 'configuration_update' in result:
            config_update = result['configuration_update']
            print(f"  Updated Files: {len(config_update['updated_files'])}")
            print(f"  Created Files: {len(config_update['created_files'])}")
        
        return 0
    else:
        print(f"❌ Update failed: {result.get('error', 'Unknown error')}")
        return 1

def handle_validate(orchestrator: DynamicModelOrchestrator, args) -> int:
    """Handle validate command"""
    print("🔍 Running model validation...")
    
    validation = orchestrator.validate_model()
    
    if validation.get('success', True):  # Validation can succeed even with warnings
        print(f"✅ Validation completed!")
        print(f"  Status: {validation['overall_status'].upper()}")
        print(f"  Checks: {validation['passed_checks']}/{validation['total_checks']} passed")
        print(f"  Warnings: {validation['warnings']}")
        print(f"  Errors: {validation['errors']}")
        
        if validation['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(validation['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # Save validation results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(validation, f, indent=2, default=str)
            print(f"📁 Validation results saved to {args.output}")
        
        return 0
    else:
        print(f"❌ Validation failed: {validation.get('error', 'Unknown error')}")
        return 1

def handle_report(orchestrator: DynamicModelOrchestrator, args) -> int:
    """Handle report command"""
    print("📄 Generating analysis report...")
    
    # Load results if input file specified
    if args.input:
        try:
            with open(args.input, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load input file: {e}")
            return 1
    else:
        # Use existing results
        results = orchestrator.analysis_results
        if not results:
            print("❌ No analysis results available. Run analysis first or specify --input")
            return 1
    
    # Generate report
    report = orchestrator.generate_comprehensive_report(results)
    
    # Save report
    output_file = args.output or 'model_analysis_report.md'
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Report generated: {output_file}")
    return 0

def handle_info(orchestrator: DynamicModelOrchestrator, args) -> int:
    """Handle info command"""
    print("ℹ️  Model Information")
    
    info = orchestrator.get_model_info()
    
    if 'error' in info:
        print(f"❌ {info['error']}")
        return 1
    
    print(f"  Last Analysis: {info.get('last_analysis', 'Never')}")
    print(f"  Model Path: {info.get('model_path', 'N/A')}")
    print(f"  Success: {'✅' if info.get('success', False) else '❌'}")
    
    if 'summary' in info and info['summary']:
        summary = info['summary']
        print(f"\n📊 Summary:")
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                if 'parameters' in key.lower():
                    print(f"  {key}: {value:,}")
                elif 'size' in key.lower() or 'mb' in key.lower():
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
    
    return 0

def handle_status(orchestrator: DynamicModelOrchestrator, args) -> int:
    """Handle status command"""
    print("📊 Analysis Status")
    
    info = orchestrator.get_model_info()
    
    if 'error' in info:
        print(f"❌ {info['error']}")
        return 1
    
    # Check if analysis exists
    if not info.get('last_analysis'):
        print("❌ No analysis has been run yet")
        print("💡 Run 'analyze --complete' or 'analyze --quick' to start")
        return 0
    
    print(f"✅ Last analysis: {info['last_analysis']}")
    print(f"📁 Model: {info.get('model_path', 'N/A')}")
    print(f"🎯 Success: {'✅' if info.get('success', False) else '❌'}")
    
    # Check file status
    model_path = info.get('model_path')
    if model_path and Path(model_path).exists():
        file_size = Path(model_path).stat().st_size / (1024 * 1024)
        print(f"📦 File size: {file_size:.2f} MB")
    else:
        print("⚠️  Model file not found")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())