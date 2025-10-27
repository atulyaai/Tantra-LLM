"""
Command Line Interface for OCR-Native LLM
Interactive CLI for experimentation and testing
"""

import argparse
import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path

from src.interfaces.conversational import OCRNativeConversational, create_conversational_interface
from src.benchmarks.performance_benchmark import OCRNativeBenchmark, run_quick_benchmark
from src.architectures.transformer_variants import TransformerVariantConfig
from src.configs.ocr_config import ConfigManager
from src.utils.error_handler import logger


class OCRNativeCLI:
    """Command Line Interface for OCR-Native LLM"""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.conversational_interface = None
        self.current_session = None
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="OCR-Native LLM Command Line Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Start interactive chat
  python -m src.interfaces.cli_interface chat --variant mamba
  
  # Run quick benchmark
  python -m src.interfaces.cli_interface benchmark --quick
  
  # Compare model variants
  python -m src.interfaces.cli_interface compare
  
  # Test specific model
  python -m src.interfaces.cli_interface test --size small --variant hybrid
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Chat command
        chat_parser = subparsers.add_parser('chat', help='Start interactive chat')
        chat_parser.add_argument('--variant', choices=['standard', 'mamba', 'hybrid', 'memory_enhanced'], 
                               default='standard', help='Model variant to use')
        chat_parser.add_argument('--size', choices=['small', 'default', 'large'], 
                               default='small', help='Model size')
        chat_parser.add_argument('--user-id', default='cli_user', help='User ID for session')
        
        # Benchmark command
        benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
        benchmark_parser.add_argument('--quick', action='store_true', help='Run quick benchmark only')
        benchmark_parser.add_argument('--output-dir', default='benchmark_results', help='Output directory')
        benchmark_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
        
        # Compare command
        compare_parser = subparsers.add_parser('compare', help='Compare model variants')
        compare_parser.add_argument('--size', choices=['small', 'default', 'large'], 
                                  default='small', help='Model size to compare')
        
        # Test command
        test_parser = subparsers.add_parser('test', help='Test specific model')
        test_parser.add_argument('--size', choices=['small', 'default', 'large'], 
                               default='small', help='Model size')
        test_parser.add_argument('--variant', choices=['standard', 'mamba', 'hybrid', 'memory_enhanced'], 
                               default='standard', help='Model variant')
        test_parser.add_argument('--input', help='Input text to test')
        
        # Info command
        info_parser = subparsers.add_parser('info', help='Show system information')
        
        return parser
    
    def run(self, args: Optional[list] = None):
        """Run the CLI"""
        if args is None:
            args = sys.argv[1:]
        
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return
        
        try:
            if parsed_args.command == 'chat':
                self._run_chat(parsed_args)
            elif parsed_args.command == 'benchmark':
                self._run_benchmark(parsed_args)
            elif parsed_args.command == 'compare':
                self._run_compare(parsed_args)
            elif parsed_args.command == 'test':
                self._run_test(parsed_args)
            elif parsed_args.command == 'info':
                self._run_info(parsed_args)
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
        except Exception as e:
            logger.error(f"CLI error: {e}")
            print(f"Error: {e}")
    
    def _run_chat(self, args):
        """Run interactive chat"""
        print("🔤 OCR-Native LLM Interactive Chat")
        print("=" * 50)
        print(f"Model: {args.size} ({args.variant})")
        print("Type 'quit' to exit, 'help' for commands")
        print("-" * 50)
        
        # Initialize conversational interface
        self.conversational_interface = create_conversational_interface(args.size, args.variant)
        self.current_session = self.conversational_interface.start_conversation(args.user_id)
        
        # Chat loop
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                elif user_input.lower() == 'help':
                    self._print_chat_help()
                    continue
                elif user_input.lower() == 'info':
                    self._print_session_info()
                    continue
                elif user_input.lower() == 'clear':
                    self._clear_conversation()
                    continue
                elif user_input.lower() == 'variants':
                    self._print_available_variants()
                    continue
                elif user_input.lower().startswith('switch '):
                    variant = user_input[7:].strip()
                    self._switch_variant(variant)
                    continue
                elif not user_input:
                    continue
                
                # Send message and get response
                response = self.conversational_interface.send_message(
                    self.current_session, user_input
                )
                
                print(f"\n🤖 OCR-Native: {response['response']}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        # End conversation
        if self.current_session:
            summary = self.conversational_interface.end_conversation(self.current_session)
            print(f"\n📊 Session Summary:")
            print(f"   Duration: {summary['duration']:.1f}s")
            print(f"   Messages: {summary['message_count']}")
            print(f"   Model: {summary['model_variant']}")
    
    def _run_benchmark(self, args):
        """Run benchmarks"""
        print("📊 Running OCR-Native LLM Benchmarks")
        print("=" * 50)
        
        benchmark = OCRNativeBenchmark(args.output_dir)
        
        if args.quick:
            print("Running quick benchmark...")
            suite = run_quick_benchmark()
        else:
            print("Running comprehensive benchmark...")
            suite = benchmark.run_comprehensive_benchmark()
        
        # Print summary
        print(f"\n📈 Benchmark Results:")
        print(f"   Total Tests: {suite.summary['total_tests']}")
        print(f"   Successful: {suite.summary['successful_tests']}")
        print(f"   Failed: {suite.summary['failed_tests']}")
        print(f"   Avg Throughput: {suite.summary['average_throughput']:.2f} ops/s")
        print(f"   Avg Duration: {suite.summary['average_duration']:.4f}s")
        print(f"   Avg Memory: {suite.summary['average_memory_usage']:.2f}%")
        
        if suite.summary.get('best_performing_model'):
            print(f"   Best Model: {suite.summary['best_performing_model']}")
        
        # Generate visualizations if requested
        if args.visualize:
            print("\n📊 Generating visualizations...")
            benchmark.generate_visualizations(suite)
            print(f"   Visualizations saved to {args.output_dir}")
        
        print(f"\n📁 Results saved to {args.output_dir}")
    
    def _run_compare(self, args):
        """Compare model variants"""
        print(f"🔍 Comparing Model Variants ({args.size} size)")
        print("=" * 50)
        
        comparison = self._run_quick_comparison(args.size)
        
        if not comparison:
            print("❌ No comparison data available")
            return
        
        print("\n📊 Performance Comparison:")
        print("-" * 30)
        
        # Sort by performance
        sorted_models = sorted(comparison.items(), key=lambda x: x[1], reverse=True)
        
        for i, (model, throughput) in enumerate(sorted_models, 1):
            size, variant = model.split('_', 1)
            print(f"{i:2d}. {variant:15s} ({size:5s}): {throughput:6.2f} ops/s")
        
        print(f"\n🏆 Best performing: {sorted_models[0][0]}")
    
    def _run_test(self, args):
        """Test specific model"""
        print(f"🧪 Testing {args.size} model ({args.variant} variant)")
        print("=" * 50)
        
        # Initialize model
        interface = create_conversational_interface(args.size, args.variant)
        session_id = interface.start_conversation("test_user")
        
        # Test with provided input or default
        test_input = args.input or "Hello, test the OCR-native model!"
        
        print(f"Input: {test_input}")
        print("Processing...")
        
        # Send message
        response = interface.send_message(session_id, test_input)
        
        print(f"\nResponse: {response['response']}")
        print(f"\nMetadata: {json.dumps(response['metadata'], indent=2)}")
        
        # End session
        interface.end_conversation(session_id)
    
    def _run_info(self, args):
        """Show system information"""
        print("ℹ️  OCR-Native LLM System Information")
        print("=" * 50)
        
        # System info
        import torch
        import psutil
        
        print(f"Python Version: {sys.version}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CPU Count: {psutil.cpu_count()}")
        print(f"Memory Total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        import platform
        print(f"Platform: {platform.platform()}")
        
        # Available variants
        print(f"\nAvailable Model Variants:")
        variants = ["standard", "mamba", "hybrid", "memory_enhanced"]
        for variant in variants:
            print(f"  - {variant}")
        
        # Available sizes
        print(f"\nAvailable Model Sizes:")
        sizes = ["small", "default", "large"]
        for size in sizes:
            print(f"  - {size}")
    
    def _print_chat_help(self):
        """Print chat help"""
        print("\n💬 Chat Commands:")
        print("  help      - Show this help")
        print("  info      - Show session information")
        print("  clear     - Clear conversation history")
        print("  variants  - Show available model variants")
        print("  switch X  - Switch to variant X")
        print("  quit      - Exit chat")
    
    def _print_session_info(self):
        """Print session information"""
        if not self.current_session:
            print("No active session")
            return
        
        info = self.conversational_interface.get_session_info(self.current_session)
        print(f"\n📊 Session Information:")
        print(f"   Session ID: {info['session_id']}")
        print(f"   User ID: {info['user_id']}")
        print(f"   Messages: {len(info['messages'])}")
        print(f"   Model: {info['model_config']['variant']}")
        print(f"   Started: {info['start_time']}")
    
    def _clear_conversation(self):
        """Clear conversation history"""
        if self.current_session:
            # End current session and start new one
            self.conversational_interface.end_conversation(self.current_session)
            self.current_session = self.conversational_interface.start_conversation("cli_user")
            print("Conversation cleared! Starting fresh session.")
    
    def _print_available_variants(self):
        """Print available variants"""
        variants = self.conversational_interface.get_available_variants()
        print(f"\n🔧 Available Model Variants:")
        for variant in variants:
            print(f"  - {variant}")
        print("Use 'switch X' to change variant")
    
    def _switch_variant(self, variant):
        """Switch model variant"""
        if self.conversational_interface.switch_model_variant(variant):
            print(f"✅ Switched to {variant} variant")
        else:
            print(f"❌ Failed to switch to {variant} variant")
    
    def _run_quick_comparison(self, size: str) -> Dict[str, float]:
        """Run quick comparison of variants"""
        from src.benchmarks.performance_benchmark import compare_variants
        
        # Override model sizes for comparison
        benchmark = OCRNativeBenchmark()
        benchmark.model_sizes = [size]
        benchmark.variants = ["standard", "mamba", "hybrid", "memory_enhanced"]
        
        suite = benchmark.run_comprehensive_benchmark()
        
        # Extract comparison data
        comparison = {}
        for result in suite.results:
            if result.error is None and result.test_name.startswith("performance"):
                key = f"{result.model_size}_{result.model_variant}"
                if key not in comparison:
                    comparison[key] = []
                comparison[key].append(result.throughput)
        
        # Calculate averages
        for key in comparison:
            comparison[key] = sum(comparison[key]) / len(comparison[key])
        
        return comparison


def main():
    """Main CLI entry point"""
    cli = OCRNativeCLI()
    cli.run()


if __name__ == "__main__":
    main()