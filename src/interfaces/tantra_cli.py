"""
Command Line Interface for Tantra v1.0
Simple, focused CLI for Tantra OCR-Native LLM
"""

import argparse
import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path

from src.interfaces.tantra_interface import TantraInterface
from src.utils.error_handler import logger


class TantraCLI:
    """Command Line Interface for Tantra v1.0"""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.tantra_interface = None
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="Tantra v1.0 Command Line Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Start interactive chat
  python -m src.interfaces.tantra_cli chat
  
  # Test with specific input
  python -m src.interfaces.tantra_cli test --input "Hello, Tantra!"
  
  # Show model info
  python -m src.interfaces.tantra_cli info
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Chat command
        chat_parser = subparsers.add_parser('chat', help='Start interactive chat with Tantra')
        chat_parser.add_argument('--user-id', default='cli_user', help='User ID for session')
        
        # Test command
        test_parser = subparsers.add_parser('test', help='Test Tantra with specific input')
        test_parser.add_argument('--input', help='Input text to test')
        
        # Info command
        info_parser = subparsers.add_parser('info', help='Show Tantra model information')
        
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
        """Run interactive chat with Tantra"""
        print("🔤 Tantra v1.0 Interactive Chat")
        print("=" * 50)
        print("Type 'quit' to exit, 'help' for commands")
        print("-" * 50)
        
        # Initialize Tantra interface
        self.tantra_interface = TantraInterface()
        
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
                    self._print_model_info()
                    continue
                elif user_input.lower() == 'clear':
                    self._clear_memory()
                    continue
                elif not user_input:
                    continue
                
                # Send message and get response
                response = self.tantra_interface.chat(user_input)
                print(f"\n🔤 Tantra: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        print("\n👋 Goodbye!")
    
    def _run_test(self, args):
        """Test Tantra with specific input"""
        print("🧪 Testing Tantra v1.0")
        print("=" * 50)
        
        # Initialize Tantra interface
        self.tantra_interface = TantraInterface()
        
        # Test with provided input or default
        test_input = args.input or "Hello, test Tantra!"
        
        print(f"Input: {test_input}")
        print("Processing...")
        
        # Send message
        response = self.tantra_interface.chat(test_input)
        
        print(f"\nResponse: {response}")
    
    def _run_info(self, args):
        """Show Tantra model information"""
        print("ℹ️  Tantra v1.0 Model Information")
        print("=" * 50)
        
        # Initialize Tantra interface
        self.tantra_interface = TantraInterface()
        
        # Get model info
        info = self.tantra_interface.get_model_info()
        
        print(f"Model: {info['name']}")
        print(f"Parameters: {info['total_parameters']:,}")
        print(f"Size: {info['model_size_mb']:.2f} MB")
        print(f"Layers: {info['n_layers']}")
        print(f"Heads: {info['n_heads']}")
        print(f"Type: {info['model_type']}")
        print(f"Max Sequence Length: {info['max_seq_length']}")
        
        # System info
        import torch
        import psutil
        
        print(f"\nSystem Information:")
        print(f"Python Version: {sys.version}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CPU Count: {psutil.cpu_count()}")
        print(f"Memory Total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    def _print_chat_help(self):
        """Print chat help"""
        print("\n💬 Chat Commands:")
        print("  help      - Show this help")
        print("  info      - Show model information")
        print("  clear     - Clear Tantra memory")
        print("  quit      - Exit chat")
    
    def _print_model_info(self):
        """Print model information"""
        if not self.tantra_interface:
            print("Tantra interface not initialized")
            return
        
        info = self.tantra_interface.get_model_info()
        print(f"\n📊 Tantra Model Information:")
        print(f"   Model: {info['name']}")
        print(f"   Parameters: {info['total_parameters']:,}")
        print(f"   Size: {info['model_size_mb']:.2f} MB")
        print(f"   Type: {info['model_type']}")
    
    def _clear_memory(self):
        """Clear Tantra memory"""
        if self.tantra_interface:
            self.tantra_interface.clear_memory()
            print("Tantra memory cleared!")
        else:
            print("Tantra interface not initialized")


def main():
    """Main CLI entry point"""
    cli = TantraCLI()
    cli.run()


if __name__ == "__main__":
    main()