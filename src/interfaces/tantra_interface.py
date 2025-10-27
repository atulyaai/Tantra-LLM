"""
Tantra v1.0 Interface
Clean, focused interface for Tantra OCR-Native LLM
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tantra_llm import TantraLLM, TantraConfig
from utils.error_handler import logger


class TantraInterface:
    """Clean interface for Tantra v1.0"""
    
    def __init__(self):
        """Initialize Tantra interface"""
        self.model = None
        self.config = None
        self._initialize_tantra()
    
    def _initialize_tantra(self):
        """Initialize Tantra model"""
        try:
            self.config = TantraConfig()
            self.model = TantraLLM(self.config)
            logger.info("Tantra v1.0 interface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tantra interface: {e}")
            raise
    
    def chat(self, message: str) -> str:
        """Simple chat with Tantra"""
        try:
            inputs = {'text': message, 'speech': None, 'image': None}
            response = self.model.generate_response(inputs, message)
            return response
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Tantra Error: {str(e)}"
    
    def process_multimodal(self, text: str = None, speech=None, image=None) -> str:
        """Process multi-modal input with Tantra"""
        try:
            inputs = {
                'text': text,
                'speech': speech,
                'image': image
            }
            response = self.model.generate_response(inputs, text or "Multi-modal input")
            return response
        except Exception as e:
            logger.error(f"Multi-modal processing error: {e}")
            return f"Tantra Error: {str(e)}"
    
    def get_model_info(self) -> dict:
        """Get Tantra model information"""
        if self.model:
            return self.model.get_model_info()
        return {"error": "Model not initialized"}
    
    def add_memory(self, content: str, memory_type: str = "conversation", importance: float = 1.0) -> str:
        """Add content to Tantra memory"""
        if self.model:
            return self.model.add_to_memory(content, memory_type, importance)
        return "Model not initialized"
    
    def get_memory(self) -> list:
        """Get Tantra memory"""
        if self.model:
            return self.model.get_conversation_history()
        return []
    
    def clear_memory(self):
        """Clear Tantra memory"""
        if self.model:
            self.model.clear_memory()
    
    def store_weights_as_ocr(self) -> list:
        """Store Tantra weights as OCR images"""
        if self.model:
            return self.model.store_weights_as_ocr()
        return []


def quick_chat(message: str) -> str:
    """Quick chat function with Tantra"""
    interface = TantraInterface()
    return interface.chat(message)


def main():
    """Main function for Tantra interface"""
    print("🔤 Tantra v1.0 Interface")
    print("=" * 40)
    
    # Initialize interface
    interface = TantraInterface()
    
    # Show model info
    info = interface.get_model_info()
    print(f"Model: {info['name']}")
    print(f"Parameters: {info['total_parameters']:,}")
    print(f"Size: {info['model_size_mb']:.2f} MB")
    print(f"Type: {info['model_type']}")
    
    # Interactive chat
    print("\n💬 Interactive Chat (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input:
                response = interface.chat(user_input)
                print(f"\nTantra: {response}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()