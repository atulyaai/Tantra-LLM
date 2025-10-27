#!/usr/bin/env python3
"""
Tantra Quick Start Script
Demonstrates the complete conversational speech training system
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.training_config import TrainingConfig
from src.training.conversational_trainer import ConversationalTrainer
from src.training.speech_trainer import SpeechTrainer
from src.training.github_integration import GitHubModelManager
from src.core.tantra_llm import TantraLLM, TantraConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_conversational_ai():
    """Demonstrate conversational AI capabilities"""
    print("\n" + "="*60)
    print("🔤 TANTRA CONVERSATIONAL AI DEMO")
    print("="*60)
    
    # Create configuration
    config = TrainingConfig()
    
    # Initialize model
    tantra_config = TantraConfig()
    model = TantraLLM(tantra_config)
    
    # Initialize conversational trainer
    trainer = ConversationalTrainer(config, model)
    
    # Demo conversations
    demo_conversations = [
        {
            "message": "Hello, how are you today?",
            "context": "greeting",
            "personality": "helpful"
        },
        {
            "message": "Can you help me with a technical problem?",
            "context": "technical_support", 
            "personality": "knowledgeable"
        },
        {
            "message": "Tell me a creative story about a robot",
            "context": "creative_writing",
            "personality": "creative"
        }
    ]
    
    for i, conv in enumerate(demo_conversations, 1):
        print(f"\n--- Conversation {i} ---")
        print(f"User: {conv['message']}")
        print(f"Context: {conv['context']}, Personality: {conv['personality']}")
        
        try:
            response = trainer.generate_response(
                conv['message'],
                conv['context'],
                conv['personality']
            )
            print(f"Tantra: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✅ Conversational AI demo completed!")


def demonstrate_speech_processing():
    """Demonstrate speech processing capabilities"""
    print("\n" + "="*60)
    print("🎤 TANTRA SPEECH PROCESSING DEMO")
    print("="*60)
    
    try:
        import numpy as np
        import soundfile as sf
        
        # Create configuration
        config = TrainingConfig()
        
        # Initialize model
        tantra_config = TantraConfig()
        model = TantraLLM(tantra_config)
        
        # Initialize speech trainer
        speech_trainer = SpeechTrainer(config, model)
        
        # Demo text-to-speech
        demo_texts = [
            "Hello, this is Tantra speaking!",
            "I can convert text to speech.",
            "Welcome to the future of AI!"
        ]
        
        print("\n--- Text-to-Speech Demo ---")
        for i, text in enumerate(demo_texts, 1):
            print(f"Text {i}: {text}")
            
            try:
                # Generate speech
                audio = speech_trainer.text_to_speech(text, voice_style="neutral")
                
                # Save audio file
                output_file = f"demo_speech_{i}.wav"
                sf.write(output_file, audio, 16000)
                print(f"✅ Speech saved to: {output_file}")
                
            except Exception as e:
                print(f"❌ Error generating speech: {e}")
        
        # Demo speech-to-text (with synthetic audio)
        print("\n--- Speech-to-Text Demo ---")
        try:
            # Create synthetic audio for demo
            duration = 2.0  # seconds
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration))
            frequency = 440  # A4 note
            demo_audio = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # Add some variation
            demo_audio += np.random.normal(0, 0.05, len(demo_audio))
            
            # Convert speech to text
            text = speech_trainer.speech_to_text(demo_audio)
            print(f"Transcribed audio: {text}")
            
        except Exception as e:
            print(f"❌ Error in speech-to-text: {e}")
        
        print("\n✅ Speech processing demo completed!")
        
    except ImportError:
        print("❌ Audio processing libraries not available.")
        print("Install with: pip install librosa soundfile")


def demonstrate_github_integration():
    """Demonstrate GitHub integration capabilities"""
    print("\n" + "="*60)
    print("🐙 TANTRA GITHUB INTEGRATION DEMO")
    print("="*60)
    
    # Check if GitHub token is available
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("⚠️  GitHub token not found. Set GITHUB_TOKEN environment variable.")
        print("   Example: export GITHUB_TOKEN='your_github_token'")
        return
    
    try:
        # Initialize GitHub manager
        github_manager = GitHubModelManager(github_token, "tantra-ai/tantra-models")
        
        if github_manager.is_available():
            print("✅ Connected to GitHub repository")
            
            # Get repository info
            repo_info = github_manager.get_repository_info()
            print(f"Repository: {repo_info.get('full_name', 'Unknown')}")
            print(f"Description: {repo_info.get('description', 'No description')}")
            print(f"Stars: {repo_info.get('stars', 0)}")
            
            # List releases
            releases = github_manager.list_releases()
            print(f"\nFound {len(releases)} releases:")
            for release in releases[:3]:  # Show first 3
                print(f"  - {release['tag_name']}: {release['name']}")
            
            print("\n✅ GitHub integration demo completed!")
            
        else:
            print("❌ Failed to connect to GitHub repository")
            
    except Exception as e:
        print(f"❌ GitHub integration error: {e}")


def demonstrate_training_pipeline():
    """Demonstrate training pipeline capabilities"""
    print("\n" + "="*60)
    print("🏋️ TANTRA TRAINING PIPELINE DEMO")
    print("="*60)
    
    # Create configuration
    config = TrainingConfig()
    
    print("Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Device: {config.device}")
    
    # Initialize model
    tantra_config = TantraConfig()
    model = TantraLLM(tantra_config)
    
    print(f"\nModel Info:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Initialize trainers
    conv_trainer = ConversationalTrainer(config, model)
    speech_trainer = SpeechTrainer(config, model)
    
    print("\n✅ Training pipeline initialized successfully!")
    print("   To start training, run: python run_training.py")


def main():
    """Main demonstration function"""
    print("🚀 TANTRA CONVERSATIONAL SPEECH SYSTEM DEMO")
    print("=" * 60)
    
    try:
        # Demonstrate conversational AI
        demonstrate_conversational_ai()
        
        # Demonstrate speech processing
        demonstrate_speech_processing()
        
        # Demonstrate GitHub integration
        demonstrate_github_integration()
        
        # Demonstrate training pipeline
        demonstrate_training_pipeline()
        
        print("\n" + "="*60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run interactive demo: python demo_conversational_speech.py")
        print("2. Start training: python run_training.py")
        print("3. Read documentation: TRAINING_GUIDE.md")
        print("4. Check examples: examples/")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())