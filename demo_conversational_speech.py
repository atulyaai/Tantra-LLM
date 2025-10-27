#!/usr/bin/env python3
"""
Tantra Conversational Speech Demo
Interactive demo showcasing conversational and speech capabilities
"""

import sys
import os
import argparse
import logging
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.tantra_llm import TantraLLM, TantraConfig
from src.training.training_config import TrainingConfig
from src.training.conversational_trainer import ConversationalTrainer
from src.training.speech_trainer import SpeechTrainer
from src.training.github_integration import GitHubModelManager, ModelVersionManager
from src.utils.error_handler import logger

# Audio processing
try:
    import soundfile as sf
    import librosa
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logger.warning("Audio processing not available. Install librosa and soundfile for speech features.")


class TantraConversationalSpeechDemo:
    """Interactive demo for Tantra conversational speech capabilities"""
    
    def __init__(self, config_path: str = None, model_path: str = None):
        self.config = self._load_config(config_path)
        self.model = None
        self.conversational_trainer = None
        self.speech_trainer = None
        self.github_manager = None
        
        # Load or initialize model
        self._load_model(model_path)
        
        # Initialize GitHub integration
        if self.config.github_token:
            self.github_manager = GitHubModelManager(
                self.config.github_token, 
                self.config.github_repo
            )
    
    def _load_config(self, config_path: str = None) -> TrainingConfig:
        """Load training configuration"""
        if config_path and os.path.exists(config_path):
            return TrainingConfig.load_config(config_path)
        else:
            return TrainingConfig()
    
    def _load_model(self, model_path: str = None):
        """Load or initialize the Tantra model"""
        try:
            if model_path and os.path.exists(model_path):
                # Load pre-trained model
                logger.info(f"Loading model from {model_path}")
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Initialize model with config
                tantra_config = TantraConfig()
                self.model = TantraLLM(tantra_config)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                logger.info("Model loaded successfully")
            else:
                # Initialize new model
                logger.info("Initializing new model")
                tantra_config = TantraConfig()
                self.model = TantraLLM(tantra_config)
                
                # Initialize trainers
                self.conversational_trainer = ConversationalTrainer(self.config, self.model)
                self.speech_trainer = SpeechTrainer(self.config, self.model)
                
                logger.info("New model initialized")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to new model
            tantra_config = TantraConfig()
            self.model = TantraLLM(tantra_config)
            self.conversational_trainer = ConversationalTrainer(self.config, self.model)
            self.speech_trainer = SpeechTrainer(self.config, self.model)
    
    def start_conversation(self):
        """Start interactive conversation"""
        print("\n" + "="*60)
        print("🔤 TANTRA CONVERSATIONAL SPEECH DEMO")
        print("="*60)
        print("Welcome to Tantra! I'm your OCR-native conversational AI.")
        print("I can help with text conversations and speech processing.")
        print("\nCommands:")
        print("  'text <message>' - Send a text message")
        print("  'speech <file.wav>' - Process speech file")
        print("  'train' - Start training mode")
        print("  'save' - Save model to GitHub")
        print("  'quit' - Exit demo")
        print("="*60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! Thanks for using Tantra.")
                    break
                
                elif user_input.lower() == 'train':
                    self._start_training_mode()
                
                elif user_input.lower() == 'save':
                    self._save_model_to_github()
                
                elif user_input.startswith('text '):
                    message = user_input[5:].strip()
                    self._handle_text_message(message)
                
                elif user_input.startswith('speech '):
                    file_path = user_input[7:].strip()
                    self._handle_speech_file(file_path)
                
                else:
                    # Default to text message
                    self._handle_text_message(user_input)
                    
            except KeyboardInterrupt:
                print("\nGoodbye! Thanks for using Tantra.")
                break
            except Exception as e:
                logger.error(f"Error in conversation: {e}")
                print(f"Sorry, I encountered an error: {e}")
    
    def _handle_text_message(self, message: str):
        """Handle text message input"""
        print(f"\nProcessing text: '{message}'")
        
        try:
            # Generate response using conversational trainer
            if self.conversational_trainer:
                response = self.conversational_trainer.generate_response(
                    message, 
                    context="conversation",
                    personality="helpful"
                )
            else:
                # Fallback to basic model response
                inputs = {
                    'text': message,
                    'speech': None,
                    'image': None
                }
                response = self.model.generate_response(inputs, message)
            
            print(f"\nTantra: {response}")
            
        except Exception as e:
            logger.error(f"Error processing text message: {e}")
            print(f"Sorry, I couldn't process that message: {e}")
    
    def _handle_speech_file(self, file_path: str):
        """Handle speech file input"""
        if not AUDIO_AVAILABLE:
            print("Speech processing not available. Please install librosa and soundfile.")
            return
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        print(f"\nProcessing speech file: {file_path}")
        
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=16000)
            
            # Speech to text
            if self.speech_trainer:
                text = self.speech_trainer.speech_to_text(audio)
            else:
                text = "Speech processing not available in this model."
            
            print(f"\nTranscribed speech: {text}")
            
            # Generate response
            if text and text != "Speech processing not available in this model.":
                self._handle_text_message(text)
            
            # Text to speech (if available)
            if self.speech_trainer and text:
                print("\nGenerating speech response...")
                response_audio = self.speech_trainer.text_to_speech(
                    text, 
                    voice_style="neutral"
                )
                
                # Save response audio
                output_path = f"response_{int(time.time())}.wav"
                sf.write(output_path, response_audio, 16000)
                print(f"Speech response saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing speech file: {e}")
            print(f"Sorry, I couldn't process that speech file: {e}")
    
    def _start_training_mode(self):
        """Start training mode"""
        print("\n" + "="*40)
        print("TRAINING MODE")
        print("="*40)
        print("Starting model training...")
        
        try:
            # Train conversational model
            if self.conversational_trainer:
                print("Training conversational capabilities...")
                self.conversational_trainer.train()
                print("✅ Conversational training completed!")
            
            # Train speech model
            if self.speech_trainer:
                print("Training speech capabilities...")
                self.speech_trainer.train()
                print("✅ Speech training completed!")
            
            print("🎉 All training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            print(f"❌ Training failed: {e}")
    
    def _save_model_to_github(self):
        """Save model to GitHub"""
        if not self.github_manager:
            print("GitHub integration not available.")
            return
        
        print("\nSaving model to GitHub...")
        
        try:
            # Save conversational model
            conv_model_path = os.path.join(
                self.config.model_output_path, 
                f"{self.config.model_name}_conversational.pt"
            )
            
            if os.path.exists(conv_model_path):
                success = self.github_manager.save_model_file(
                    conv_model_path,
                    f"models/{self.config.model_name}_conversational.pt",
                    f"Update {self.config.model_name} conversational model"
                )
                if success:
                    print("✅ Conversational model saved to GitHub")
                else:
                    print("❌ Failed to save conversational model")
            
            # Save speech model
            speech_model_path = os.path.join(
                self.config.model_output_path, 
                f"{self.config.model_name}_speech.pt"
            )
            
            if os.path.exists(speech_model_path):
                success = self.github_manager.save_model_file(
                    speech_model_path,
                    f"models/{self.config.model_name}_speech.pt",
                    f"Update {self.config.model_name} speech model"
                )
                if success:
                    print("✅ Speech model saved to GitHub")
                else:
                    print("❌ Failed to save speech model")
            
            print("🎉 Model saving completed!")
            
        except Exception as e:
            logger.error(f"Failed to save model to GitHub: {e}")
            print(f"❌ Failed to save model: {e}")
    
    def run_benchmark(self):
        """Run performance benchmark"""
        print("\n" + "="*40)
        print("PERFORMANCE BENCHMARK")
        print("="*40)
        
        # Test conversation quality
        if self.conversational_trainer:
            test_data = [
                {
                    'user_message': 'Hello, how are you?',
                    'context': 'greeting',
                    'personality': 'helpful'
                },
                {
                    'user_message': 'Can you help me with a technical problem?',
                    'context': 'technical_support',
                    'personality': 'knowledgeable'
                }
            ]
            
            print("Testing conversation quality...")
            metrics = self.conversational_trainer.evaluate_conversation_quality(test_data)
            
            print("Conversation Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
        
        # Test speech quality
        if self.speech_trainer and AUDIO_AVAILABLE:
            print("\nTesting speech quality...")
            
            # Generate test audio
            test_audio = np.random.randn(16000)  # 1 second of noise
            test_data = [{'audio': test_audio, 'text': 'test'}]
            
            metrics = self.speech_trainer.evaluate_speech_quality(test_data)
            
            print("Speech Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
        
        print("\n✅ Benchmark completed!")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Tantra Conversational Speech Demo")
    parser.add_argument("--config", type=str, help="Path to training config file")
    parser.add_argument("--model", type=str, help="Path to pre-trained model file")
    parser.add_argument("--github-token", type=str, help="GitHub token for model saving")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create demo instance
        demo = TantraConversationalSpeechDemo(
            config_path=args.config,
            model_path=args.model
        )
        
        # Set GitHub token if provided
        if args.github_token:
            demo.config.github_token = args.github_token
            demo.github_manager = GitHubModelManager(
                args.github_token, 
                demo.config.github_repo
            )
        
        # Run benchmark if requested
        if args.benchmark:
            demo.run_benchmark()
        else:
            # Start interactive conversation
            demo.start_conversation()
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    main()