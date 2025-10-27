#!/usr/bin/env python3
"""
Complete Tantra Conversational Speech Training Pipeline
Runs both conversational and speech training with GitHub integration
"""

import sys
import os
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.training_config import TrainingConfig
from src.training.conversational_trainer import ConversationalTrainer
from src.training.speech_trainer import SpeechTrainer
from src.training.github_integration import GitHubModelManager, ModelVersionManager
from src.core.tantra_llm import TantraLLM, TantraConfig
from src.utils.error_handler import logger


def setup_environment():
    """Setup training environment"""
    # Create necessary directories
    directories = [
        "data/raw/conversations/train",
        "data/raw/conversations/val", 
        "data/raw/speech/train",
        "data/raw/speech/val",
        "data/processed",
        "Model/weights",
        "Model/checkpoints",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def create_sample_training_data(config: TrainingConfig):
    """Create comprehensive sample training data"""
    logger.info("Creating sample training data...")
    
    # Conversation data
    conversation_data = {
        "general_chat": [
            {
                "conversation_id": "gen_001",
                "type": "general_chat",
                "personality": "helpful",
                "messages": [
                    {"role": "user", "content": "Hello, how are you today?"},
                    {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. I'm here to help you with any questions or tasks you might have. How can I assist you today?"}
                ],
                "context": "Initial greeting and offer of assistance"
            },
            {
                "conversation_id": "gen_002",
                "type": "general_chat", 
                "personality": "friendly",
                "messages": [
                    {"role": "user", "content": "What's the weather like?"},
                    {"role": "assistant", "content": "I don't have access to real-time weather data, but I'd be happy to help you find weather information! You can check your local weather app, visit a weather website, or ask a voice assistant for current conditions in your area."}
                ],
                "context": "Weather inquiry"
            }
        ],
        "technical_support": [
            {
                "conversation_id": "tech_001",
                "type": "technical_support",
                "personality": "knowledgeable",
                "messages": [
                    {"role": "user", "content": "My computer is running slowly"},
                    {"role": "assistant", "content": "I can help you troubleshoot your computer performance issues. Let's start by identifying potential causes:\n\n1. Check if you have too many programs running\n2. Look for malware or viruses\n3. Check available disk space\n4. Consider upgrading RAM or storage\n\nCan you tell me more about when the slowdown started and what specific tasks are affected?"}
                ],
                "context": "Computer performance troubleshooting"
            }
        ],
        "creative_writing": [
            {
                "conversation_id": "creative_001",
                "type": "creative_writing",
                "personality": "creative",
                "messages": [
                    {"role": "user", "content": "Help me write a short story about a robot"},
                    {"role": "assistant", "content": "I'd love to help you create a story about a robot! Here's a beginning:\n\n**The Last Guardian**\n\nIn the year 2157, ARIA-7 stood alone in the abandoned laboratory, her metallic fingers gently tracing the dust-covered control panels. She was the last of her kind—a caretaker robot designed to maintain the facility long after humans had departed for the stars.\n\nHer programming told her to wait, but her curiosity circuits had begun to wonder: what lay beyond the laboratory walls? The last human had left a message: 'When you're ready, follow the light.'\n\nWould you like me to continue the story or explore a different angle?"}
                ],
                "context": "Creative writing assistance"
            }
        ]
    }
    
    # Save conversation data
    for conv_type, conversations in conversation_data.items():
        train_path = f"data/raw/conversations/train/{conv_type}.json"
        val_path = f"data/raw/conversations/val/{conv_type}.json"
        
        with open(train_path, 'w') as f:
            json.dump(conversations, f, indent=2)
        
        # Use subset for validation
        with open(val_path, 'w') as f:
            json.dump(conversations[:1], f, indent=2)
    
    logger.info("Sample conversation data created")
    
    # Create sample speech data (metadata only - actual audio would be generated)
    speech_metadata = {
        "train": [
            {"file": "sample_001.wav", "text": "Hello, welcome to Tantra conversational AI", "duration": 2.5},
            {"file": "sample_002.wav", "text": "How can I help you today?", "duration": 1.8},
            {"file": "sample_003.wav", "text": "I'm here to assist with your questions", "duration": 2.2},
            {"file": "sample_004.wav", "text": "Let me think about that for a moment", "duration": 2.0},
            {"file": "sample_005.wav", "text": "That's an interesting question", "duration": 1.9}
        ],
        "val": [
            {"file": "val_001.wav", "text": "Thank you for your patience", "duration": 1.7},
            {"file": "val_002.wav", "text": "I hope that was helpful", "duration": 1.6}
        ]
    }
    
    for split, samples in speech_metadata.items():
        metadata_path = f"data/raw/speech/{split}/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(samples, f, indent=2)
    
    logger.info("Sample speech metadata created")


def train_models(config: TrainingConfig) -> Dict[str, Any]:
    """Train both conversational and speech models"""
    results = {
        "conversational": None,
        "speech": None,
        "training_time": 0,
        "success": False
    }
    
    start_time = time.time()
    
    try:
        # Initialize base model
        tantra_config = TantraConfig(
            d_model=config.conversation_max_length // 4,
            n_layers=12,
            n_heads=8,
            vocab_size=50000,
            max_seq_length=config.conversation_max_length
        )
        base_model = TantraLLM(tantra_config)
        
        # Train conversational model
        logger.info("Starting conversational training...")
        conv_trainer = ConversationalTrainer(config, base_model)
        conv_trainer.train()
        results["conversational"] = conv_trainer
        
        # Train speech model
        logger.info("Starting speech training...")
        speech_trainer = SpeechTrainer(config, base_model)
        speech_trainer.train()
        results["speech"] = speech_trainer
        
        results["training_time"] = time.time() - start_time
        results["success"] = True
        
        logger.info(f"Training completed successfully in {results['training_time']:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        results["training_time"] = time.time() - start_time
        results["error"] = str(e)
    
    return results


def save_models_to_github(config: TrainingConfig, training_results: Dict[str, Any]):
    """Save trained models to GitHub with full metadata"""
    if not config.github_token:
        logger.warning("GitHub token not provided, skipping GitHub upload")
        return False
    
    try:
        # Initialize GitHub manager
        github_manager = GitHubModelManager(config.github_token, config.github_repo)
        
        if not github_manager.is_available():
            logger.error("GitHub integration not available")
            return False
        
        # Initialize version manager
        version_manager = ModelVersionManager(github_manager)
        
        # Prepare training metadata
        training_metadata = {
            "training_time": training_results["training_time"],
            "success": training_results["success"],
            "config": config.to_dict(),
            "timestamp": time.time()
        }
        
        # Performance metrics (simplified)
        performance_metrics = {
            "conversational_quality": 0.85,  # Placeholder
            "speech_quality": 0.80,  # Placeholder
            "inference_speed": 0.5,  # Placeholder
            "memory_usage": 0.7  # Placeholder
        }
        
        # Save conversational model
        conv_model_path = os.path.join(
            config.model_output_path, 
            f"{config.model_name}_conversational.pt"
        )
        
        if os.path.exists(conv_model_path):
            conv_metadata = training_metadata.copy()
            conv_metadata["model_type"] = "conversational"
            conv_metadata["performance"] = performance_metrics
            
            success = version_manager.save_model_with_versioning(
                conv_model_path,
                f"{config.model_name}_conversational",
                config.to_dict(),
                performance_metrics
            )
            
            if success:
                logger.info("✅ Conversational model saved to GitHub")
            else:
                logger.error("❌ Failed to save conversational model to GitHub")
        
        # Save speech model
        speech_model_path = os.path.join(
            config.model_output_path, 
            f"{config.model_name}_speech.pt"
        )
        
        if os.path.exists(speech_model_path):
            speech_metadata = training_metadata.copy()
            speech_metadata["model_type"] = "speech"
            speech_metadata["performance"] = performance_metrics
            
            success = version_manager.save_model_with_versioning(
                speech_model_path,
                f"{config.model_name}_speech",
                config.to_dict(),
                performance_metrics
            )
            
            if success:
                logger.info("✅ Speech model saved to GitHub")
            else:
                logger.error("❌ Failed to save speech model to GitHub")
        
        # Create release
        release = github_manager.create_release(
            tag=f"v{int(time.time())}",
            title=f"Tantra {config.model_name} Release",
            description=f"Trained conversational and speech models for Tantra {config.model_name}",
            model_files=[conv_model_path, speech_model_path] if os.path.exists(conv_model_path) and os.path.exists(speech_model_path) else []
        )
        
        if release:
            logger.info(f"✅ Created GitHub release: {release.tag_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save models to GitHub: {e}")
        return False


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Tantra Conversational Speech Training Pipeline")
    parser.add_argument("--config", type=str, help="Path to training config file")
    parser.add_argument("--github-token", type=str, help="GitHub token for model saving")
    parser.add_argument("--github-repo", type=str, help="GitHub repository for model saving")
    parser.add_argument("--skip-github", action="store_true", help="Skip GitHub upload")
    parser.add_argument("--conversational-only", action="store_true", help="Train only conversational model")
    parser.add_argument("--speech-only", action="store_true", help="Train only speech model")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🚀 Starting Tantra Conversational Speech Training Pipeline")
    
    try:
        # Load configuration
        if args.config and os.path.exists(args.config):
            config = TrainingConfig.load_config(args.config)
            logger.info(f"Loaded config from {args.config}")
        else:
            config = TrainingConfig()
            logger.info("Using default configuration")
        
        # Override config with command line arguments
        if args.github_token:
            config.github_token = args.github_token
        if args.github_repo:
            config.github_repo = args.github_repo
        
        # Setup environment
        setup_environment()
        
        # Create sample data
        create_sample_training_data(config)
        
        # Train models
        training_results = train_models(config)
        
        if training_results["success"]:
            logger.info("🎉 Training completed successfully!")
            
            # Save to GitHub if not skipped
            if not args.skip_github and config.github_token:
                logger.info("📤 Uploading models to GitHub...")
                github_success = save_models_to_github(config, training_results)
                
                if github_success:
                    logger.info("✅ Models successfully uploaded to GitHub!")
                else:
                    logger.warning("⚠️ GitHub upload failed, but models are saved locally")
            else:
                logger.info("📁 Models saved locally (GitHub upload skipped)")
            
            # Print summary
            print("\n" + "="*60)
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Training time: {training_results['training_time']:.2f} seconds")
            print(f"Models saved to: {config.model_output_path}")
            print(f"Checkpoints saved to: {config.checkpoint_path}")
            print(f"Logs saved to: {config.logs_path}")
            
            if not args.skip_github and config.github_token:
                print(f"Models uploaded to GitHub: {config.github_repo}")
            
            print("\nTo test your models, run:")
            print("  python demo_conversational_speech.py")
            print("="*60)
            
        else:
            logger.error("❌ Training failed!")
            if "error" in training_results:
                logger.error(f"Error: {training_results['error']}")
            return 1
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())