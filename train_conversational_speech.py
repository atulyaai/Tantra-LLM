#!/usr/bin/env python3
"""
Tantra Conversational Speech Training Script
Main training pipeline for conversational and speech capabilities
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.training_config import TrainingConfig
from src.training.conversational_trainer import ConversationalTrainer
from src.training.speech_trainer import SpeechTrainer
from src.training.data_loader import ConversationDataLoader, SpeechDataLoader
from src.core.tantra_llm import TantraLLM, TantraConfig
from src.utils.error_handler import logger

# GitHub integration
try:
    from github import Github
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    logger.warning("GitHub integration not available. Install PyGithub: pip install PyGithub")


class GitHubModelManager:
    """Manages model saving and versioning on GitHub"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.github = None
        self.repo = None
        
        if GITHUB_AVAILABLE and config.github_token:
            try:
                self.github = Github(config.github_token)
                self.repo = self.github.get_repo(config.github_repo)
                logger.info(f"Connected to GitHub repository: {config.github_repo}")
            except Exception as e:
                logger.error(f"Failed to connect to GitHub: {e}")
                self.github = None
                self.repo = None
        else:
            logger.warning("GitHub integration disabled")
    
    def save_model_to_github(self, model_path: str, model_name: str, 
                           commit_message: str = None) -> bool:
        """Save model to GitHub repository"""
        if not self.repo:
            logger.warning("GitHub repository not available")
            return False
        
        try:
            if commit_message is None:
                commit_message = self.config.commit_message
            
            # Read model file
            with open(model_path, 'rb') as f:
                model_content = f.read()
            
            # Create path in repository
            github_path = f"models/{model_name}"
            
            # Upload file
            try:
                # Try to get existing file
                contents = self.repo.get_contents(github_path)
                # Update existing file
                self.repo.update_file(
                    github_path,
                    commit_message,
                    model_content,
                    contents.sha
                )
                logger.info(f"Updated model {model_name} on GitHub")
            except:
                # Create new file
                self.repo.create_file(
                    github_path,
                    commit_message,
                    model_content
                )
                logger.info(f"Created new model {model_name} on GitHub")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model to GitHub: {e}")
            return False
    
    def create_release(self, tag: str, title: str, description: str) -> bool:
        """Create a GitHub release"""
        if not self.repo:
            logger.warning("GitHub repository not available")
            return False
        
        try:
            # Get the latest commit
            latest_commit = self.repo.get_commits()[0]
            
            # Create release
            release = self.repo.create_git_release(
                tag=tag,
                name=title,
                message=description,
                target_commitish=latest_commit.sha
            )
            
            logger.info(f"Created release {tag}: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create GitHub release: {e}")
            return False


def setup_logging(config: TrainingConfig):
    """Setup logging configuration"""
    log_level = logging.INFO
    
    # Create logs directory
    os.makedirs(config.logs_path, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.logs_path, 'training.log')),
            logging.StreamHandler()
        ]
    )


def create_sample_data(config: TrainingConfig):
    """Create sample training data if it doesn't exist"""
    logger.info("Creating sample training data...")
    
    # Create conversation data
    conv_train_path = os.path.join(config.conversation_data_path, "train")
    conv_val_path = os.path.join(config.conversation_data_path, "val")
    os.makedirs(conv_train_path, exist_ok=True)
    os.makedirs(conv_val_path, exist_ok=True)
    
    # Sample conversation data
    sample_conversations = [
        {
            "conversation_id": "sample_001",
            "type": "general_chat",
            "personality": "helpful",
            "messages": [
                {"role": "user", "content": "Hello, how are you today?"},
                {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. I'm here to help you with any questions or tasks you might have. How can I assist you today?"}
            ],
            "context": "Initial greeting and offer of assistance"
        },
        {
            "conversation_id": "sample_002",
            "type": "technical_support",
            "personality": "knowledgeable",
            "messages": [
                {"role": "user", "content": "I'm having trouble with my computer running slowly"},
                {"role": "assistant", "content": "I'd be happy to help you troubleshoot your computer performance issues. Let's start by identifying the potential causes. Can you tell me what specific tasks seem to be running slowly, and when did you first notice the slowdown?"}
            ],
            "context": "Technical support for computer performance"
        }
    ]
    
    # Save conversation data
    with open(os.path.join(conv_train_path, "conversations.json"), 'w') as f:
        json.dump(sample_conversations, f, indent=2)
    
    with open(os.path.join(conv_val_path, "conversations.json"), 'w') as f:
        json.dump(sample_conversations[:1], f, indent=2)
    
    # Create speech data directories
    speech_train_path = os.path.join(config.speech_data_path, "train")
    speech_val_path = os.path.join(config.speech_data_path, "val")
    os.makedirs(speech_train_path, exist_ok=True)
    os.makedirs(speech_val_path, exist_ok=True)
    
    logger.info("Sample data created successfully")


def train_conversational_model(config: TrainingConfig) -> ConversationalTrainer:
    """Train the conversational model"""
    logger.info("Starting conversational model training...")
    
    # Initialize trainer
    trainer = ConversationalTrainer(config)
    
    # Train the model
    trainer.train()
    
    logger.info("Conversational model training completed!")
    return trainer


def train_speech_model(config: TrainingConfig) -> SpeechTrainer:
    """Train the speech model"""
    logger.info("Starting speech model training...")
    
    # Initialize trainer
    trainer = SpeechTrainer(config)
    
    # Train the model
    trainer.train()
    
    logger.info("Speech model training completed!")
    return trainer


def save_models_to_github(config: TrainingConfig, github_manager: GitHubModelManager):
    """Save trained models to GitHub"""
    if not github_manager.repo:
        logger.warning("GitHub not available, skipping model upload")
        return
    
    logger.info("Saving models to GitHub...")
    
    # Save conversational model
    conv_model_path = os.path.join(config.model_output_path, f"{config.model_name}_conversational.pt")
    if os.path.exists(conv_model_path):
        success = github_manager.save_model_to_github(
            conv_model_path,
            f"{config.model_name}_conversational.pt",
            f"Update {config.model_name} conversational model"
        )
        if success:
            logger.info("Conversational model saved to GitHub")
        else:
            logger.error("Failed to save conversational model to GitHub")
    
    # Save speech model
    speech_model_path = os.path.join(config.model_output_path, f"{config.model_name}_speech.pt")
    if os.path.exists(speech_model_path):
        success = github_manager.save_model_to_github(
            speech_model_path,
            f"{config.model_name}_speech.pt",
            f"Update {config.model_name} speech model"
        )
        if success:
            logger.info("Speech model saved to GitHub")
        else:
            logger.error("Failed to save speech model to GitHub")
    
    # Create release
    github_manager.create_release(
        tag=f"v{config.model_name.split('_')[-1]}",
        title=f"Tantra {config.model_name} Release",
        description=f"Trained conversational and speech models for Tantra {config.model_name}"
    )


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Tantra Conversational Speech Model")
    parser.add_argument("--config", type=str, help="Path to training config file")
    parser.add_argument("--conversational-only", action="store_true", 
                       help="Train only conversational model")
    parser.add_argument("--speech-only", action="store_true", 
                       help="Train only speech model")
    parser.add_argument("--github-token", type=str, help="GitHub token for model saving")
    parser.add_argument("--github-repo", type=str, help="GitHub repository for model saving")
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        config = TrainingConfig.load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = TrainingConfig()
        logger.info("Using default config")
    
    # Override config with command line arguments
    if args.github_token:
        config.github_token = args.github_token
    if args.github_repo:
        config.github_repo = args.github_repo
    
    # Setup logging
    setup_logging(config)
    
    # Create sample data if needed
    create_sample_data(config)
    
    # Initialize GitHub manager
    github_manager = GitHubModelManager(config)
    
    try:
        # Train conversational model
        if not args.speech_only:
            conversational_trainer = train_conversational_model(config)
            logger.info("Conversational training completed successfully!")
        
        # Train speech model
        if not args.conversational_only:
            speech_trainer = train_speech_model(config)
            logger.info("Speech training completed successfully!")
        
        # Save models to GitHub
        if config.auto_commit:
            save_models_to_github(config, github_manager)
        
        logger.info("All training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()