#!/usr/bin/env python3
"""
Train Large Tantra Model
Comprehensive training script for large model with fine-tuning
"""

import sys
import os
import argparse
import logging
import json
import time
import torch
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from large_model_config import create_large_tantra_config, create_large_training_config
from src.core.tantra_llm import TantraLLM
from src.training.conversational_trainer import ConversationalTrainer
from src.training.speech_trainer import SpeechTrainer
from src.training.lora_trainer import LoRATrainer, LoRAConfig
from src.utils.error_handler import logger

class LargeModelTrainer:
    """Trainer for large Tantra model"""
    
    def __init__(self):
        self.tantra_config = create_large_tantra_config()
        self.training_config = create_large_training_config()
        self.device = torch.device(self.training_config.device)
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training components
        self.conv_trainer = None
        self.speech_trainer = None
        self.lora_trainer = None
        
        logger.info("LargeModelTrainer initialized")
    
    def create_large_model(self):
        """Create the large Tantra model"""
        logger.info("Creating large Tantra model...")
        
        try:
            # Create model
            self.model = TantraLLM(self.tantra_config).to(self.device)
            
            # Print model info
            model_info = self.model.get_model_info()
            logger.info(f"Large model created:")
            logger.info(f"  • Parameters: {model_info['total_parameters']:,}")
            logger.info(f"  • Size: {model_info['model_size_mb']:.2f} MB")
            logger.info(f"  • Layers: {model_info['n_layers']}")
            logger.info(f"  • Dimensions: {model_info['d_model']}")
            
            # Setup optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=0.01
            )
            
            # Setup scheduler
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.num_epochs,
                eta_min=self.training_config.learning_rate * 0.1
            )
            
            logger.info("Large model setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create large model: {e}")
            return False
    
    def setup_training_components(self):
        """Setup training components"""
        logger.info("Setting up training components...")
        
        try:
            # Conversational trainer
            self.conv_trainer = ConversationalTrainer(self.training_config, self.model)
            
            # Speech trainer
            self.speech_trainer = SpeechTrainer(self.training_config, self.model)
            
            # LoRA trainer for fine-tuning
            lora_config = LoRAConfig(
                rank=32,                    # Larger rank for large model
                alpha=64.0,                 # Larger alpha
                learning_rate=1e-4,         # LoRA learning rate
                lora_learning_rate=2e-3     # Higher LoRA learning rate
            )
            self.lora_trainer = LoRATrainer(self.model, lora_config)
            
            logger.info("Training components setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup training components: {e}")
            return False
    
    def create_training_data(self):
        """Create comprehensive training data"""
        logger.info("Creating training data...")
        
        # Create directories
        os.makedirs("data/raw/conversations/train", exist_ok=True)
        os.makedirs("data/raw/conversations/val", exist_ok=True)
        os.makedirs("data/raw/speech/train", exist_ok=True)
        os.makedirs("data/raw/speech/val", exist_ok=True)
        
        # Create conversation data
        conversation_data = {
            "general_chat": [
                {
                    "conversation_id": f"gen_{i:03d}",
                    "type": "general_chat",
                    "personality": "helpful",
                    "messages": [
                        {"role": "user", "content": f"Hello, how are you today? This is conversation {i}."},
                        {"role": "assistant", "content": f"Hello! I'm doing well, thank you for asking. I'm here to help you with any questions or tasks you might have. This is response {i} in our conversation. How can I assist you today?"}
                    ],
                    "context": f"General conversation {i}"
                }
                for i in range(100)  # 100 conversations
            ],
            "technical_support": [
                {
                    "conversation_id": f"tech_{i:03d}",
                    "type": "technical_support",
                    "personality": "knowledgeable",
                    "messages": [
                        {"role": "user", "content": f"I'm having a technical issue {i}. Can you help me troubleshoot?"},
                        {"role": "assistant", "content": f"I'd be happy to help you with your technical issue {i}. Let's start by identifying the problem and working through potential solutions step by step."}
                    ],
                    "context": f"Technical support conversation {i}"
                }
                for i in range(50)  # 50 technical conversations
            ],
            "creative_writing": [
                {
                    "conversation_id": f"creative_{i:03d}",
                    "type": "creative_writing",
                    "personality": "creative",
                    "messages": [
                        {"role": "user", "content": f"Help me write a creative story {i} about artificial intelligence."},
                        {"role": "assistant", "content": f"I'd love to help you create a creative story {i} about AI! Let me craft something imaginative and engaging for you."}
                    ],
                    "context": f"Creative writing conversation {i}"
                }
                for i in range(50)  # 50 creative conversations
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
                json.dump(conversations[:10], f, indent=2)
        
        # Create speech metadata
        speech_metadata = {
            "train": [
                {
                    "file": f"sample_{i:03d}.wav",
                    "text": f"This is sample speech data {i} for Tantra large model training.",
                    "duration": 2.0 + (i % 10) * 0.5
                }
                for i in range(200)  # 200 speech samples
            ],
            "val": [
                {
                    "file": f"val_{i:03d}.wav",
                    "text": f"This is validation speech data {i} for Tantra large model.",
                    "duration": 1.5 + (i % 5) * 0.3
                }
                for i in range(50)  # 50 validation samples
            ]
        }
        
        for split, samples in speech_metadata.items():
            metadata_path = f"data/raw/speech/{split}/metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(samples, f, indent=2)
        
        logger.info("Training data created successfully")
        return True
    
    def train_large_model(self):
        """Train the large model"""
        logger.info("Starting large model training...")
        
        start_time = time.time()
        
        try:
            # Training loop
            for epoch in range(self.training_config.num_epochs):
                logger.info(f"Starting epoch {epoch + 1}/{self.training_config.num_epochs}")
                
                # Train conversational
                if self.conv_trainer:
                    logger.info("Training conversational capabilities...")
                    self.conv_trainer.train_epoch(epoch)
                
                # Train speech
                if self.speech_trainer:
                    logger.info("Training speech capabilities...")
                    self.speech_trainer.train_epoch(epoch)
                
                # Train LoRA
                if self.lora_trainer:
                    logger.info("Training LoRA fine-tuning...")
                    self.lora_trainer.train_epoch(epoch)
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
                
                # Save checkpoint
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(epoch + 1)
                
                logger.info(f"Epoch {epoch + 1} completed")
            
            training_time = time.time() - start_time
            logger.info(f"Large model training completed in {training_time:.2f} seconds")
            
            # Save final model
            self.save_final_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint_path = f"Model/checkpoints/tantra_large_epoch_{epoch}.pt"
        os.makedirs("Model/checkpoints", exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'tantra_config': self.tantra_config,
            'training_config': self.training_config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self):
        """Save final trained model"""
        model_path = "Model/weights/Tantra_large_v1.0.pt"
        os.makedirs("Model/weights", exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), model_path)
        
        # Save configuration
        config_path = "Model/weights/Tantra_large_config.json"
        config_data = {
            'tantra_config': {
                'd_model': self.tantra_config.d_model,
                'n_layers': self.tantra_config.n_layers,
                'n_heads': self.tantra_config.n_heads,
                'd_ff': self.tantra_config.d_ff,
                'vocab_size': self.tantra_config.vocab_size,
                'max_seq_length': self.tantra_config.max_seq_length
            },
            'training_config': self.training_config.to_dict(),
            'model_info': self.model.get_model_info()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Final model saved: {model_path}")
        logger.info(f"Configuration saved: {config_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Large Tantra Model")
    parser.add_argument("--skip-data", action="store_true", help="Skip data creation")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('large_model_training.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🚀 Starting Large Tantra Model Training")
    logger.info("=" * 60)
    
    try:
        # Initialize trainer
        trainer = LargeModelTrainer()
        
        # Override config with command line arguments
        if args.epochs:
            trainer.training_config.num_epochs = args.epochs
        if args.batch_size:
            trainer.training_config.batch_size = args.batch_size
        if args.learning_rate:
            trainer.training_config.learning_rate = args.learning_rate
        
        # Create large model
        if not trainer.create_large_model():
            logger.error("Failed to create large model")
            return 1
        
        # Setup training components
        if not trainer.setup_training_components():
            logger.error("Failed to setup training components")
            return 1
        
        # Create training data
        if not args.skip_data:
            if not trainer.create_training_data():
                logger.error("Failed to create training data")
                return 1
        
        # Train model
        if not trainer.train_large_model():
            logger.error("Training failed")
            return 1
        
        logger.info("🎉 Large model training completed successfully!")
        logger.info("=" * 60)
        logger.info("Model saved to: Model/weights/Tantra_large_v1.0.pt")
        logger.info("Checkpoints saved to: Model/checkpoints/")
        logger.info("Logs saved to: large_model_training.log")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())