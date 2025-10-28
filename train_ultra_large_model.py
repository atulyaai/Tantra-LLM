#!/usr/bin/env python3
"""
Train Tantra Ultra Large Model with Real Data
Uses the ultra-large configuration and real training data
"""

import os
import sys
import json
import torch
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append('src')

from core.tantra_llm import TantraLLM, TantraConfig
from training.training_config import TrainingConfig
from training.conversational_trainer import ConversationalTrainer
from training.data_loader import DataLoader
from utils.error_handler import ErrorHandler
from utils.validation_system import ValidationSystem

# Import ultra-large config
from ultra_large_model_config import create_ultra_large_tantra_config, create_ultra_large_training_config

class UltraLargeModelTrainer:
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.validator = ValidationSystem()
        self.model_config = create_ultra_large_tantra_config()
        self.training_config = create_ultra_large_training_config()
        
        # Create necessary directories
        self.setup_directories()
        
        # Initialize data loader
        self.data_loader = DataLoader(self.training_config)
        
    def setup_directories(self):
        """Create necessary directories for training"""
        directories = [
            "data/processed",
            "Model/weights",
            "checkpoints",
            "logs",
            "outputs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {directory}")
    
    def load_real_data(self) -> List[Dict]:
        """Load and validate real training data"""
        print("📊 Loading real training data...")
        
        data_files = [
            "data/raw/real_conversations.json",
            "data/raw/real_technical_qa.json", 
            "data/raw/real_creative_tasks.json"
        ]
        
        all_data = []
        
        for file_path in data_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_data.extend(data)
                        print(f"✅ Loaded {len(data)} examples from {file_path}")
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")
            else:
                print(f"⚠️  File not found: {file_path}")
        
        if not all_data:
            print("❌ No real data found! Please run prepare_real_data.py first.")
            return []
        
        # Validate data format
        validated_data = []
        for item in all_data:
            if self.validator.validate_training_example(item):
                validated_data.append(item)
            else:
                print(f"⚠️  Skipping invalid example: {item.get('id', 'unknown')}")
        
        print(f"📈 Total validated examples: {len(validated_data)}")
        return validated_data
    
    def create_model(self) -> TantraLLM:
        """Create the ultra-large Tantra model"""
        print("🏗️  Creating Ultra Large Tantra Model...")
        print(f"Model parameters:")
        print(f"  - d_model: {self.model_config.d_model}")
        print(f"  - n_layers: {self.model_config.n_layers}")
        print(f"  - n_heads: {self.model_config.n_heads}")
        print(f"  - d_ff: {self.model_config.d_ff}")
        print(f"  - vocab_size: {self.model_config.vocab_size}")
        print(f"  - max_seq_length: {self.model_config.max_seq_length}")
        
        try:
            model = TantraLLM(self.model_config)
            
            # Calculate model size
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"📊 Model Statistics:")
            print(f"  - Total parameters: {total_params:,}")
            print(f"  - Trainable parameters: {trainable_params:,}")
            print(f"  - Model size: {total_params * 4 / 1024**3:.2f} GB (FP32)")
            
            return model
            
        except Exception as e:
            self.error_handler.handle_error(f"Failed to create model: {e}")
            return None
    
    def setup_training(self, model: TantraLLM, training_data: List[Dict]):
        """Setup training environment and data"""
        print("⚙️  Setting up training environment...")
        
        # Update training config with data info
        self.training_config.training_data_path = "data/raw/real_combined_dataset.json"
        self.training_config.num_training_examples = len(training_data)
        
        # Create trainer
        trainer = ConversationalTrainer(
            model=model,
            config=self.training_config,
            data_loader=self.data_loader
        )
        
        print(f"🎯 Training Configuration:")
        print(f"  - Model: {self.training_config.model_name}")
        print(f"  - Learning rate: {self.training_config.learning_rate}")
        print(f"  - Batch size: {self.training_config.batch_size}")
        print(f"  - Epochs: {self.training_config.num_epochs}")
        print(f"  - Gradient accumulation: {self.training_config.gradient_accumulation_steps}")
        print(f"  - Device: {self.training_config.device}")
        print(f"  - Mixed precision: {self.training_config.mixed_precision}")
        print(f"  - Gradient checkpointing: {self.training_config.gradient_checkpointing}")
        
        return trainer
    
    def train_model(self, trainer: ConversationalTrainer, training_data: List[Dict]):
        """Train the ultra-large model"""
        print("🚀 Starting Ultra Large Model Training...")
        print("=" * 60)
        
        try:
            # Save training data
            with open("data/raw/real_combined_dataset.json", 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            # Start training
            start_time = time.time()
            
            trainer.train(training_data)
            
            end_time = time.time()
            training_time = end_time - start_time
            
            print("=" * 60)
            print("✅ Training completed successfully!")
            print(f"⏱️  Total training time: {training_time/3600:.2f} hours")
            print(f"💾 Model saved to: {self.training_config.output_dir}")
            
        except Exception as e:
            self.error_handler.handle_error(f"Training failed: {e}")
            print("❌ Training failed. Check logs for details.")
    
    def run_training(self):
        """Main training pipeline"""
        print("🎯 Tantra Ultra Large Model Training Pipeline")
        print("=" * 60)
        
        # Load real data
        training_data = self.load_real_data()
        if not training_data:
            print("❌ No training data available. Exiting.")
            return
        
        # Create model
        model = self.create_model()
        if model is None:
            print("❌ Failed to create model. Exiting.")
            return
        
        # Setup training
        trainer = self.setup_training(model, training_data)
        
        # Train model
        self.train_model(trainer, training_data)
        
        print("🎉 Ultra Large Model Training Complete!")

def main():
    """Main entry point"""
    print("🚀 Starting Tantra Ultra Large Model Training...")
    
    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"🔥 GPU available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️  No GPU available. Training will be slower on CPU.")
    
    # Create trainer and run training
    trainer = UltraLargeModelTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main()