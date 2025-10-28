#!/usr/bin/env python3
"""
Execute Ultra Large Model Training with Real Data
Main training execution script that integrates all components
"""

import os
import sys
import json
import time
import torch
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.append('src')

from ultra_large_training_config import load_training_config, validate_training_setup, print_training_summary
from train_ultra_large_model import UltraLargeModelTrainer

def check_system_requirements():
    """Check system requirements for ultra-large model training"""
    print("🔍 Checking System Requirements...")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return "cuda"
    else:
        print("⚠️  CUDA not available - training will use CPU")
        print("⚠️  CPU training will be very slow for ultra-large model")
        return "cpu"
    
    print("=" * 50)

def check_data_availability():
    """Check if all required data is available"""
    print("📊 Checking Data Availability...")
    print("=" * 50)
    
    data_files = [
        "data/raw/real_conversations.json",
        "data/raw/real_technical_qa.json", 
        "data/raw/real_creative_tasks.json",
        "data/raw/real_combined_dataset.json"
    ]
    
    total_examples = 0
    for file_path in data_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✅ {file_path}: {len(data)} examples")
                total_examples += len(data)
        else:
            print(f"❌ {file_path}: Not found")
    
    print(f"📈 Total training examples: {total_examples}")
    
    if total_examples < 100:
        print("⚠️  Warning: Very few training examples available")
        return False
    
    print("=" * 50)
    return True

def estimate_training_time(device: str, num_examples: int):
    """Estimate training time based on system capabilities"""
    print("⏱️  Estimating Training Time...")
    print("=" * 50)
    
    # Rough estimates based on model size and data
    if device == "cuda":
        # GPU training estimates
        time_per_epoch = num_examples * 0.1  # 0.1 seconds per example on GPU
        total_time = time_per_epoch * 30  # 30 epochs
        print(f"🚀 GPU Training Estimate:")
        print(f"   - Time per epoch: {time_per_epoch/60:.1f} minutes")
        print(f"   - Total training time: {total_time/3600:.1f} hours")
    else:
        # CPU training estimates
        time_per_epoch = num_examples * 10  # 10 seconds per example on CPU
        total_time = time_per_epoch * 30  # 30 epochs
        print(f"🐌 CPU Training Estimate:")
        print(f"   - Time per epoch: {time_per_epoch/60:.1f} minutes")
        print(f"   - Total training time: {total_time/3600:.1f} hours")
        print("⚠️  CPU training will be very slow!")
    
    print("=" * 50)

def create_training_plan():
    """Create a detailed training plan"""
    print("📋 Creating Training Plan...")
    print("=" * 50)
    
    plan = {
        "phase_1": {
            "name": "Warmup Phase",
            "epochs": 2,
            "description": "Gradual learning rate increase to prevent instability",
            "learning_rate": 1e-6,
            "batch_size": 1
        },
        "phase_2": {
            "name": "Main Training Phase", 
            "epochs": 25,
            "description": "Full training with real data",
            "learning_rate": 2e-5,
            "batch_size": 1
        },
        "phase_3": {
            "name": "Fine-tuning Phase",
            "epochs": 3,
            "description": "Final optimization with reduced learning rate",
            "learning_rate": 5e-6,
            "batch_size": 1
        }
    }
    
    print("🎯 Training Phases:")
    for phase, config in plan.items():
        print(f"   {config['name']}:")
        print(f"     - Epochs: {config['epochs']}")
        print(f"     - Learning Rate: {config['learning_rate']}")
        print(f"     - Description: {config['description']}")
    
    print("=" * 50)
    return plan

def save_training_log(config: Dict[str, Any], start_time: float):
    """Save training execution log"""
    log_data = {
        "training_start_time": start_time,
        "configuration": config,
        "system_info": {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }
    }
    
    with open("training_execution_log.json", 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print("📝 Training log saved to training_execution_log.json")

def main():
    """Main training execution function"""
    print("🚀 TANTRA ULTRA LARGE MODEL TRAINING EXECUTION")
    print("=" * 60)
    
    # Record start time
    start_time = time.time()
    
    # Check system requirements
    device = check_system_requirements()
    
    # Check data availability
    if not check_data_availability():
        print("❌ Insufficient training data. Please run prepare_real_data.py first.")
        return
    
    # Load training configuration
    print("⚙️  Loading Training Configuration...")
    try:
        config = load_training_config("ultra_large_training_setup.json")
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return
    
    # Validate configuration
    if not validate_training_setup(config):
        print("❌ Configuration validation failed")
        return
    
    # Print training summary
    print_training_summary(config)
    
    # Estimate training time
    estimate_training_time(device, 225)  # 225 examples from our data
    
    # Create training plan
    training_plan = create_training_plan()
    
    # Ask for confirmation (in a real scenario)
    print("\n🤔 Ready to start training?")
    print("⚠️  This will take a significant amount of time, especially on CPU")
    print("💡 Consider using a GPU-enabled environment for faster training")
    
    # Save training log
    save_training_log(config.__dict__, start_time)
    
    # Start training
    print("\n🚀 Starting Ultra Large Model Training...")
    print("=" * 60)
    
    try:
        # Create trainer
        trainer = UltraLargeModelTrainer()
        
        # Run training
        trainer.run_training()
        
        # Calculate total time
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total execution time: {total_time/3600:.2f} hours")
        print("💾 Model saved to Model/weights/")
        print("📊 Training logs available in logs/")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("📝 Check logs for detailed error information")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)