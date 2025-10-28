#!/usr/bin/env python3
"""
Ultra Large Model Training Configuration
Integrates ultra-large model config with real data training
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

from src.training.training_config import TrainingConfig
from src.core.tantra_llm import TantraConfig
from ultra_large_model_config import create_ultra_large_tantra_config, create_ultra_large_training_config

@dataclass
class UltraLargeTrainingConfig:
    """Configuration for ultra-large model training with real data"""
    
    # Model configuration
    model_config: TantraConfig
    
    # Training configuration  
    training_config: TrainingConfig
    
    # Data configuration
    data_config: Dict[str, Any]
    
    # Hardware configuration
    hardware_config: Dict[str, Any]
    
    # Training strategy
    training_strategy: Dict[str, Any]
    
    # Evaluation configuration
    evaluation_config: Dict[str, Any]

def create_ultra_large_training_setup() -> UltraLargeTrainingConfig:
    """Create comprehensive ultra-large training configuration"""
    
    # Get base configurations
    model_config = create_ultra_large_tantra_config()
    training_config = create_ultra_large_training_config()
    
    # Data configuration
    data_config = {
        "real_data_sources": [
            "data/raw/real_conversations.json",
            "data/raw/real_technical_qa.json", 
            "data/raw/real_creative_tasks.json",
            "data/raw/real_combined_dataset.json"
        ],
        "data_validation": {
            "min_examples": 100,
            "max_examples": 10000,
            "quality_threshold": 0.8,
            "diversity_requirement": True
        },
        "data_preprocessing": {
            "tokenization": "custom_tantra_tokenizer",
            "max_length": 32768,  # Ultra-large max sequence length
            "padding": "dynamic",
            "truncation": "smart",
            "special_tokens": ["<|start|>", "<|end|>", "<|user|>", "<|assistant|>"]
        },
        "data_augmentation": {
            "enabled": True,
            "paraphrasing": True,
            "back_translation": False,
            "noise_injection": True,
            "augmentation_factor": 1.5
        }
    }
    
    # Hardware configuration
    hardware_config = {
        "device": "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
        "mixed_precision": True,
        "gradient_checkpointing": True,
        "dataloader_workers": 1,  # Reduced for ultra-large model
        "pin_memory": True,
        "memory_optimization": {
            "activation_checkpointing": True,
            "cpu_offload": False,  # Keep on GPU for ultra-large
            "gradient_accumulation": True,
            "micro_batch_size": 1
        }
    }
    
    # Training strategy
    training_strategy = {
        "learning_rate_schedule": {
            "type": "cosine_with_warmup",
            "warmup_steps": 1000,
            "max_lr": 2e-5,
            "min_lr": 1e-6
        },
        "optimizer": {
            "type": "AdamW",
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8
        },
        "scheduler": {
            "type": "cosine_with_warmup",
            "warmup_steps": 1000,
            "total_steps": 30000
        },
        "gradient_handling": {
            "max_grad_norm": 0.3,
            "gradient_accumulation_steps": 32,
            "gradient_clipping": True
        },
        "training_phases": [
            {
                "name": "warmup",
                "epochs": 2,
                "learning_rate": 1e-6,
                "batch_size": 1
            },
            {
                "name": "main_training", 
                "epochs": 25,
                "learning_rate": 2e-5,
                "batch_size": 1
            },
            {
                "name": "fine_tuning",
                "epochs": 3,
                "learning_rate": 5e-6,
                "batch_size": 1
            }
        ]
    }
    
    # Evaluation configuration
    evaluation_config = {
        "metrics": [
            "perplexity",
            "bleu_score", 
            "rouge_score",
            "accuracy",
            "f1_score",
            "custom_ocr_accuracy"
        ],
        "evaluation_frequency": {
            "steps": 100,
            "epochs": 1
        },
        "validation_split": 0.1,
        "test_split": 0.05,
        "evaluation_tasks": [
            "conversational_qa",
            "technical_problem_solving",
            "creative_writing",
            "code_generation",
            "ocr_text_understanding"
        ]
    }
    
    return UltraLargeTrainingConfig(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        hardware_config=hardware_config,
        training_strategy=training_strategy,
        evaluation_config=evaluation_config
    )

def save_training_config(config: UltraLargeTrainingConfig, filepath: str = "ultra_large_training_setup.json"):
    """Save training configuration to JSON file"""
    
    # Convert dataclasses to dictionaries
    config_dict = {
        "model_config": asdict(config.model_config),
        "training_config": asdict(config.training_config),
        "data_config": config.data_config,
        "hardware_config": config.hardware_config,
        "training_strategy": config.training_strategy,
        "evaluation_config": config.evaluation_config
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Training configuration saved to {filepath}")

def load_training_config(filepath: str = "ultra_large_training_setup.json") -> UltraLargeTrainingConfig:
    """Load training configuration from JSON file"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # Reconstruct dataclasses
    model_config = TantraConfig(**config_dict["model_config"])
    training_config = TrainingConfig(**config_dict["training_config"])
    
    return UltraLargeTrainingConfig(
        model_config=model_config,
        training_config=training_config,
        data_config=config_dict["data_config"],
        hardware_config=config_dict["hardware_config"],
        training_strategy=config_dict["training_strategy"],
        evaluation_config=config_dict["evaluation_config"]
    )

def validate_training_setup(config: UltraLargeTrainingConfig) -> bool:
    """Validate the training configuration"""
    
    print("🔍 Validating Ultra Large Training Configuration...")
    
    # Check model parameters
    model = config.model_config
    if model.d_model < 2048:
        print("⚠️  Warning: d_model is less than 2048 for ultra-large model")
    
    if model.n_layers < 32:
        print("⚠️  Warning: n_layers is less than 32 for ultra-large model")
    
    if model.vocab_size < 100000:
        print("⚠️  Warning: vocab_size is less than 100,000 for ultra-large model")
    
    # Check data sources
    data_sources = config.data_config["real_data_sources"]
    for source in data_sources:
        if not os.path.exists(source):
            print(f"❌ Data source not found: {source}")
            return False
    
    # Check hardware requirements
    if config.hardware_config["device"] == "cpu":
        print("⚠️  Warning: Training on CPU will be very slow for ultra-large model")
    
    # Check memory requirements
    total_params = model.d_model * model.n_layers * model.n_heads * 4  # Rough estimate
    estimated_memory_gb = total_params * 4 / (1024**3)  # 4 bytes per parameter
    
    print(f"📊 Estimated model memory requirement: {estimated_memory_gb:.2f} GB")
    
    if estimated_memory_gb > 16:
        print("⚠️  Warning: Model may require more than 16GB GPU memory")
    
    print("✅ Configuration validation complete")
    return True

def print_training_summary(config: UltraLargeTrainingConfig):
    """Print a summary of the training configuration"""
    
    print("\n" + "="*60)
    print("🎯 ULTRA LARGE MODEL TRAINING SUMMARY")
    print("="*60)
    
    # Model summary
    model = config.model_config
    print(f"🏗️  MODEL ARCHITECTURE:")
    print(f"   - Embedding dimension: {model.d_model:,}")
    print(f"   - Number of layers: {model.n_layers}")
    print(f"   - Attention heads: {model.n_heads}")
    print(f"   - Feed-forward dimension: {model.d_ff:,}")
    print(f"   - Vocabulary size: {model.vocab_size:,}")
    print(f"   - Max sequence length: {model.max_seq_length:,}")
    
    # Training summary
    training = config.training_config
    print(f"\n🚀 TRAINING CONFIGURATION:")
    print(f"   - Model name: {training.model_name}")
    print(f"   - Learning rate: {training.learning_rate}")
    print(f"   - Batch size: {training.batch_size}")
    print(f"   - Epochs: {training.num_epochs}")
    print(f"   - Gradient accumulation: {training.gradient_accumulation_steps}")
    print(f"   - Device: {training.device}")
    
    # Data summary
    data_sources = config.data_config["real_data_sources"]
    print(f"\n📊 DATA SOURCES:")
    for source in data_sources:
        if os.path.exists(source):
            with open(source, 'r') as f:
                data = json.load(f)
                print(f"   - {source}: {len(data)} examples")
        else:
            print(f"   - {source}: Not found")
    
    # Hardware summary
    hardware = config.hardware_config
    print(f"\n💻 HARDWARE CONFIGURATION:")
    print(f"   - Device: {hardware['device']}")
    print(f"   - Mixed precision: {hardware['mixed_precision']}")
    print(f"   - Gradient checkpointing: {hardware['gradient_checkpointing']}")
    
    print("="*60)

if __name__ == "__main__":
    # Create and save configuration
    config = create_ultra_large_training_setup()
    save_training_config(config)
    
    # Validate configuration
    validate_training_setup(config)
    
    # Print summary
    print_training_summary(config)