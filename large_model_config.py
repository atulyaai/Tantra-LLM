#!/usr/bin/env python3
"""
Large Tantra Model Configuration
Creates a much larger model for training and fine-tuning
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.tantra_llm import TantraConfig
from src.training.training_config import TrainingConfig

def create_large_tantra_config():
    """Create a large Tantra model configuration"""
    return TantraConfig(
        # Much larger model architecture
        d_model=2048,           # 4x larger than default (512)
        n_layers=32,            # 2.67x larger than default (12)
        n_heads=16,             # 2x larger than default (8)
        d_ff=8192,              # 4x larger than default (2048)
        vocab_size=100000,      # 2x larger than default (50000)
        max_seq_length=16384,   # 2x larger than default (8192)
        
        # OCR settings optimized for large model
        ocr_image_width=1024,   # 2x larger
        ocr_image_height=1024,  # 2x larger
        ocr_font_size=10,       # Smaller font for more content
        ocr_precision=8,        # Higher precision
        
        # Memory settings for large model
        memory_window_size=200000,  # 4x larger
        ocr_memory_bank_size=5000, # 5x larger
        context_retention=0.95,    # Higher retention
        
        # Training settings for large model
        learning_rate=5e-5,     # Lower learning rate for stability
        batch_size=2,           # Smaller batch size due to memory
        gradient_accumulation_steps=8,  # More accumulation steps
    )

def create_large_training_config():
    """Create training configuration for large model"""
    return TrainingConfig(
        # Model settings
        model_name="tantra_large_v1.0",
        base_model_path="Model/weights/Tantra_large_v1.0.pt",
        
        # Training parameters optimized for large model
        learning_rate=5e-5,     # Lower learning rate
        batch_size=1,           # Very small batch size
        num_epochs=20,          # More epochs for fine-tuning
        gradient_accumulation_steps=16,  # More accumulation
        warmup_steps=500,       # More warmup steps
        max_grad_norm=0.5,      # Lower gradient clipping
        
        # Conversational settings
        conversation_max_length=4096,    # 2x larger
        conversation_context_window=1024, # 2x larger
        conversation_temperature=0.6,    # Slightly lower
        conversation_top_p=0.85,         # Slightly lower
        conversation_top_k=40,           # Slightly lower
        
        # Speech settings
        speech_sample_rate=22050,        # Higher quality
        speech_max_duration=60.0,        # 2x longer
        speech_hop_length=256,           # Higher resolution
        speech_n_fft=4096,               # 2x larger
        speech_n_mels=128,               # More mel bins
        speech_f_max=11025.0,            # Higher frequency range
        
        # Hardware settings for large model
        device="cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
        num_workers=2,                   # Fewer workers due to memory
        mixed_precision=True,            # Essential for large model
        gradient_checkpointing=True,     # Essential for memory
        
        # Evaluation settings
        eval_steps=250,                  # More frequent evaluation
        save_steps=500,                  # More frequent saving
        logging_steps=50,                # More frequent logging
    )

if __name__ == "__main__":
    # Create configurations
    tantra_config = create_large_tantra_config()
    training_config = create_large_training_config()
    
    print("🔤 Large Tantra Model Configuration Created!")
    print("=" * 60)
    print(f"Model Architecture:")
    print(f"  • Dimensions: {tantra_config.d_model}")
    print(f"  • Layers: {tantra_config.n_layers}")
    print(f"  • Attention Heads: {tantra_config.n_heads}")
    print(f"  • Feed Forward: {tantra_config.d_ff}")
    print(f"  • Vocabulary: {tantra_config.vocab_size:,}")
    print(f"  • Max Sequence: {tantra_config.max_seq_length:,}")
    print()
    print(f"Estimated Parameters: ~{tantra_config.d_model * tantra_config.n_layers * tantra_config.d_ff // 1000000}M")
    print(f"Estimated Size: ~{tantra_config.d_model * tantra_config.n_layers * tantra_config.d_ff * 4 // (1024*1024)}MB")
    print()
    print(f"Training Configuration:")
    print(f"  • Learning Rate: {training_config.learning_rate}")
    print(f"  • Batch Size: {training_config.batch_size}")
    print(f"  • Epochs: {training_config.num_epochs}")
    print(f"  • Device: {training_config.device}")
    print("=" * 60)