#!/usr/bin/env python3
"""
Ultra Large Tantra Model Configuration
Creates a massive model for training with real data
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.tantra_llm import TantraConfig
from src.training.training_config import TrainingConfig

def create_ultra_large_tantra_config():
    """Create an ultra-large Tantra model configuration"""
    return TantraConfig(
        # Massive model architecture - 10x larger than default
        d_model=4096,           # 8x larger than default (512)
        n_layers=48,            # 4x larger than default (12)
        n_heads=32,             # 4x larger than default (8)
        d_ff=16384,             # 8x larger than default (2048)
        vocab_size=200000,      # 4x larger than default (50000)
        max_seq_length=32768,   # 4x larger than default (8192)
        
        # OCR settings optimized for ultra-large model
        ocr_image_width=2048,   # 4x larger
        ocr_image_height=2048,  # 4x larger
        ocr_font_size=8,        # Smaller font for more content
        ocr_precision=10,       # Higher precision
        
        # Memory settings for ultra-large model
        memory_window_size=500000,  # 10x larger
        ocr_memory_bank_size=10000, # 10x larger
        context_retention=0.98,     # Higher retention
        
        # Training settings for ultra-large model
        learning_rate=2e-5,     # Lower learning rate for stability
        batch_size=1,           # Very small batch size due to memory
        gradient_accumulation_steps=16,  # More accumulation steps
    )

def create_ultra_large_training_config():
    """Create training configuration for ultra-large model"""
    return TrainingConfig(
        # Model settings
        model_name="tantra_ultra_large_v1.0",
        base_model_path="Model/weights/Tantra_ultra_large_v1.0.pt",
        
        # Training parameters optimized for ultra-large model
        learning_rate=2e-5,     # Lower learning rate
        batch_size=1,           # Very small batch size
        num_epochs=30,          # More epochs for fine-tuning
        gradient_accumulation_steps=32,  # More accumulation
        warmup_steps=1000,      # More warmup steps
        max_grad_norm=0.3,      # Lower gradient clipping
        
        # Conversational settings
        conversation_max_length=8192,    # 4x larger
        conversation_context_window=2048, # 4x larger
        conversation_temperature=0.5,    # Lower for stability
        conversation_top_p=0.8,          # Lower for stability
        conversation_top_k=30,           # Lower for stability
        
        # Speech settings
        speech_sample_rate=44100,        # Higher quality
        speech_max_duration=120.0,       # 4x longer
        speech_hop_length=512,           # Higher resolution
        speech_n_fft=8192,               # 4x larger
        speech_n_mels=256,               # More mel bins
        speech_f_max=22050.0,            # Higher frequency range
        
        # Hardware settings for ultra-large model
        device="cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
        num_workers=1,                   # Fewer workers due to memory
        mixed_precision=True,            # Essential for ultra-large model
        gradient_checkpointing=True,     # Essential for memory
        
        # Evaluation settings
        eval_steps=100,                  # More frequent evaluation
        save_steps=200,                  # More frequent saving
        logging_steps=25,                # More frequent logging
    )

if __name__ == "__main__":
    # Create configurations
    tantra_config = create_ultra_large_tantra_config()
    training_config = create_ultra_large_training_config()
    
    print("🔤 Ultra Large Tantra Model Configuration Created!")
    print("=" * 70)
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
    print("=" * 70)