"""
Tantra-LLM Fusion Projector Training Script

Trains vision and audio projection layers to align multimodal embeddings
into the base model's hidden space using contrastive learning.

Usage:
    python scripts/train_fusion.py
    python scripts/train_fusion.py --epochs 10 --batch-size 16 --lr 3e-4
"""

import sys
import os
import argparse
import logging
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from training.training_config import FusionTrainingConfig
from training.fusion_trainer import FusionTrainer, FusionProjector
from training.datasets.multimodal_dataset import MultimodalDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("train_fusion")


def parse_args():
    parser = argparse.ArgumentParser(description="Train fusion projectors")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--vision-dim", type=int, default=768, help="Vision encoder output dim")
    parser.add_argument("--audio-dim", type=int, default=512, help="Audio encoder output dim")
    parser.add_argument("--model-dim", type=int, default=4096, help="Base model hidden dim (projector target)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on projectors")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"  TANTRA-LLM FUSION PROJECTOR TRAINING")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {'Available (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'Not available (CPU mode)'}")
    print(f"{'='*60}")
    
    # ── 1. Generate synthetic dataset ──
    # In production, replace this with real pre-computed embeddings
    logger.info(f"Generating {args.num_samples} synthetic multimodal samples...")
    full_dataset = MultimodalDataset.generate_synthetic(
        num_samples=args.num_samples,
        vision_dim=args.vision_dim,
        audio_dim=args.audio_dim,
        target_dim=args.model_dim,
        seq_len=32
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    logger.info(f"Train: {train_size} samples, Val: {val_size} samples")
    
    # ── 2. Create projectors ──
    vision_projector = FusionProjector(
        input_dim=args.vision_dim, 
        output_dim=args.model_dim
    )
    audio_projector = FusionProjector(
        input_dim=args.audio_dim, 
        output_dim=args.model_dim
    )
    
    # ── 3. Configure trainer ──
    config = FusionTrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
    )
    
    trainer = FusionTrainer(
        config=config,
        vision_projector=vision_projector,
        audio_projector=audio_projector,
        base_model=None,  # No base model needed for projector-only training
        use_compile=args.compile,
        patience=args.patience,
    )
    
    # -- 4. Train --
    start = time.time()
    history = trainer.fit(train_dataset, val_dataset)
    elapsed = time.time() - start
    
    # -- 5. Report --
    print(f"\n{'-'*60}")
    print(f"  TRAINING REPORT")
    print(f"{'-'*60}")
    print(f"  Wall time:        {elapsed:.2f}s")
    print(f"  Epochs completed: {len(history['train_loss'])}")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    if history["val_loss"]:
        print(f"  Best val loss:    {min(history['val_loss']):.6f}")
    
    # Loss convergence check
    if len(history["train_loss"]) >= 2:
        first = history["train_loss"][0]
        last = history["train_loss"][-1]
        reduction = ((first - last) / abs(first)) * 100 if first != 0 else 0
        print(f"  Loss reduction:   {reduction:.1f}%")
        
        if reduction > 5:
            print(f"  Status:           [PASS] Loss is converging")
        elif reduction > 0:
            print(f"  Status:           [WARN] Slow convergence - try higher LR or more epochs")
        else:
            print(f"  Status:           [FAIL] Loss not decreasing - check loss function or LR")
    
    print(f"{'-'*60}")
    
    # Per-epoch details
    print(f"\n  Epoch | Train Loss | Val Loss   | LR         | Time")
    print(f"  ------|------------|------------|------------|------")
    for i in range(len(history["train_loss"])):
        t_loss = history["train_loss"][i]
        v_loss = history["val_loss"][i] if i < len(history["val_loss"]) else float("nan")
        lr = history["lr"][i]
        t = history["epoch_time_s"][i]
        print(f"  {i+1:5d} | {t_loss:10.6f} | {v_loss:10.6f} | {lr:10.2e} | {t:.2f}s")
    
    print()


if __name__ == "__main__":
    main()
