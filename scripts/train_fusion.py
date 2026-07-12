"""
Tantra-LLM Fusion Projector Training + Verification Script

Trains projectors, tests checkpoint save/load, measures inference speed.

Usage:
    python scripts/train_fusion.py
    python scripts/train_fusion.py --epochs 15 --batch-size 32 --lr 5e-4
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_fusion")


def parse_args():
    p = argparse.ArgumentParser(description="Train fusion projectors")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--vision-dim", type=int, default=768)
    p.add_argument("--audio-dim", type=int, default=512)
    p.add_argument("--model-dim", type=int, default=4096)
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  TANTRA-LLM FUSION PROJECTOR TRAINING")
    print(f"  PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  CPU mode ({os.cpu_count()} threads)")
    print(f"{'='*60}")

    # -- 1. Dataset --
    logger.info(f"Generating {args.num_samples} synthetic samples...")
    full_ds = MultimodalDataset.generate_synthetic(
        num_samples=args.num_samples,
        vision_dim=args.vision_dim,
        audio_dim=args.audio_dim,
        target_dim=args.model_dim,
        seq_len=32
    )

    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
    logger.info(f"Train: {train_size}, Val: {val_size}")

    # -- 2. Projectors (smaller, with dropout) --
    vis_proj = FusionProjector(args.vision_dim, args.model_dim, dropout=args.dropout)
    aud_proj = FusionProjector(args.audio_dim, args.model_dim, dropout=args.dropout)

    # -- 3. Config --
    config = FusionTrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        grad_accum=args.grad_accum,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        dropout=args.dropout,
    )

    trainer = FusionTrainer(
        config=config,
        vision_projector=vis_proj,
        audio_projector=aud_proj,
        patience=args.patience,
    )

    # -- 4. Train --
    t0 = time.time()
    history = trainer.fit(train_ds, val_ds)
    train_time = time.time() - t0

    # -- 5. Checkpoint Save/Load Verification --
    print(f"\n{'-'*60}")
    print(f"  CHECKPOINT VERIFICATION")
    print(f"{'-'*60}")

    ckpt_path = "checkpoints/final_projectors.pt"
    if os.path.exists(ckpt_path):
        ckpt_size = os.path.getsize(ckpt_path) / (1024 * 1024)
        print(f"  Checkpoint file: {ckpt_path} ({ckpt_size:.1f} MB)")

        # Create fresh projectors and load
        vis_proj2 = FusionProjector(args.vision_dim, args.model_dim, dropout=args.dropout)
        aud_proj2 = FusionProjector(args.audio_dim, args.model_dim, dropout=args.dropout)
        trainer2 = FusionTrainer(
            config=config,
            vision_projector=vis_proj2,
            audio_projector=aud_proj2,
            patience=args.patience,
        )
        trainer2.load_checkpoint(ckpt_path)

        # Verify weights match (using allclose to support FP16 conversion tolerance)
        for (n1, p1), (n2, p2) in zip(
            trainer.vision_projector.named_parameters(),
            trainer2.vision_projector.named_parameters()
        ):
            # Convert p1 to match the loaded dtype of p2 for correct comparison
            if not torch.allclose(p1.to(p2.dtype), p2, atol=1e-3):
                print(f"  [FAIL] Vision param mismatch: {n1}")
                break
        else:
            print(f"  [PASS] Vision projector weights verified")

        for (n1, p1), (n2, p2) in zip(
            trainer.audio_projector.named_parameters(),
            trainer2.audio_projector.named_parameters()
        ):
            if not torch.allclose(p1.to(p2.dtype), p2, atol=1e-3):
                print(f"  [FAIL] Audio param mismatch: {n1}")
                break
        else:
            print(f"  [PASS] Audio projector weights verified")
    else:
        print(f"  [WARN] No checkpoint found at {ckpt_path}")

    # -- 6. Inference Speed Test --
    print(f"\n{'-'*60}")
    print(f"  INFERENCE SPEED TEST")
    print(f"{'-'*60}")

    trainer.vision_projector.eval()
    trainer.audio_projector.eval()

    # Warmup
    dummy_v = torch.randn(1, args.vision_dim, device=trainer.device)
    dummy_a = torch.randn(1, args.audio_dim, device=trainer.device)
    with torch.no_grad():
        for _ in range(5):
            trainer.vision_projector(dummy_v)
            trainer.audio_projector(dummy_a)

    # Benchmark single inference
    n_runs = 100
    t_start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            trainer.vision_projector(dummy_v)
            trainer.audio_projector(dummy_a)
    t_total = time.time() - t_start
    per_call_ms = (t_total / n_runs) * 1000

    print(f"  Single inference: {per_call_ms:.2f}ms (avg over {n_runs} runs)")

    # Benchmark batch inference
    dummy_v_batch = torch.randn(32, args.vision_dim, device=trainer.device)
    dummy_a_batch = torch.randn(32, args.audio_dim, device=trainer.device)
    t_start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            trainer.vision_projector(dummy_v_batch)
            trainer.audio_projector(dummy_a_batch)
    t_total = time.time() - t_start
    batch_ms = (t_total / n_runs) * 1000

    print(f"  Batch-32 inference: {batch_ms:.2f}ms (avg over {n_runs} runs)")
    print(f"  Throughput: {32 / (batch_ms / 1000):.0f} samples/sec")

    # -- 7. Final Report --
    print(f"\n{'='*60}")
    print(f"  TRAINING REPORT")
    print(f"{'='*60}")
    print(f"  Wall time:        {train_time:.2f}s")
    print(f"  Epochs completed: {len(history['train_loss'])}")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    if history["val_loss"]:
        print(f"  Best val loss:    {min(history['val_loss']):.6f}")

    if len(history["train_loss"]) >= 2:
        first = history["train_loss"][0]
        last = history["train_loss"][-1]
        reduction = ((first - last) / abs(first)) * 100 if first != 0 else 0
        print(f"  Loss reduction:   {reduction:.1f}%")
        if reduction > 5:
            print(f"  Status:           [PASS] Converging")
        elif reduction > 0:
            print(f"  Status:           [WARN] Slow - try higher LR or more epochs")
        else:
            print(f"  Status:           [FAIL] Not decreasing")

    print(f"\n  Epoch | Train Loss | Val Loss   | LR         | Time")
    print(f"  ------|------------|------------|------------|------")
    for i in range(len(history["train_loss"])):
        tl = history["train_loss"][i]
        vl = history["val_loss"][i] if i < len(history["val_loss"]) else float("nan")
        lr = history["lr"][i]
        et = history["epoch_time_s"][i]
        print(f"  {i+1:5d} | {tl:10.6f} | {vl:10.6f} | {lr:10.2e} | {et:.2f}s")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
