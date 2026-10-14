"""
Tantra-LLM Full Base Training Pipeline
======================================
Trains the canonical 28.8M base model in Model/ with the 10,277 ByteLevel
BPE bilingual tokenizer using gradient accumulation and cosine decay.
"""

import logging
import sys
from pathlib import Path

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("logs/train.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

from Tantra.training.tantra_trainer import TantraTrainer, TrainArgs, setup_cpu


def main():
    setup_cpu()
    logging.info("Starting canonical Tantra 28.8M base model training in Model/...")

    trainer = TantraTrainer(
        model_dir="Model",
        data_dirs=["Datasets/curated"],
        general_dir="Datasets/general",
    )

    args = TrainArgs(
        steps=1000,
        batch_size=8,
        gradient_accumulation=2,
        max_len=160,
        lr=1.5e-4,
        warmup=30,
        log_every=25,
        weight_decay=0.01,
    )

    logging.info(
        f"Configuration: steps={args.steps}, batch_size={args.batch_size}, "
        f"accum={args.gradient_accumulation}, lr={args.lr}, max_len={args.max_len}"
    )

    model = trainer.train_base(
        preset="small",
        a=args,
        resume=True,
        chat_fraction=0.20,
        general_limit=60_000,
    )

    trainer.snapshot("base_full_1000")
    logging.info(f"Training complete. Base parameters: {model.num_params():,}")


if __name__ == "__main__":
    main()
