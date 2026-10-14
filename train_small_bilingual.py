"""
Pretraining / Adaptation pipeline for Model_small with 10k Bilingual Tokenizer
=============================================================================
Runs a controlled base model training run on CPU with gradient accumulation
and cosine learning rate decay, logging step latency and loss progression.
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
        logging.FileHandler("logs/train_small_bilingual.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

from Tantra.training.tantra_trainer import TantraTrainer, TrainArgs, setup_cpu


def main():
    setup_cpu()
    logging.info("Starting Model_small bilingual base adaptation...")

    trainer = TantraTrainer(
        model_dir="Model_small",
        data_dirs=["Datasets/curated"],
        general_dir="Datasets/general",
    )

    args = TrainArgs(
        steps=100,
        batch_size=8,
        gradient_accumulation=2,
        max_len=160,
        lr=2e-4,
        warmup=20,
        log_every=10,
        weight_decay=0.01,
    )

    logging.info(f"Loaded configuration: steps={args.steps}, batch_size={args.batch_size}, accum={args.gradient_accumulation}, lr={args.lr}")
    model = trainer.train_base(
        preset="small",
        a=args,
        resume=True,
        chat_fraction=0.20,
        general_limit=40_000,
    )

    logging.info(f"Adaptation finished. Current model parameters: {model.num_params():,}")


if __name__ == "__main__":
    main()
