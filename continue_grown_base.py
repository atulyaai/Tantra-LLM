"""Continue base training on the grown 8-layer model.

Teaches layers 7-8 (currently identity) to contribute. After this, all
20 experts are stale and must be retrained.
Run: python continue_grown_base.py  (log: logs/grow_continue.log)
"""
import logging
import sys
from pathlib import Path

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.FileHandler("logs/grow_continue.log", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])

from Tantra.training.tantra_trainer import TantraTrainer, TrainArgs, setup_cpu

setup_cpu()
t = TantraTrainer("Model")
model = t.train_base("tiny",
                     TrainArgs(steps=500, batch_size=16, max_len=192, lr=5e-4, warmup=50, log_every=50),
                     resume=True)
t.snapshot("grown8_base_continued")
logging.info("DONE: base continuation finished")
