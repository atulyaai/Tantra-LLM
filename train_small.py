"""Train the 'small' (~28M) base + all experts + routers into Model_small/.
Run: python train_small.py   (log: logs/train_small.log)"""
import logging
import sys
from pathlib import Path

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.FileHandler("logs/train_small.log", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])

from Tantra.main import STAGE1_EXPERTS, STAGE2_EXPERTS
from Tantra.training.tantra_trainer import TantraTrainer, TrainArgs, setup_cpu

setup_cpu()
t = TantraTrainer("Model_small", ["Datasets/curated"])
if not t.base_path.exists():
    t.train_base("small", TrainArgs(steps=2500, batch_size=16, max_len=192, lr=1e-3, warmup=100, log_every=50),
                 resume=False)
    t.snapshot("small_base")
model = t.load_model()
for name in STAGE1_EXPERTS + STAGE2_EXPERTS:
    if name in model.expert_ranks and (t.experts_dir / f"{name}.pt").exists():
        continue  # resume: already trained
    t.train_expert(name, model, TrainArgs(steps=300, lr=3e-3, warmup=20, batch_size=16, max_len=192, log_every=100))
t.train_router(model)
t.snapshot("small_full")
logging.info("DONE")
