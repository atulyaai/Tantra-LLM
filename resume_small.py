"""Resume small model training from step 400 checkpoint."""
import logging
import sys
from pathlib import Path

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.FileHandler("logs/train_small_resume.log", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])

from Tantra.training.tantra_trainer import TantraTrainer, TrainArgs, setup_cpu
from Tantra.main import STAGE1_EXPERTS, STAGE2_EXPERTS

setup_cpu()

# Resume from the checkpoint at step 400
t = TantraTrainer("Model_small", ["Datasets/curated"])
logging.info("Resuming small model from step 400 checkpoint...")

# Load the base model (checkpoint exists at step 400)
model = t.load_model()
logging.info(f"Loaded base: {model.num_params()/1e6:.2f}M params, {model.cfg.n_layers} layers")

# Continue base training from step 400 to 2500 (2100 more steps)
# Reduced batch + gradient accumulation to prevent OOM on 14GB RAM
model = t.train_base("small",
                     TrainArgs(steps=2100, batch_size=8, max_len=128, lr=1e-3, warmup=100, log_every=50,
                               gradient_accumulation=2),
                     resume=True)
t.snapshot("small_base")
logging.info("Base training complete, snapshot saved")

# Train all experts
experts = STAGE1_EXPERTS + STAGE2_EXPERTS
logging.info(f"Training {len(experts)} experts...")
for name in experts:
    if name in model.expert_ranks and (t.experts_dir / f"{name}.pt").exists():
        logging.info(f"Skipping {name} (already trained)")
        continue
    t.train_expert(name, model, TrainArgs(steps=300, lr=3e-3, warmup=20, batch_size=8, max_len=128, log_every=100))

# Train router
t.train_router(model)
t.snapshot("small_full")
logging.info("DONE: small model complete")