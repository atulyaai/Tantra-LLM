"""Train a separate CPU profile without touching Model/Latest."""
from __future__ import annotations
import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Tantra.config import VocabConfig
from Tantra.cpu_profiles import build_cpu_model
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from main import run_dataset_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a separate CPU Tantra profile.")
    parser.add_argument("--profile", choices=["dense", "moe2", "micro10"], default="dense")
    parser.add_argument("--attention", choices=["causal", "alra"], default="causal")
    parser.add_argument("--vocab-size", type=int, default=32768, help="Must match the supplied tokenizer and checkpoint (32768 or 65536)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", default=os.path.join(REPO_ROOT, "Datasets"))
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--data-workers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in --model-dir if present; otherwise start fresh.")
    parser.add_argument("--auto-growth", action="store_true")
    parser.add_argument("--growth-patience", type=int, default=2000, help="Optimizer steps of a sustained plateau before a layer may be added")
    parser.add_argument("--growth-min-delta", type=float, default=0.005, help="Minimum EMA-loss improvement needed to avoid growth")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated installed categories, e.g. math,science,conversation")
    parser.add_argument("--adapter", type=str, default=None, help="Train only this installed category adapter")
    parser.add_argument("--adapter-depth", type=int, default=1, help="Specialist layer stack depth for category adapters")
    args = parser.parse_args()
    vocab_size = args.vocab_size
    cfg = VocabConfig(vocab_size=vocab_size)
    tokenizer = UnifiedTokenizer(cfg, ByteBPETokenizer.load(args.tokenizer, cfg), MegabytePatcher())
    model = build_cpu_model(args.profile, attention_kind=args.attention, vocab_size=vocab_size).to("cpu")
    if args.categories:
        categories = [name.strip() for name in args.categories.split(",") if name.strip()]
        model.add_category_layers(categories, depth=args.adapter_depth, clone_layer_index=-1)
    if args.adapter:
        model.freeze_for_category(args.adapter)
    run_dataset_training(model, tokenizer, args.dataset, steps=args.steps, resume=args.resume,
        eval_every=args.eval_every, log_every=10, checkpoint_every=args.checkpoint_every,
        batch_size=args.batch_size, seq_len=args.seq_len, grad_accumulation_steps=args.grad_accum,
        data_workers=args.data_workers, use_latent_reasoning=False, use_mtp_loss=False,
        training_stage="pretrain", auto_growth=args.auto_growth, growth_patience=args.growth_patience,
        growth_min_delta=args.growth_min_delta, max_layers=args.max_layers, model_dir=args.model_dir,
        archive_checkpoints=False)

if __name__ == "__main__":
    main()
