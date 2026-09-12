"""
serious_train.py — strict from-scratch real-dataset trainer.

This launcher intentionally bypasses main.py's synthetic demo path and its
DataParallel wrapper. It is designed for the 0.6B dense Tantra baseline:
  vocab=65536, layers=32, dim=1024, heads=16

Safety invariants:
  * real JSONL dataset only
  * fresh tokenizer if the cached tokenizer is absent or has the wrong vocab
  * no checkpoint resume
  * no synthetic enrichment
  * no auto-growth
  * no DPO / auto-pilot
  * one GPU process (safe baseline; add DDP only after this path is validated)
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import torch
from torch.utils.data import DataLoader

from Tantra.config import NeuroCoreConfig, VocabConfig
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.model import NeuroCoreModel
from Tantra.train import NeuroTrainer
from Tantra.dataset import JSONLDataset, extract_corpus_sample


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(ROOT, "Model", "Serious_0.608B")


def log(msg: str) -> None:
    print(f"[SERIOUS] {msg}", flush=True)


def build_strict_tokenizer(vcfg: VocabConfig, dataset_path: str, model_dir: str) -> UnifiedTokenizer:
    os.makedirs(model_dir, exist_ok=True)
    tok_path = os.path.join(model_dir, "tokenizer_65536.json")

    # Never silently use a tokenizer with a different vocabulary.
    if os.path.isfile(tok_path):
        tok = ByteBPETokenizer.load(tok_path, vcfg)
        actual = tok.vocab_size
        if actual != vcfg.vocab_size:
            raise RuntimeError(
                f"Cached tokenizer has vocab={actual}, expected {vcfg.vocab_size}. "
                "Delete tokenizer_65536.json and rebuild."
            )
        log(f"Using verified tokenizer: {tok_path} | vocab={actual:,}")
        return UnifiedTokenizer(vcfg, tok, MegabytePatcher())

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Real training dataset not found: {dataset_path}")

    corpus_path = os.path.join(model_dir, "tokenizer_corpus.txt")
    log(f"Building fresh 65,536-token BPE tokenizer from: {dataset_path}")
    extract_corpus_sample(dataset_path, corpus_path, max_lines=None)

    bpe = ByteBPETokenizer(vcfg)
    bpe.train(
        [corpus_path],
        vocab_size=vcfg.vocab_size,
        special_tokens=list(vcfg.special_tokens.keys()),
    )
    actual = bpe.vocab_size
    if actual != vcfg.vocab_size:
        raise RuntimeError(
            f"Tokenizer training produced vocab={actual}, expected exactly {vcfg.vocab_size}. "
            "Training is aborted rather than using a mismatched embedding vocabulary."
        )
    bpe.save(tok_path)
    log(f"Fresh tokenizer saved: {tok_path} | vocab={actual:,}")
    return UnifiedTokenizer(vcfg, bpe, MegabytePatcher())


def main() -> None:
    p = argparse.ArgumentParser(description="Strict Tantra real-dataset 0.608B trainer")
    p.add_argument("--dataset", required=True)
    p.add_argument("--val-dataset", required=True)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--data-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("STRICT TRAINING ABORTED: CUDA GPU is required for the 0.608B baseline.")

    device = torch.device("cuda:0")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Explicit architecture — never restore architecture from a checkpoint.
    vcfg = VocabConfig(vocab_size=65536, byte_bpe_vocab=65536)
    cfg = NeuroCoreConfig(vocab=vcfg)
    cfg.block.num_layers = 32
    cfg.block.alra.dim = 1024
    cfg.block.alra.num_heads = 16
    cfg.block.alra.head_dim = 64
    cfg.block.sgp.dim = 1024
    cfg.moe.num_experts = 1
    cfg.moe.real_top1 = False
    cfg.bitnet.enabled = True
    cfg.bitnet.quantize_mode = "ternary"
    cfg.bitnet.use_shadow_weights = True

    log("Architecture: vocab=65,536 | layers=32 | dim=1024 | heads=16 | dense | BitNet=ternary")
    tok = build_strict_tokenizer(vcfg, args.dataset, args.model_dir)

    model = NeuroCoreModel(cfg).to(device)
    total_params = sum(x.numel() for x in model.parameters())
    log(f"Model parameters: {total_params:,} ({total_params/1e9:.3f}B)")
    if total_params < 500_000_000:
        raise RuntimeError("Model is below the requested 0.5B class; aborting.")

    # Real JSONL only. SFT masking is enabled, but synthetic enrichment is 0.0.
    train_ds = JSONLDataset(
        args.dataset,
        tok,
        seq_len=args.seq_len,
        max_samples=None,
        mask_non_assistant=True,
        split="all",
        val_ratio=0.0,
        pack_sequences=False,
        shuffle=True,
        shuffle_buf_size=2000,
        seed=args.seed,
    )
    val_ds = JSONLDataset(
        args.val_dataset,
        tok,
        seq_len=args.seq_len,
        max_samples=None,
        mask_non_assistant=True,
        split="all",
        val_ratio=0.0,
        pack_sequences=False,
        shuffle=False,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.data_workers,
        pin_memory=True,
        persistent_workers=args.data_workers > 0,
        prefetch_factor=2 if args.data_workers > 0 else None,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    trainer = NeuroTrainer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name="adamw",
        total_steps=args.steps,
        warmup_steps=args.warmup,
        grad_accumulation_steps=args.grad_accum,
        use_latent_reasoning=False,
        use_mtp_loss=False,
        mtp_loss_weight=0.0,
        max_grad_norm=0.5,
    )

    latest = os.path.join(args.model_dir, "Latest", "checkpoint_latest.pt")
    os.makedirs(os.path.dirname(latest), exist_ok=True)

    # Hard safety check: this launcher never loads an existing checkpoint.
    if os.path.exists(latest):
        log(f"Existing checkpoint detected at {latest} — IGNORING it (fresh run by design).")

    last_saved = {"step": -1}

    def checkpoint_cb(step: int) -> None:
        if step != last_saved["step"]:
            trainer.save_checkpoint(latest, save_optimizer=True, async_write=True)
            last_saved["step"] = step

    t0 = time.time()
    log("============================================================")
    log("REAL DATASET TRAINING START — NO SYNTHETIC / NO RESUME")
    log(f"Steps={args.steps:,} | seq={args.seq_len} | batch={args.batch_size} | grad_accum={args.grad_accum}")
    log("============================================================")

    try:
        trainer.train_dataset(
            train_loader,
            max_steps=args.steps,
            log_every=args.log_every,
            eval_every=args.eval_every,
            checkpoint_every=args.checkpoint_every,
            checkpoint_callback=checkpoint_cb,
            tokenizer=tok,
            enrichment_rate=0.0,
            use_latent_reasoning=False,
            auto_growth=False,
            val_loader=val_loader,
            max_val_batches=50,
        )
    finally:
        trainer.save_checkpoint(latest, save_optimizer=True, async_write=False)
        elapsed = time.time() - t0
        log(f"Training ended at step {trainer.step_count:,}; elapsed={elapsed/3600:.2f}h")
        log(f"Checkpoint: {latest}")


if __name__ == "__main__":
    main()
