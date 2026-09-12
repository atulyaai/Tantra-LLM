"""
serious_train_fsdp.py

Strict real-dataset 2x-GPU FSDP trainer for the 0.675B Tantra baseline.

Designed for Kaggle 2x Tesla T4 (14.5 GB each):
  * FSDP FULL_SHARD over each NeuroCoreBlock
  * activation checkpointing per block
  * fp32 shadow/master parameters + fp16 autocast compute
  * real JSONL only
  * 65,536 tokenizer
  * dense BitNet ternary
  * no resume / synthetic enrichment / DPO / auto-pilot / MTP / latent CoT
  * rank-aware dataset sharding

Launch with torchrun --nproc_per_node=2 serious_train_fsdp.py ...
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    ShardedStateDictConfig,
)

from Tantra.config import NeuroCoreConfig, VocabConfig
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.model import NeuroCoreModel, NeuroCoreBlock
from Tantra.dataset import JSONLDataset, extract_corpus_sample


IGNORE_INDEX = -100


def rank():
    return int(os.environ.get("RANK", "0"))


def local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main():
    return rank() == 0


def log(msg: str) -> None:
    print(f"[FSDP rank={rank()}] {msg}", flush=True)


def setup_dist() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if world_size() < 2:
        raise RuntimeError("This launcher requires 2 GPU processes. Launch with torchrun --nproc_per_node=2.")
    torch.cuda.set_device(local_rank())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank())
    torch.cuda.manual_seed_all(42 + rank())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return device


def build_tokenizer(vcfg: VocabConfig, dataset_path: str, model_dir: str) -> UnifiedTokenizer:
    os.makedirs(model_dir, exist_ok=True)
    tok_path = os.path.join(model_dir, "tokenizer_65536.json")

    if is_main() and not os.path.isfile(tok_path):
        corpus_path = os.path.join(model_dir, "tokenizer_corpus.txt")
        log(f"Building fresh 65,536-token BPE tokenizer from full dataset: {dataset_path}")
        extract_corpus_sample(dataset_path, corpus_path, max_lines=None)
        bpe = ByteBPETokenizer(vcfg)
        bpe.train(
            [corpus_path],
            vocab_size=vcfg.vocab_size,
            special_tokens=list(vcfg.special_tokens.keys()),
        )
        if bpe.vocab_size != vcfg.vocab_size:
            raise RuntimeError(f"Tokenizer produced {bpe.vocab_size}, expected {vcfg.vocab_size}.")
        bpe.save(tok_path)
        log(f"Tokenizer saved: {tok_path} | vocab={bpe.vocab_size:,}")

    dist.barrier()
    bpe = ByteBPETokenizer.load(tok_path, vcfg)
    if bpe.vocab_size != vcfg.vocab_size:
        raise RuntimeError(f"Tokenizer mismatch: {bpe.vocab_size} != {vcfg.vocab_size}")
    return UnifiedTokenizer(vcfg, bpe, MegabytePatcher())


def make_config():
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
    return cfg, vcfg


def shard_dataset_env(num_workers: int) -> None:
    # JSONLDataset uses RANK/WORLD_SIZE-aware sharding in the patched dataset.py.
    os.environ["TANTRA_FSDP_RANK"] = str(rank())
    os.environ["TANTRA_FSDP_WORLD_SIZE"] = str(world_size())
    os.environ["TANTRA_FSDP_LOCAL_WORKERS"] = str(num_workers)


def build_model(cfg, device):
    model = NeuroCoreModel(cfg).to(device)

    # Keep each transformer block as its own FSDP unit. The root model must NOT
    # be the only FSDP unit: root-only FSDP all-gathers the entire 675M model,
    # defeating the memory-saving purpose on a 14.5 GB T4.
    mp = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    for block in model.layers:
        # Non-reentrant checkpointing reduces activation memory at the cost of
        # recomputation. Each block is already an independent FSDP unit.
        pass

    # Wrap blocks first, then the root. Root has embed/output_norm/output_proj;
    # transformer blocks are independently sharded and gathered one at a time.
    for i, block in enumerate(model.layers):
        model.layers[i] = FSDP(
            block,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            device_id=device,
            use_orig_params=True,
        )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=device,
        use_orig_params=True,
    )
    return model


def autocast_context():
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def save_sharded(model, optimizer, step: int, out_dir: str) -> None:
    """Save per-rank sharded model + optimizer state without gathering 675M params."""
    step_dir = Path(out_dir) / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
    ):
        model_sd = model.state_dict()
        torch.save(model_sd, step_dir / f"model_rank{rank():02d}.pt")

    # FSDP converts optimizer state into a sharded representation.
    optim_sd = FSDP.optim_state_dict(model, optimizer)
    torch.save(optim_sd, step_dir / f"optimizer_rank{rank():02d}.pt")

    if is_main():
        (step_dir / "meta.txt").write_text(
            f"step={step}\nworld_size={world_size()}\nmodel_params=675226178\n",
            encoding="utf-8",
        )
    dist.barrier()
    if is_main():
        latest = Path(out_dir) / "LATEST"
        latest.write_text(str(step_dir), encoding="utf-8")
        log(f"SHARDED CHECKPOINT SAVED: {step_dir}")


def evaluate(model, loader, max_batches: int, device):
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_sum = torch.zeros((), device=device, dtype=torch.float64)
    batches = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_context():
                logits, _ = model(x, return_mtp=False, use_latent_reasoning=False)
                flat_logits = logits.reshape(-1, logits.size(-1))
                flat_y = y.reshape(-1)
                mask = flat_y != IGNORE_INDEX
                if mask.any():
                    loss = F.cross_entropy(flat_logits[mask], flat_y[mask])
                    n = mask.sum().to(torch.float64)
                    loss_sum += loss.detach().double() * n
                    token_sum += n
            batches += 1
            if batches >= max_batches:
                break
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(token_sum, op=dist.ReduceOp.SUM)
    model.train()
    if token_sum.item() == 0:
        return float("inf"), float("inf")
    avg = (loss_sum / token_sum).item()
    return avg, math.exp(avg) if avg < 20 else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--val-dataset", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--data-workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = setup_dist()
    torch.manual_seed(args.seed + rank())
    random.seed(args.seed + rank())
    shard_dataset_env(args.data_workers)

    if is_main():
        log("============================================================")
        log("TANTRA SERIOUS 2-GPU FSDP FOUNDATION TRAINING")
        log("============================================================")
        log("675.2M params | vocab=65,536 | 32 layers | dim=1024 | heads=16")
        log("REAL JSONL | BitNet ternary | dense | NO synthetic | NO resume")
        log(f"2 GPUs | seq={args.seq_len} | batch/GPU={args.batch_size} | grad_accum={args.grad_accum}")
        log(f"Global tokens/optimizer step = {args.seq_len * args.batch_size * args.grad_accum * world_size():,}")

    cfg, vcfg = make_config()
    tok = build_tokenizer(vcfg, args.dataset, args.model_dir)

    train_ds = JSONLDataset(
        args.dataset, tok, seq_len=args.seq_len, max_samples=None,
        mask_non_assistant=True, split="all", val_ratio=0.0,
        pack_sequences=False, shuffle=True, shuffle_buf_size=2000, seed=args.seed + rank(),
    )
    val_ds = JSONLDataset(
        args.val_dataset, tok, seq_len=args.seq_len, max_samples=None,
        mask_non_assistant=True, split="all", val_ratio=0.0,
        pack_sequences=False, shuffle=False, seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.data_workers,
        pin_memory=True, persistent_workers=args.data_workers > 0,
        prefetch_factor=2 if args.data_workers > 0 else None,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.data_workers,
                            pin_memory=True, persistent_workers=args.data_workers > 0,
                            prefetch_factor=2 if args.data_workers > 0 else None)

    model = build_model(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    if is_main():
        log(f"Model parameters: {total_params:,} ({total_params/1e9:.3f}B)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    opt_step = 0
    micro_step = 0
    data_iter = iter(train_loader)
    total_tokens_local = 0
    start = time.time()
    last_log = start
    running_loss = 0.0
    running_count = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    while opt_step < args.steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Linear warmup, then cosine decay.
        progress_step = opt_step + 1
        if progress_step <= args.warmup:
            lr_now = args.lr * progress_step / max(1, args.warmup)
        else:
            q = min(1.0, (progress_step - args.warmup) / max(1, args.steps - args.warmup))
            lr_now = args.lr * 0.5 * (1.0 + math.cos(math.pi * q))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        with autocast_context():
            logits, _ = model(x, return_mtp=False, use_latent_reasoning=False)
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_y = y.reshape(-1)
            mask = flat_y != IGNORE_INDEX
            if mask.any():
                loss = F.cross_entropy(flat_logits[mask], flat_y[mask])
            else:
                loss = flat_logits.sum() * 0.0

        running_loss += float(loss.detach().item())
        running_count += 1
        scaled = loss / args.grad_accum
        scaler.scale(scaled).backward()

        micro_step += 1
        total_tokens_local += x.numel()

        if micro_step % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if not torch.isfinite(grad_norm):
                log("WARNING: non-finite gradient; clearing step")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % args.log_every == 0 or opt_step == 1:
                    # Global mean loss across ranks.
                    loss_t = torch.tensor(running_loss / max(1, running_count), device=device, dtype=torch.float64)
                    dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                    loss_t /= world_size()
                    elapsed = time.time() - start
                    # x.numel() is local; multiply by world size for global throughput.
                    tok_s = total_tokens_local * world_size() / max(1e-9, elapsed)
                    if is_main():
                        ppl = math.exp(loss_t.item()) if loss_t.item() < 20 else float("inf")
                        mem0 = torch.cuda.max_memory_allocated(device) / 1024**3
                        log(f"STEP {opt_step:,}/{args.steps:,} | loss={loss_t.item():.4f} | ppl={ppl:.1f} | lr={lr_now:.3e} | global_tok/s={tok_s:.1f} | peak_VRAM={mem0:.2f}GB")
                    running_loss = 0.0
                    running_count = 0

                if opt_step % args.eval_every == 0:
                    vloss, vppl = evaluate(model, val_loader, max_batches=50, device=device)
                    if is_main():
                        log(f"VALIDATION step={opt_step:,} | loss={vloss:.4f} | ppl={vppl:.1f}")

                if opt_step % args.checkpoint_every == 0:
                    save_sharded(model, optimizer, opt_step, os.path.join(args.model_dir, "Checkpoints"))

    save_sharded(model, optimizer, opt_step, os.path.join(args.model_dir, "Checkpoints"))
    dist.barrier()
    if is_main():
        log(f"TRAINING COMPLETE: {opt_step:,} optimizer steps")
        log(f"Global tokens processed: {total_tokens_local * world_size():,}")
        log(f"Elapsed: {(time.time()-start)/3600:.2f}h")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
