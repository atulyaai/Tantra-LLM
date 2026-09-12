"""Tantra 0.675B robust 2x-T4 FSDP trainer.

Goals:
- 675,226,178 parameter model, 65,536 vocab, 32x1024, 16 heads.
- Real JSONL SFT only.
- Two visible T4s with explicit rank/device binding.
- Grouped FSDP: 8 groups x 4 NeuroCoreBlocks.
- Correct mask/state forwarding (no Sequential API mismatch).
- Activation checkpointing disabled for speed; memory is controlled by grouped FULL_SHARD.
- Persistent tokenizer; never rebuild when present.
- Explicit safety checks before expensive training.
- Per-optimizer-step timing (not misleading cumulative timing).
- FP32 cross-entropy for stability with 65K vocabulary.
- No synthetic data, DPO, auto-pilot, resume, MTP, or latent reasoning.
"""
from __future__ import annotations
import argparse, math, os, random, time
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from Tantra.config import NeuroCoreConfig, VocabConfig
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.model import NeuroCoreModel
from Tantra.dataset import JSONLDataset, extract_corpus_sample

IGNORE_INDEX = -100
EXPECTED_PARAMS = 675_226_178
EXPECTED_LAYERS = 32
EXPECTED_GROUPS = 8
BLOCKS_PER_GROUP = 4


def R(): return int(os.environ.get("RANK", "0"))
def LR(): return int(os.environ.get("LOCAL_RANK", "0"))
def WS(): return int(os.environ.get("WORLD_SIZE", "1"))
def main_rank(): return R() == 0

def log(msg):
    print(f"[FSDP rank={R()} local={LR()}] {msg}", flush=True)


def setup():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if WS() != 2:
        raise RuntimeError(f"Expected WORLD_SIZE=2, got {WS()}")
    n = torch.cuda.device_count()
    local = LR()
    if n < 2:
        raise RuntimeError(
            f"Worker sees {n} CUDA device(s); CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    if not 0 <= local < n:
        raise RuntimeError(f"LOCAL_RANK={local}, visible GPUs={n}")
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    log(f"CUDA devices visible={n} | selected device={local} | {torch.cuda.get_device_name(local)}")
    return device


def make_cfg():
    v = VocabConfig(vocab_size=65536, byte_bpe_vocab=65536)
    c = NeuroCoreConfig(vocab=v)
    c.block.num_layers = 32
    c.block.alra.dim = 1024
    c.block.alra.num_heads = 16
    c.block.alra.head_dim = 64
    c.block.sgp.dim = 1024
    c.moe.num_experts = 1
    c.moe.real_top1 = False
    c.bitnet.enabled = True
    c.bitnet.quantize_mode = "ternary"
    c.bitnet.use_shadow_weights = True
    return c, v


def get_tokenizer(vcfg, dataset, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "tokenizer_65536.json")
    if main_rank() and not os.path.isfile(path):
        corpus = os.path.join(model_dir, "tokenizer_corpus.txt")
        log("TOKENIZER: building once from all training records")
        extract_corpus_sample(dataset, corpus, max_lines=None)
        b = ByteBPETokenizer(vcfg)
        b.train([corpus], vocab_size=65536,
                special_tokens=list(vcfg.special_tokens.keys()))
        if b.vocab_size != 65536:
            raise RuntimeError(f"Tokenizer produced {b.vocab_size}; expected 65536")
        b.save(path)
        log(f"TOKENIZER CREATED: {path} | vocab=65,536")
    dist.barrier()
    if not os.path.isfile(path):
        raise RuntimeError(f"Tokenizer missing after barrier: {path}")
    b = ByteBPETokenizer.load(path, vcfg)
    if b.vocab_size != 65536:
        raise RuntimeError(f"Tokenizer mismatch: {b.vocab_size}")
    if main_rank():
        log("TOKENIZER VERIFIED: 65,536 (reused; no rebuild)")
    return UnifiedTokenizer(vcfg, b, MegabytePatcher())


class RankShard(IterableDataset):
    def __init__(self, base):
        self.base = base

    def __iter__(self):
        for i, item in enumerate(iter(self.base)):
            if i % WS() == R():
                yield item


class FSDPBlockGroup(nn.Module):
    """A group of NeuroCoreBlocks with the exact API expected by NeuroCoreModel."""
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, mask=None, state=None):
        new_states = [] if state is not None else None
        for i, block in enumerate(self.blocks):
            block_state = None
            if state is not None:
                if isinstance(state, (list, tuple)):
                    if i < len(state):
                        block_state = state[i]
                elif isinstance(state, dict) and i == 0:
                    block_state = state
            x, ns = block(x, mask=mask, state=block_state)
            if new_states is not None:
                new_states.append(ns)
        return x, new_states


def build_model(cfg, device):
    model = NeuroCoreModel(cfg).to(device)

    actual = sum(p.numel() for p in model.parameters())
    if actual != EXPECTED_PARAMS:
        raise RuntimeError(
            f"MODEL PARAMETER MISMATCH: expected {EXPECTED_PARAMS:,}, got {actual:,}. "
            "Refusing to train the wrong architecture."
        )
    if len(model.layers) != EXPECTED_LAYERS:
        raise RuntimeError(f"Expected {EXPECTED_LAYERS} layers, got {len(model.layers)}")

    mp = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    original_layers = list(model.layers)
    grouped = nn.ModuleList()
    for g in range(EXPECTED_GROUPS):
        start = g * BLOCKS_PER_GROUP
        group = FSDPBlockGroup(original_layers[start:start + BLOCKS_PER_GROUP])
        wrapped = FSDP(
            group,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            device_id=device,
            use_orig_params=True,
        )
        grouped.append(wrapped)
    model.layers = grouped

    wrapped_root = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=device,
        use_orig_params=True,
    )
    return wrapped_root


def sanity_forward(model, device, seq_len):
    """One tiny real forward/backward before the timed loop catches wrapper/API bugs."""
    model.train()
    x = torch.randint(0, 65536, (1, min(seq_len, 8)), device=device, dtype=torch.long)
    y = torch.randint(0, 65536, (1, min(seq_len, 8)), device=device, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.float16):
        z, _ = model(x, return_mtp=False, use_latent_reasoning=False)
    loss = F.cross_entropy(z.reshape(-1, z.size(-1)).float(), y.reshape(-1))
    loss.backward()
    model.zero_grad(set_to_none=True)
    dist.barrier()
    if main_rank():
        log(f"SANITY FORWARD/BACKWARD OK | loss={loss.item():.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--val-dataset", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=.1)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--data-workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    device = setup()
    seed = a.seed + R()
    torch.manual_seed(seed)
    random.seed(seed)
    cfg, vcfg = make_cfg()

    if main_rank():
        log("=" * 60)
        log("TANTRA 0.675B — ROBUST 2-GPU FSDP v7")
        log("=" * 60)
        log("675,226,178 params | vocab=65,536 | 32x1024 | 16 heads | dense | BitNet ternary")
        log("8 FSDP groups x 4 blocks | NO activation checkpointing | REAL JSONL")
        log("NO synthetic | NO DPO | NO auto-pilot | NO resume | NO MTP | NO latent")

    tok = get_tokenizer(vcfg, a.dataset, a.model_dir)

    train_ds = JSONLDataset(
        a.dataset, tok,
        seq_len=a.seq_len,
        max_samples=None,
        mask_non_assistant=True,
        split="all",
        val_ratio=0.0,
        pack_sequences=False,
        shuffle=True,
        shuffle_buf_size=2000,
        seed=a.seed,
    )
    train_loader = DataLoader(
        RankShard(train_ds),
        batch_size=a.batch_size,
        num_workers=a.data_workers,
        pin_memory=True,
        persistent_workers=a.data_workers > 0,
        prefetch_factor=2 if a.data_workers > 0 else None,
    )

    model = build_model(cfg, device)
    visible_params = sum(p.numel() for p in model.parameters())
    if main_rank():
        log(f"Model parameters visible in this rank: {visible_params:,}")
    sanity_forward(model, device, a.seq_len)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=a.lr,
        weight_decay=a.weight_decay,
        betas=(.9, .95),
        eps=1e-8,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    it = iter(train_loader)
    micro = 0
    step = 0
    total_tokens = 0
    interval_tokens = 0
    interval_start = time.perf_counter()
    run_loss = 0.0
    run_n = 0
    opt.zero_grad(set_to_none=True)
    model.train()

    while step < a.steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_tokens = x.numel()
        total_tokens += x_tokens
        interval_tokens += x_tokens

        # Optimizer-step based warmup, held constant over all micro-batches.
        next_step = step + 1
        if next_step <= a.warmup:
            lr = a.lr * next_step / max(1, a.warmup)
        else:
            progress = min(1.0, (next_step - a.warmup) / max(1, a.steps - a.warmup))
            lr = a.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = lr

        with torch.autocast("cuda", dtype=torch.float16):
            z, _ = model(x, return_mtp=False, use_latent_reasoning=False)
        z = z.reshape(-1, z.size(-1))
        yy = y.reshape(-1)
        valid = yy != IGNORE_INDEX
        if valid.any():
            # FP32 CE is safer for a 65K-class output head on T4 FP16.
            loss = F.cross_entropy(z[valid].float(), yy[valid])
        else:
            loss = z.sum() * 0.0

        run_loss += loss.detach().item()
        run_n += 1
        scaler.scale(loss / a.grad_accum).backward()
        micro += 1

        if micro % a.grad_accum == 0:
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient at optimizer step {step + 1}: {grad_norm}")
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % a.log_every == 0 or step == 1:
                torch.cuda.synchronize(device)
                now = time.perf_counter()
                elapsed = max(now - interval_start, 1e-9)
                speed = interval_tokens * WS() / elapsed
                lt = torch.tensor(run_loss / max(1, run_n), device=device, dtype=torch.float64)
                dist.all_reduce(lt, op=dist.ReduceOp.SUM)
                lt /= WS()
                peak = torch.cuda.max_memory_allocated(device) / 1024**3
                if main_rank():
                    ppl = math.exp(lt.item()) if lt.item() < 20 else float("inf")
                    log(
                        f"STEP {step:,}/{a.steps:,} | loss={lt.item():.4f} | "
                        f"ppl={ppl:.1f} | lr={lr:.3e} | GLOBAL tok/s={speed:.1f} | "
                        f"peak VRAM={peak:.2f}GB"
                    )
                interval_tokens = 0
                interval_start = time.perf_counter()
                run_loss = 0.0
                run_n = 0

    torch.cuda.synchronize(device)
    if main_rank():
        log(f"COMPLETE | optimizer steps={step:,} | global tokens={total_tokens * WS():,}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
