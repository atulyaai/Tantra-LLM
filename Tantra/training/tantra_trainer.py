"""
Tantra training pipeline (CPU)
==============================
Stage 0  tokenizer       train a small BPE on the expert data
Stage 1  base            train the shared base on ALL expert data (no experts active)
Stage 2  experts         for each expert: freeze everything, train only its LoRA on its own data
Stage 3  router          train the router to map a user message -> expert
Stage 4  grow (optional) grow_depth() and continue base training

Experts are independent: training `greeting` never touches `identity_self`,
so you can (re)train one expert at a time and the rest keep working.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from Tantra.config import ModelConfig, get_preset
from Tantra.data.expert_dataset import (
    EncodedDataset, find_expert_files, first_user_text, read_jsonl, row_texts,
)
from Tantra.model.tantra_model import TantraModel, save_expert
from Tantra.tokenizer import TantraTokenizer

log = logging.getLogger("Tantra.train")


def setup_cpu(threads: Optional[int] = None) -> None:
    n = threads or max(1, (os.cpu_count() or 2) // 2 * 2)
    torch.set_num_threads(n)


@dataclass
class TrainArgs:
    steps: int = 600
    batch_size: int = 16
    lr: float = 2e-3
    warmup: int = 50
    weight_decay: float = 0.05
    max_len: int = 192
    max_minutes: float = 0.0          # 0 = no time limit
    log_every: int = 50
    seed: int = 0
    gradient_accumulation: int = 1     # effective batch = batch_size * gradient_accumulation


def _lr(step: int, a: TrainArgs) -> float:
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def _loop(model: TantraModel, params: List[torch.nn.Parameter], data: EncodedDataset,
          a: TrainArgs, tag: str, save_fn=None) -> float:
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))
    rng = random.Random(a.seed)
    model.train()
    step, t0, ema = 0, time.time(), None
    grad_accum = max(1, a.gradient_accumulation)
    micro_step = 0
    opt.zero_grad(set_to_none=True)
    while step < a.steps:
        for ids, labels, _ in data.batches(a.batch_size, rng=rng):
            for g in opt.param_groups:
                g["lr"] = _lr(step, a)
            loss = model.loss(ids, labels) / grad_accum
            loss.backward()
            micro_step += 1
            if micro_step == grad_accum:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                micro_step = 0
                ema = loss.item() * grad_accum if ema is None else 0.95 * ema + 0.05 * loss.item() * grad_accum
                step += 1
                if step % a.log_every == 0 or step == a.steps:
                    log.info(f"[{tag}] step {step}/{a.steps} loss {ema:.3f} ({time.time() - t0:.0f}s)")
                if save_fn and step % a.log_every == 0:
                    save_fn()
                if step >= a.steps or (a.max_minutes and time.time() - t0 > a.max_minutes * 60):
                    return ema
    return ema


@torch.no_grad()
def eval_loss(model: TantraModel, data: EncodedDataset, batch_size: int = 32, max_batches: int = 10) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i, (ids, labels, _) in enumerate(data.batches(batch_size, shuffle=False)):
        if i >= max_batches:
            break
        tot += model.loss(ids, labels).item()
        n += 1
    return tot / max(1, n)


class TantraTrainer:
    def __init__(self, model_dir: str | Path = "Model", data_dirs: Sequence[str | Path] = ("Datasets/curated",),
                 general_dir: str | Path = "Datasets/general"):
        self.model_dir = Path(model_dir)
        self.data_dirs = [Path(d) for d in data_dirs]
        self.general_dir = Path(general_dir)
        self.feedback_dir = Path("Datasets/feedback")
        self.tok_path = self.model_dir / "tokenizer.json"
        self.base_path = self.model_dir / "base.pt"
        self.experts_dir = self.model_dir / "experts"

    # ── helpers ──
    def expert_files(self) -> Dict[str, Path]:
        return find_expert_files(self.data_dirs)

    def general_files(self) -> List[Path]:
        """Plain-text corpus for base training (broadens the base; chat data keeps it fluent)."""
        if not self.general_dir.exists():
            return []
        return (sorted(self.general_dir.glob("*.jsonl")) + sorted(self.general_dir.glob("*/*.jsonl"))
                + sorted(self.general_dir.glob("*.txt")))

    def general_rows(self, limit: int = 120_000, seed: int = 0) -> List[Dict]:
        """General English for the base. .txt files are split on <|endoftext|>
        (TinyStories format); .jsonl rows are {"text": ...}. Rows are spread
        evenly across files up to `limit`."""
        files = self.general_files()
        if not files:
            return []
        per_file = max(1, limit // len(files))
        rows: List[Dict] = []
        for f in files:
            if f.suffix == ".txt":
                got = 0
                buf: List[str] = []
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        if "<|endoftext|>" in line:
                            text = " ".join(buf).strip()
                            buf = []
                            if len(text) > 40:
                                rows.append({"text": text})
                                got += 1
                                if got >= per_file:
                                    break
                        elif line.strip():
                            buf.append(line.strip())
            else:
                rows += read_jsonl(f, limit=per_file)
        random.Random(seed).shuffle(rows)
        return rows

    def expert_rows(self, name: str, limit: Optional[int] = None, feedback_weight: int = 3) -> List[Dict]:
        """Curated rows + rated chat feedback (Datasets/feedback/<name>.jsonl),
        feedback repeated `feedback_weight`x because it is real usage."""
        files = self.expert_files()
        rows = read_jsonl(files[name], limit=limit) if name in files else []
        for fb in (self.feedback_dir / f"{name}.jsonl",            # your /good and /fix
                   self.feedback_dir / "auto" / f"{name}.jsonl"):  # learned by the reasoner
            if fb.exists():
                rows += read_jsonl(fb) * feedback_weight
        return rows

    def tokenizer(self) -> TantraTokenizer:
        return TantraTokenizer.load(self.tok_path)

    def load_model(self) -> TantraModel:
        return TantraModel.load(self.base_path, self.experts_dir)

    def save_model(self, model: TantraModel) -> None:
        model.save(self.base_path)

    def snapshot(self, tag: str) -> Path:
        """Copy the full model state (tokenizer + base/router + experts) to
        checkpoints/<tag>/ so a working version can always be restored."""
        import shutil
        dst = Path("checkpoints") / tag
        dst.mkdir(parents=True, exist_ok=True)
        for src in (self.tok_path, self.base_path):
            if src.exists():
                shutil.copy2(src, dst / src.name)
        if self.experts_dir.exists():
            shutil.copytree(self.experts_dir, dst / "experts", dirs_exist_ok=True)
        log.info(f"checkpoint saved -> {dst}")
        return dst

    # ── stage 0 ──
    def train_tokenizer(self, vocab_size: int = 8192) -> TantraTokenizer:
        def texts():
            for row in self.general_rows(limit=60_000):
                yield from row_texts(row)
            for f in self.expert_files().values():
                for row in read_jsonl(f):
                    yield from row_texts(row)
        tok = TantraTokenizer.train(texts(), vocab_size=vocab_size)
        tok.save(self.tok_path)
        log.info(f"tokenizer: vocab {tok.vocab_size} -> {self.tok_path}")
        return tok

    # ── stage 1 ──
    def train_base(self, preset: str | ModelConfig = "tiny", a: Optional[TrainArgs] = None,
                   experts: Optional[Sequence[str]] = None, resume: bool = True,
                   chat_fraction: float = 0.15, general_limit: int = 120_000) -> TantraModel:
        a = a or TrainArgs()
        tok = self.tokenizer()
        if resume and self.base_path.exists():
            model = self.load_model()
            log.info("resuming base from checkpoint")
        else:
            cfg = get_preset(preset) if isinstance(preset, str) else preset
            cfg.vocab_size = tok.vocab_size
            model = TantraModel(cfg)
        files = self.expert_files()
        rows = self.general_rows(limit=general_limit)
        n_gen = len(rows)
        # Base = general English + a SMALL slice of conversation data, only so the
        # base learns the chat format (<|user|> ... <|assistant|> ...). The rest of
        # each expert's data is reserved for that expert. Knowledge data
        # (gk_*) is never shown to the base: those experts must add the knowledge.
        rng = random.Random(a.seed)
        n_chat = 0
        for name, f in files.items():
            if name.startswith("gk_") or (experts is not None and name not in experts):
                continue
            chat = read_jsonl(f)
            rng.shuffle(chat)
            take = chat[:int(len(chat) * chat_fraction)]
            rows += take
            n_chat += len(take)
        rng.shuffle(rows)
        src = f"{n_gen} general texts + {n_chat} chat rows ({chat_fraction:.0%} of conversation data)"
        data = EncodedDataset(tok, rows, a.max_len, assistant_only=False)
        log.info(f"base: {model.num_params() / 1e6:.2f}M params, {len(data)} samples, {src}")
        model.set_active([])
        for p in model.parameters():
            p.requires_grad = False
        params = list(model.base_parameters())
        for p in params:
            p.requires_grad = True
        _loop(model, params, data, a, "base", save_fn=lambda: self.save_model(model))
        self.save_model(model)
        return model

    # ── stage 2 ──
    def train_expert(self, name: str, model: Optional[TantraModel] = None, a: Optional[TrainArgs] = None,
                     rank: Optional[int] = None) -> Dict[str, float]:
        a = a or TrainArgs(steps=300, lr=3e-3, warmup=20)
        model = model or self.load_model()
        tok = self.tokenizer()
        files = self.expert_files()
        rows = self.expert_rows(name)
        if not rows:
            raise FileNotFoundError(f"no data for expert {name!r} in {self.data_dirs} or {self.feedback_dir}")
        # Train on the expert's full data (minus a held-out slice for eval).
        # Training on a small slice makes the LoRA overfit and WORSEN held-out
        # loss; the full set keeps it stable (matches the proven v1 behaviour).
        random.Random(a.seed).shuffle(rows)
        n_eval = max(1, len(rows) // 10)
        train = EncodedDataset(tok, rows[n_eval:], a.max_len)
        held = EncodedDataset(tok, rows[:n_eval], a.max_len)

        new = name not in model.expert_ranks
        model.add_expert(name, rank)
        model.set_active([])
        before = eval_loss(model, held)

        for p in model.parameters():
            p.requires_grad = False
        params = model.expert_parameters(name)
        for p in params:
            p.requires_grad = True
        model.set_active([(name, 1.0)])
        _loop(model, params, train, a, name)
        after = eval_loss(model, held)
        model.set_active([])

        save_expert(model, name, self.experts_dir)
        if new:
            self.save_model(model)  # router learned a new row
        n_p = sum(p.numel() for p in params)
        log.info(f"[{name}] {n_p / 1e3:.0f}K params | held-out loss base {before:.3f} -> expert {after:.3f}")
        return {"expert": name, "params": n_p, "loss_base": before, "loss_expert": after}

    # ── stage 3 ──
    def train_ngram_router(self, model: Optional[TantraModel] = None, per_expert: int = 1500) -> float:
        """Word/char n-gram router (more accurate on short, typo-heavy chat).
        Saved to Model/router_ngram.pt; the engine prefers it when present."""
        from Tantra.model.ngram_router import train_ngram_router
        model = model or self.load_model()
        names = model.expert_names
        texts, labels = [], []
        for i, name in enumerate(names):
            for row in self.expert_rows(name, limit=per_expert):
                t = first_user_text(row)
                if t:
                    texts.append(t); labels.append(i)
        # mixed messages ("hi, who made you?") labelled with both experts
        rng = random.Random(1)
        by = {}
        for t, l in zip(texts, labels):
            by.setdefault(l, []).append(t)
        keys = list(by)
        for _ in range(len(texts) // 4):
            a, b = rng.sample(keys, 2)
            mix = rng.choice(by[a]).rstrip("?.!") + rng.choice([", ", " and ", ". "]) + rng.choice(by[b])
            texts += [mix, mix]; labels += [a, b]
        router, acc = train_ngram_router(texts, labels, names)
        router.save(self.model_dir / "router_ngram.pt")
        log.info(f"[ngram router] {len(names)} experts, {len(texts)} examples, held-out acc {acc:.1%}")
        return acc

    def train_router(self, model: Optional[TantraModel] = None, steps: int = 800, lr: float = 3e-3,
                     batch_size: int = 64, per_expert: int = 600) -> float:
        model = model or self.load_model()
        self.train_ngram_router(model)
        tok = self.tokenizer()
        files = self.expert_files()
        X, Y = [], []
        for i, name in enumerate(model.expert_names):
            for row in self.expert_rows(name, limit=per_expert):
                X.append(tok.encode(first_user_text(row))[:64] or [tok.pad_id])
                Y.append(i)
        if not X:
            raise RuntimeError("no router data - train some experts first")
        # Mixed messages: "hi, who made you?" should activate greeting AND identity_self.
        # Concatenate two experts' user texts and label the mix with BOTH experts
        # (same input, two rows, two labels -> softmax splits probability ->
        #  top-1 ~0.5 < cum_prob -> route() activates both).
        rng_mix = random.Random(1)
        by_name = {n: i for i, n in enumerate(model.expert_names) if n in files}
        pool = {n: [tok.encode(first_user_text(r))[:48] or [tok.pad_id]
                    for r in read_jsonl(files[n], limit=200)] for n in by_name}
        names_ok = [n for n in pool if pool[n]]
        for _ in range(len(X) // 3):
            a, b = rng_mix.sample(names_ok, 2)
            mix = (rng_mix.choice(pool[a]) + rng_mix.choice(pool[b]))[:64]
            X.append(mix); Y.append(by_name[a])
            X.append(mix); Y.append(by_name[b])
        for p in model.parameters():
            p.requires_grad = False
        params = list(model.router.parameters())
        for p in params:
            p.requires_grad = True
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        rng = random.Random(0)
        idx = list(range(len(X)))
        model.train()
        acc = 0.0
        for step in range(steps):
            b = rng.sample(idx, min(batch_size, len(idx)))
            L = max(len(X[j]) for j in b)
            ids = torch.full((len(b), L), tok.pad_id)
            mask = torch.zeros((len(b), L))
            for r, j in enumerate(b):
                ids[r, :len(X[j])] = torch.tensor(X[j])
                mask[r, :len(X[j])] = 1
            y = torch.tensor([Y[j] for j in b])
            logits = model.router(model.embed(ids).detach(), mask)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            acc = 0.9 * acc + 0.1 * (logits.argmax(-1) == y).float().mean().item()
            if (step + 1) % 100 == 0:
                log.info(f"[router] step {step + 1}/{steps} loss {loss.item():.3f} acc~{acc:.2f}")
        self.save_model(model)
        return acc

    # ── full stage-1 pipeline ──
    def run(self, experts: Sequence[str], preset: str = "tiny", vocab_size: int = 8192,
            base_args: Optional[TrainArgs] = None, expert_args: Optional[TrainArgs] = None) -> List[Dict]:
        if not self.tok_path.exists():
            self.train_tokenizer(vocab_size)
        model = self.train_base(preset, base_args, experts=experts, resume=False) \
            if not self.base_path.exists() else self.load_model()
        results = [self.train_expert(e, model, expert_args) for e in experts]
        self.train_router(model)
        self.snapshot("stage1_latest")
        return results
