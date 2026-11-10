"""Unified safe training entry point: python -m Tantra.train [run|status|watch]."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import psutil
from Tantra import ROOT


def snapshot(stage):
    stage=Path(stage);base=stage/'base-training'
    def read(path):return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    status=read(stage/'status.json');manifest=read(stage/'corpus-19m'/'manifest.json')
    rows=[]
    if (base/'metrics.jsonl').exists():
        for line in (base/'metrics.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass
    alive=[]
    for lock in (stage/'worker.lock',base/'training.lock'):
        try:
            pid=int(lock.read_text());process=psutil.Process(pid)
            command=' '.join(process.cmdline())
            if 'Tantra.pretrain' in command:alive.append(pid)
        except (OSError,ValueError,psutil.Error):pass
    total=sum(v['tokens'] for k,v in manifest.get('files',{}).items() if k.endswith('-train'))
    validation=next((r['validation'] for r in reversed(rows) if 'validation' in r),{})
    checkpoint=base/'latest.pt'
    return {'stage':status.get('stage','not_started'),'worker_pids':sorted(set(alive)),
        'latest':rows[-1] if rows else {},'validation':validation,'target_tokens':total,
        'checkpoint':str(checkpoint) if checkpoint.exists() else None,
        'checkpoint_age_seconds':time.time()-checkpoint.stat().st_mtime if checkpoint.exists() else None}


def display(s,rate=None):
    row=s['latest'];tokens=row.get('target_tokens',0);target=s['target_tokens']
    print(f"State: {s['stage']} | worker: {s['worker_pids'] or 'STOPPED'}",flush=True)
    print(f"Update {row.get('update',0)}/{math.ceil(target/4096) if target else '?'} | train loss {row.get('loss','-')} | tokens {tokens:,}/{target:,}",flush=True)
    if rate and rate>0:print(f"Speed {rate:.1f} tokens/s | ETA {max(0,target-tokens)/rate/3600:.2f} hours",flush=True)
    for lang,value in s['validation'].items():print(f"Validation {lang}: {value['loss']:.4f} ({value['windows']} windows)",flush=True)
    print(f"Checkpoint: {s['checkpoint'] or 'not saved yet'} | age: {round(s['checkpoint_age_seconds']) if s['checkpoint_age_seconds'] is not None else '-'} seconds",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',nargs='?',choices=['run','status','watch','stop'],default='run')
    p.add_argument('--stage',type=Path,default=ROOT/'runs/pretraining-20m-v1');a=p.parse_args()
    s=snapshot(a.stage)
    if a.command=='stop':
        if not s['worker_pids']:p.error('No verified live worker for this stage')
        (a.stage/'base-training'/'stop.request').write_text('Stop after current optimizer update; save checkpoint.\n')
        print('Clean stop requested; wait for worker to exit before resume.');return
    if a.command=='run' and not s['worker_pids']:
        if (a.stage/'worker.lock').exists():p.error('Stale worker lock: verify its PID and archive the lock before resuming.')
        from Tantra.pretrain import continue_cli as train
        import sys
        sys.argv=[sys.argv[0],'--output',str(a.stage),'--pilot',str(ROOT/'runs/vocabulary-64k-v1'),'--initial-model',str(ROOT/'runs/model-64k-initial-v1')]
        train();return
    if a.command=='status':display(s);return
    print('Attached to existing training; Ctrl+C closes this viewer without stopping its worker.',flush=True)
    start=time.monotonic();initial=s['latest'].get('target_tokens',0);previous=None
    try:
        while True:
            s=snapshot(a.stage);signature=(s['latest'].get('update'),bool(s['worker_pids']),s['stage'])
            if signature!=previous:
                elapsed=time.monotonic()-start;delta=s['latest'].get('target_tokens',0)-initial
                display(s,delta/elapsed if elapsed and delta>0 else None);previous=signature
            if not s['worker_pids']:break
            time.sleep(2)
    except KeyboardInterrupt:pass


# ──  training/tantra_trainer.py  ──
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

from Tantra import ModelConfig, get_preset
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
    if not len(data):
        raise ValueError('No trainable samples; check data and truncation')
    if min(a.steps,a.batch_size,a.log_every,a.gradient_accumulation)<1:
        raise ValueError('Training steps, batch size, logging and accumulation must be positive')
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
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                raise FloatingPointError('Non-finite training loss')
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
        from Tantra.main import train_ngram_router
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
        for _ in range(len(texts) // 4 if len(keys)>=2 else 0):
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
        for _ in range(len(X) // 3 if len(names_ok)>=2 else 0):
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



# ──  training/checkpoints.py  ──
"""Integrity-checked resumable checkpoints with three bounded generations.

Only corruption permits fallback. Identity/configuration mismatches fail closed.
Model files are trusted local training artifacts (PyTorch pickle), not uploads.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Mapping

import torch


class CheckpointCorruptError(RuntimeError):
    """No intact compatible resumable generation could be read."""


class CheckpointIdentityError(ValueError):
    """A readable checkpoint belongs to different data, model or settings."""


def _digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as reader:
        for block in iter(lambda:reader.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _sidecar(path):
    return Path(str(path)+'.integrity.json')


def generations(path):
    path=Path(path)
    return [path,path.with_name(path.stem+'.previous-1'+path.suffix),
            path.with_name(path.stem+'.previous-2'+path.suffix)]


def _write_integrity(path, digest, size):
    sidecar=_sidecar(path); temporary=sidecar.with_name(sidecar.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with temporary.open('w',encoding='utf-8') as writer:
            json.dump({'version':1,'sha256':digest,'bytes':size},writer,sort_keys=True)
            writer.flush();os.fsync(writer.fileno())
        os.replace(temporary,sidecar)
    finally:
        if temporary.exists():temporary.unlink()


def _copy_generation(source,target):
    """Publish the copied bytes before their checksum; interrupted pairs are rejected."""
    temporary=target.with_name(target.name+'.tmp-'+uuid.uuid4().hex)
    try:
        try:os.link(source,temporary)
        except OSError:
            with source.open('rb') as reader,temporary.open('wb') as writer:
                shutil.copyfileobj(reader,writer,1024*1024);writer.flush();os.fsync(writer.fileno())
        # Retain the recorded checksum rather than blessing a corrupted source anew.
        integrity=_sidecar(source)
        if integrity.exists():
            try:
                recorded=json.loads(integrity.read_text(encoding='utf-8'))
                digest,size=recorded['sha256'],recorded['bytes']
            except (OSError,ValueError,KeyError,TypeError):
                digest,size='unreadable-source-integrity',temporary.stat().st_size
        else:digest,size=_digest(temporary),temporary.stat().st_size
        os.replace(temporary,target);_write_integrity(target,digest,size)
    finally:
        if temporary.exists():temporary.unlink()


def save_resumable(path,payload):
    """Save latest plus two previous generations; preserve best checkpoints separately."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        # Do all expensive serialization before rotating anything.
        with temporary.open('wb') as writer:
            torch.save(payload,writer);writer.flush();os.fsync(writer.fileno())
        digest,size=_digest(temporary),temporary.stat().st_size
        latest,previous1,previous2=generations(path)
        if previous1.exists():_copy_generation(previous1,previous2)
        if latest.exists():_copy_generation(latest,previous1)
        os.replace(temporary,path);_write_integrity(path,digest,size)
    finally:
        if temporary.exists():temporary.unlink()


def _finite(value):
    if isinstance(value,torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
    if isinstance(value,Mapping):return all(_finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(_finite(v) for v in value)
    return True


def load_resumable(path,*,identity_key,expected_identity,expected_config,required,
                   expected_fields=None,state_shapes=None):
    """Return (payload, recovery evidence), trying at most three generations.

    Missing checksums are accepted only for existing legacy checkpoints, and are
    disclosed in recovery evidence. Checksummed generations reject any changed
    byte before unpickling. A readable identity/config mismatch is never fallback.
    """
    rejected=[]
    for candidate in generations(path):
        if not candidate.exists():continue
        integrity=_sidecar(candidate);verified=False
        try:
            if integrity.exists():
                metadata=json.loads(integrity.read_text(encoding='utf-8'))
                if metadata.get('version')!=1 or metadata.get('bytes')!=candidate.stat().st_size or metadata.get('sha256')!=_digest(candidate):
                    raise CheckpointCorruptError('Integrity checksum/length mismatch')
                verified=True
            payload=torch.load(candidate,map_location='cpu',weights_only=False)
            if not isinstance(payload,dict):raise CheckpointCorruptError('Checkpoint is not a dictionary')
            if identity_key not in payload or 'config' not in payload:
                raise CheckpointCorruptError('Missing checkpoint identity/configuration')
            if payload[identity_key]!=expected_identity:raise CheckpointIdentityError('Resume identity mismatch: '+str(candidate))
            if payload['config']!=expected_config:raise CheckpointIdentityError('Resume model configuration mismatch: '+str(candidate))
            for key,value in (expected_fields or {}).items():
                if key not in payload:raise CheckpointCorruptError('Missing checkpoint field: '+key)
                if payload[key]!=value:raise CheckpointIdentityError('Resume setting mismatch ('+key+'): '+str(candidate))
            missing=set(required)-set(payload)
            if missing:raise CheckpointCorruptError('Missing resumable fields: '+', '.join(sorted(missing)))
            if not isinstance(payload.get('state'),dict) or not isinstance(payload.get('optimizer'),dict):
                raise CheckpointCorruptError('Invalid model/optimizer state')
            if state_shapes is not None:
                saved=payload['state']
                if set(saved)!=set(state_shapes) or any(not isinstance(saved[k],torch.Tensor) or saved[k].shape!=value.shape for k,value in state_shapes.items()):
                    raise CheckpointIdentityError('Resume model state layout mismatch: '+str(candidate))
            if any(not isinstance(payload[key],int) or isinstance(payload[key],bool) or payload[key]<0 for key in ('cursor','updates','tokens')):
                raise CheckpointCorruptError('Invalid checkpoint progress counters')
            if not isinstance(payload['torch_rng'],torch.Tensor) or payload['torch_rng'].dtype!=torch.uint8:
                raise CheckpointCorruptError('Invalid Torch RNG state')
            if not _finite(payload['state']) or not _finite(payload['optimizer']):
                raise CheckpointCorruptError('Non-finite model/optimizer tensors')
            return payload,{'loaded_path':str(candidate),'fallback_used':candidate!=Path(path),
                            'integrity_verified':verified,'rejected':rejected,
                            'legacy_without_checksum':not verified}
        except CheckpointIdentityError:raise
        except (OSError,ValueError,RuntimeError,EOFError,KeyError,TypeError,IndexError) as exc:
            rejected.append({'path':str(candidate),'error':type(exc).__name__+': '+str(exc)})
        except Exception as exc:
            # PyTorch/pickle corruption has several exception types; do not hide it.
            import pickle
            if not isinstance(exc,pickle.UnpicklingError):raise
            rejected.append({'path':str(candidate),'error':type(exc).__name__+': '+str(exc)})
    raise CheckpointCorruptError('No intact resumable checkpoint among three generations: '+json.dumps(rejected))



# ──  model/tantra_model.py  ──
"""
Tantra Sparse Mixture-of-Adapters
=================================

  shared base transformer  (RoPE, GQA, SwiGLU, RMSNorm, tied embeddings)
+ N tiny LoRA experts       (each trained alone on its own data)
+ a cheap router            (picks top-k experts per request)

Only the selected experts run, so adding expert #1000 costs disk space,
not CPU time. Each expert is a separate small file, so you can train /
replace one expert without touching the others.

Growth: `grow_depth()` inserts new layers whose output projections are
zero, so the grown model computes exactly the same function and training
continues from there (tiny -> small -> ... -> 1B on the same laptop).
"""


import copy
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from Tantra import ModelConfig


# ── building blocks ───────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class LoRALinear(nn.Module):
    """Linear layer with any number of named LoRA experts attached.

    Which experts are active (and their weights) is set by the parent model
    via `self.active`, a list of (name, weight) pairs. Inactive experts cost
    nothing.
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = False):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=bias)
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.scale: Dict[str, float] = {}
        self.active: List[Tuple[str, float]] = []

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    def _device(self) -> torch.device:
        # After dynamic INT8 quantisation `base` has no parameters (weight is packed); experts stay on CPU.
        p = next(self.base.parameters(), None)
        return p.device if p is not None else torch.device("cpu")

    def add_expert(self, name: str, rank: int, alpha: float) -> None:
        d_in, d_out = self.base.in_features, self.base.out_features
        A = torch.empty(rank, d_in, device=self._device())
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        self.lora_A[name] = nn.Parameter(A)
        self.lora_B[name] = nn.Parameter(torch.zeros(d_out, rank, device=self._device()))
        self.scale[name] = alpha / rank

    def remove_expert(self, name: str) -> None:
        for d in (self.lora_A, self.lora_B):
            if name in d:
                del d[name]
        self.scale.pop(name, None)

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        for name, w in self.active:
            if name in self.lora_A:
                y = y + (x @ self.lora_A[name].t() @ self.lora_B[name].t()) * (self.scale[name] * w)
        return y


def rope_cache(head_dim: int, max_len: int, theta: float) -> Tuple[Tensor, Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [B, H, T, Dh]; cos/sin: [T, Dh/2]
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv = cfg.n_heads, cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        kv_dim = self.head_dim * self.n_kv
        self.q = LoRALinear(cfg.d_model, cfg.d_model)
        self.k = LoRALinear(cfg.d_model, kv_dim)
        self.v = LoRALinear(cfg.d_model, kv_dim)
        self.o = LoRALinear(cfg.d_model, cfg.d_model)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: Optional[Dict] = None) -> Tensor:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # causal only needed when processing more than one new token
        if T > 1 and k.shape[2] != T:  # prefill with existing cache: explicit mask
            S = k.shape[2]
            mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(diagonal=S - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = LoRALinear(cfg.d_model, cfg.mlp_hidden)
        self.up = LoRALinear(cfg.d_model, cfg.mlp_hidden)
        self.down = LoRALinear(cfg.mlp_hidden, cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        return x + self.mlp(self.norm2(x))

    def lora_modules(self) -> Dict[str, LoRALinear]:
        a, m = self.attn, self.mlp
        return {"q": a.q, "k": a.k, "v": a.v, "o": a.o, "gate": m.gate, "up": m.up, "down": m.down}


# ── router ────────────────────────────────────────────────────────────────────

class Router(nn.Module):
    """Bag-of-embeddings classifier over the prompt -> expert scores.

    Uses the (frozen) token embeddings, so routing costs one tiny matmul and
    runs before the transformer. Growable: adding an expert adds one row.
    """

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)  # starts with 0 outputs; grows one row per expert
        self.head.weight = nn.Parameter(torch.empty(0, hidden))
        self.head.bias = nn.Parameter(torch.empty(0))
        self.head.out_features = 0
        self.names: List[str] = []

    def add_expert(self, name: str) -> None:
        old = self.head
        new = nn.Linear(old.in_features, old.out_features + 1).to(old.weight.device)
        with torch.no_grad():
            new.weight[:-1] = old.weight
            new.bias[:-1] = old.bias
            new.weight[-1].normal_(0, 0.02)
            new.bias[-1] = 0.0
        self.head = new
        self.names.append(name)

    def remove_expert(self, name: str) -> None:
        i = self.names.index(name)
        keep = [j for j in range(len(self.names)) if j != i]
        old = self.head
        new = nn.Linear(old.in_features, len(keep)).to(old.weight.device)
        with torch.no_grad():
            new.weight.copy_(old.weight[keep])
            new.bias.copy_(old.bias[keep])
        self.head = new
        self.names.pop(i)

    def forward(self, emb: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # emb: [B, T, D] token embeddings; mask: [B, T] 1 for real tokens
        if mask is None:
            pooled = emb.mean(1)
        else:
            m = mask.unsqueeze(-1).to(emb.dtype)
            pooled = (emb * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.proj(pooled))


# ── full model ────────────────────────────────────────────────────────────────

class TantraModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.router = Router(cfg.d_model)
        self.expert_ranks: Dict[str, int] = {}
        cos, sin = rope_cache(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for layer in self.layers:  # scaled residual init
            nn.init.normal_(layer.attn.o.base.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(layer.mlp.down.base.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ── parameters ──
    def base_parameters(self) -> Iterable[nn.Parameter]:
        for n, p in self.named_parameters():
            if "lora_" not in n and not n.startswith("router."):
                yield p

    def expert_parameters(self, name: str) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if n.endswith(f"lora_A.{name}") or n.endswith(f"lora_B.{name}")]

    def num_params(self, base_only: bool = True) -> int:
        return sum(p.numel() for p in (self.base_parameters() if base_only else self.parameters()))

    # ── experts ──
    @property
    def expert_names(self) -> List[str]:
        return list(self.router.names)

    def _lora_targets(self):
        for layer in self.layers:
            mods = layer.lora_modules()
            for t in self.cfg.lora_targets:
                yield mods[t]

    def _all_lora(self):
        for layer in self.layers:
            yield from layer.lora_modules().values()

    def add_expert(self, name: str, rank: Optional[int] = None) -> None:
        if name in self.expert_ranks:
            return
        rank = rank or self.cfg.lora_rank
        for mod in self._lora_targets():
            mod.add_expert(name, rank, self.cfg.lora_alpha)
        self.router.add_expert(name)
        self.expert_ranks[name] = rank

    def remove_expert(self, name: str) -> None:
        for mod in self._all_lora():
            mod.remove_expert(name)
        self.router.remove_expert(name)
        self.expert_ranks.pop(name, None)

    def set_active(self, experts: Sequence[Tuple[str, float]]) -> None:
        experts = list(experts)
        for mod in self._all_lora():
            mod.active = experts

    def expert_state(self, name: str) -> Dict[str, Tensor]:
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters()
                if n.endswith(f"lora_A.{name}") or n.endswith(f"lora_B.{name}")}

    def lora_shapes(self, name: str, rank: int) -> Dict[str, Tuple[int, ...]]:
        """Parameter names and shapes an adapter `name` of `rank` must provide for this base."""
        where = {id(m): n for n, m in self.named_modules()}
        shapes = {}
        for mod in self._lora_targets():
            prefix = where[id(mod)]
            shapes[f"{prefix}.lora_A.{name}"] = (rank, mod.base.in_features)
            shapes[f"{prefix}.lora_B.{name}"] = (mod.base.out_features, rank)
        return shapes

    def attach_lora(self, name: str, state: Dict[str, Tensor], rank: int) -> None:
        """Attach only the LoRA weights of an expert already known to the router (lazy loading).
        Refuses any missing, extra or mis-shaped tensor instead of skipping it."""
        expected = self.lora_shapes(name, rank)
        given = {k: tuple(v.shape) for k, v in state.items()}
        if given != expected:
            missing = sorted(set(expected) - set(given)); extra = sorted(set(given) - set(expected))
            wrong = sorted(k for k in set(given) & set(expected) if given[k] != expected[k])
            raise ValueError(f"Adapter '{name}' does not fit this base: missing={missing[:3]} extra={extra[:3]} wrong_shape={wrong[:3]}")
        for mod in self._lora_targets():
            mod.add_expert(name, rank, self.cfg.lora_alpha)
        params = dict(self.named_parameters())
        with torch.no_grad():
            for k, v in state.items():
                params[k].copy_(v)
        self.expert_ranks[name] = rank

    def detach_lora(self, name: str) -> None:
        """Free an expert's LoRA weights but keep its router entry."""
        for mod in self._all_lora():
            mod.remove_expert(name)
            mod.active = [(n, w) for n, w in mod.active if n != name]
        self.expert_ranks.pop(name, None)

    def load_expert_state(self, name: str, state: Dict[str, Tensor], rank: int) -> None:
        self.add_expert(name, rank)
        params = dict(self.named_parameters())
        with torch.no_grad():
            for n, t in state.items():
                if n in params and params[n].shape == t.shape:
                    params[n].copy_(t)

    # ── routing ──
    @torch.no_grad()
    def route(self, input_ids: Tensor, max_experts: int = 4, cum_prob: float = 0.8,
              min_prob: float = 0.08) -> List[Tuple[str, float]]:
        """Adaptive routing: the router decides HOW MANY experts a request needs.

        Experts are added in order of confidence until their combined probability
        reaches `cum_prob` (or `max_experts`). A clear request ("hi") uses 1 expert;
        a mixed one ("hi, who are you?") uses 2-3. Experts below `min_prob` are
        dropped. Returned weights are renormalised to sum to 1.
        """
        if not self.router.names:
            return []
        probs = self.router(self.embed(input_ids)).softmax(-1)[0]
        vals, idx = probs.sort(descending=True)
        chosen, total = [], 0.0
        for v, i in zip(vals.tolist(), idx.tolist()):
            if len(chosen) >= max_experts or total >= cum_prob or (chosen and v < min_prob):
                break
            chosen.append((self.router.names[i], v))
            total += v
        return [(n, w / total) for n, w in chosen]

    # ── forward ──
    def forward(self, input_ids: Tensor, caches: Optional[List[Dict]] = None, pos: int = 0) -> Tensor:
        T = input_ids.shape[1]
        needed = pos + T
        if needed > self.rope_cos.size(0):
            new_max = max(needed, self.rope_cos.size(0) * 2)
            c, s = rope_cache(self.cfg.d_model // self.cfg.n_heads, new_max, self.cfg.rope_theta)
            self.rope_cos = c.to(self.rope_cos.device)
            self.rope_sin = s.to(self.rope_sin.device)
        cos, sin = self.rope_cos[pos:needed], self.rope_sin[pos:needed]
        x = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, None if caches is None else caches[i])
        q_head = self.__dict__.get("_q_head")  # INT8 copy of the tied head, set by quantize_for_cpu
        if q_head is not None:
            return q_head(self.norm(x))
        return F.linear(self.norm(x), self.embed.weight)  # tied head

    def loss(self, input_ids: Tensor, labels: Tensor) -> Tensor:
        logits = self(input_ids)
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                               labels[:, 1:].reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 64, temperature: float = 0.7,
                 top_k: int = 40, top_p: float = 0.9, stop_ids: Sequence[int] = (),
                 repetition_penalty: float = 1.1, no_repeat_ngram: int = 0):
        """Yields token ids one at a time (KV-cached).

        no_repeat_ngram=n forbids any token that would repeat an n-gram already generated in
        this reply (small models otherwise loop: "I'm Tantra, and I'm Tantra")."""
        caches = [{} for _ in self.layers]
        ids = input_ids[:, -(self.cfg.max_seq_len - max_new_tokens):]
        logits = self(ids, caches, 0)[:, -1]
        pos = ids.shape[1]
        seen = ids[0].tolist()
        gen: List[int] = []
        for _ in range(max_new_tokens):
            if no_repeat_ngram and len(gen) >= no_repeat_ngram - 1:
                prefix = tuple(gen[len(gen) - no_repeat_ngram + 1:]) if no_repeat_ngram > 1 else ()
                banned = [gen[i + no_repeat_ngram - 1] for i in range(len(gen) - no_repeat_ngram + 1)
                          if tuple(gen[i:i + no_repeat_ngram - 1]) == prefix]
                if banned:
                    logits[0, banned] = -float("inf")
            if repetition_penalty != 1.0 and seen:
                s = torch.tensor(sorted(set(seen[-64:])), device=logits.device)
                lg = logits[0, s]
                logits[0, s] = torch.where(lg > 0, lg / repetition_penalty, lg * repetition_penalty)
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = logits.topk(min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = -float("inf")
                probs = logits.softmax(-1)
                if top_p < 1.0:
                    sp, si = probs.sort(descending=True)
                    cut = sp.cumsum(-1) - sp > top_p
                    sp[cut] = 0
                    probs = torch.zeros_like(probs).scatter(-1, si, sp)
                nxt = torch.multinomial(probs / probs.sum(-1, keepdim=True), 1)
            tok = int(nxt)
            if tok in stop_ids or pos >= self.cfg.max_seq_len:
                return
            yield tok
            seen.append(tok); gen.append(tok)
            logits = self(nxt, caches, pos)[:, -1]
            pos += 1

    # ── growth ──
    def grow_depth(self, n_new: int, where: str = "interleave") -> None:
        """Add `n_new` layers without changing the model's output.

        New layers are copies of existing ones with zeroed output projections
        (attn.o, mlp.down) and zeroed expert LoRA-B, so each starts as the
        identity on the residual stream. Train afterwards to use them.
        """
        old = list(self.layers)
        n = len(old)
        src = [round(i * n / n_new) for i in range(n_new)] if where == "interleave" else [n - 1] * n_new
        new_layers: List[nn.Module] = []
        for i, layer in enumerate(old):
            new_layers.append(layer)
            for _ in range(src.count(i)):
                blk = copy.deepcopy(layer)
                with torch.no_grad():
                    blk.attn.o.base.weight.zero_()
                    blk.mlp.down.base.weight.zero_()
                    for mod in blk.lora_modules().values():
                        for p in mod.lora_B.values():
                            p.zero_()
                new_layers.append(blk)
        self.layers = nn.ModuleList(new_layers)
        self.cfg.n_layers = len(new_layers)

    # ── io ──
    def save(self, path: str | Path, experts_dir: Optional[str | Path] = None) -> None:
        """Save base (+ router) to `path`; each expert to experts_dir/<name>.pt."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        base_sd = {k: v for k, v in self.state_dict().items() if "lora_" not in k}
        torch.save({"config": self.cfg.to_dict(), "state": base_sd,
                    "router_names": self.router.names, "expert_ranks": self.expert_ranks}, path)
        if experts_dir is not None:
            for name in self.expert_names:
                save_expert(self, name, experts_dir)

    @classmethod
    def load(cls, path: str | Path, experts_dir: Optional[str | Path] = None,
             map_location: str = "cpu", lazy: bool = False) -> "TantraModel":
        """lazy=True registers experts with the router only; Tantra.inference.expert_cache
        attaches each adapter's weights when a request is routed to it."""
        # weights_only: model files hold tensors and plain metadata; never unpickle arbitrary objects.
        ck = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(ModelConfig.from_dict(ck["config"]))
        ranks = ck.get("expert_ranks", {})
        for name in ck.get("router_names", []):
            if lazy:
                model.router.add_expert(name)
            else:
                model.add_expert(name, ranks.get(name))
        if lazy:
            model.load_state_dict({k: v for k, v in ck["state"].items() if "lora_" not in k}, strict=False)
            return model
        model.load_state_dict(ck["state"], strict=False)
        if experts_dir is not None:
            for name in model.expert_names:
                f = Path(experts_dir) / f"{name}.pt"
                if f.exists():
                    e = torch.load(f, map_location=map_location, weights_only=True)
                    model.load_expert_state(name, e["state"], e["rank"])
        return model


def save_expert(model: TantraModel, name: str, experts_dir: str | Path, base_sha256: Optional[str] = None,
                tokenizer_sha256: Optional[str] = None) -> Path:
    d = Path(experts_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{name}.pt"
    payload = {"rank": model.expert_ranks[name], "state": model.expert_state(name)}
    if base_sha256:
        payload["base_sha256"] = base_sha256
    if tokenizer_sha256:
        payload["tokenizer_sha256"] = tokenizer_sha256
    torch.save(payload, f)
    return f


def build_model(preset_or_cfg="tiny", vocab_size: Optional[int] = None) -> TantraModel:
    from Tantra import get_preset
    cfg = get_preset(preset_or_cfg) if isinstance(preset_or_cfg, str) else preset_or_cfg
    if vocab_size:
        cfg.vocab_size = vocab_size
    return TantraModel(cfg)


def quantize_for_cpu(model: TantraModel, layers: bool = True, head: bool = True) -> TantraModel:
    """INT8 dynamic quantization for CPU inference; LoRA experts stay fp32 (they are tiny).

    Measured on the 64K base (docs/ROADMAP_CPU_EXPERTS.md): head-only keeps ~94% top-1
    agreement with FP32; quantising the transformer layers drops it to ~84%."""
    model.eval()
    for layer in (model.layers if layers else []):
        for mod in layer.lora_modules().values():
            mod.base = torch.ao.quantization.quantize_dynamic(
                nn.Sequential(mod.base), {nn.Linear}, dtype=torch.qint8)[0]
    if not head:
        return model
    # The tied output head (vocab x d_model) is the largest matmul per generated token; give
    # inference an INT8 copy. Kept outside the module tree so state_dict and training are unchanged.
    head = nn.Linear(model.cfg.d_model, model.cfg.vocab_size, bias=False)
    with torch.no_grad():
        head.weight.copy_(model.embed.weight)
    model.__dict__["_q_head"] = torch.ao.quantization.quantize_dynamic(nn.Sequential(head), {nn.Linear}, dtype=torch.qint8)[0]
    return model



# ──  model/peer.py  ──
"""Product-key expert layer: thousands of tiny experts, a handful active per token (CPU-friendly).

Each expert is one hidden neuron (a down vector u_i and an up vector v_i). A token picks its
top-k experts out of N = n_keys**2 with product keys (two sub-key searches of n_keys each, then
top-k over k*k candidates), so lookup costs O(sqrt(N)) instead of O(N). Only the selected rows of
the expert tables are read, and with sparse gradients only those rows are updated. Total
parameters grow with N (knowledge capacity) while compute per token stays roughly flat.
Based on product-key memories (Lample et al. 2019) and PEER (He 2024).

CPU size path (optional, inference):
- factor_low_rank_: shared basis + per-expert codes (about d/rank smaller)
- quantize_int8_: per-row INT8 + scale on whatever table is stored (~4x)
- compact_used_: drop experts that were never (or rarely) activated
Together these are the practical route toward ~100:1 expert-table size on CPU.
Float training path is unchanged until one of those methods is called.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _row_int8(weight: Tensor):
    """Per-row INT8. Returns (codes int8 [N, D], scales float [N])."""
    scale = weight.detach().abs().amax(dim=-1).clamp_min(1e-8) / 127.0
    codes = torch.round(weight.detach() / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return codes, scale


class ProductKeyExperts(nn.Module):
    def __init__(self, d_model: int, n_keys: int = 128, heads: int = 4, top_k: int = 16,
                 key_dim: int = 64, sparse: bool = True, track_activations: bool = False):
        super().__init__()
        self.n_keys, self.heads, self.top_k, self.key_dim = n_keys, heads, top_k, key_dim
        self.n_experts = n_keys * n_keys
        self.track_activations = track_activations
        self.query = nn.Linear(d_model, heads * key_dim, bias=False)
        self.keys = nn.Parameter(torch.randn(heads, 2, n_keys, key_dim // 2) / math.sqrt(key_dim // 2))
        # Expert tables as embeddings so backward touches only the selected rows (sparse=True).
        self.down = nn.Embedding(self.n_experts, d_model, sparse=sparse)
        self.up = nn.Embedding(self.n_experts, d_model, sparse=sparse)
        nn.init.normal_(self.down.weight, std=d_model ** -0.5)
        nn.init.zeros_(self.up.weight)  # starts as a no-op, so it can be added to a trained model safely

        if track_activations:
            self.register_buffer("activation_counts", torch.zeros(self.n_experts, dtype=torch.long), persistent=False)

        self.low_rank = False
        self.quantized = False
        self.compacted = False

    def route(self, x: Tensor):
        """Top-k (scores, expert ids) per head for x of shape [T, d]."""
        T = x.shape[0]
        q = self.query(x).view(T, self.heads, 2, self.key_dim // 2)
        s1 = torch.einsum('thd,hnd->thn', q[:, :, 0], self.keys[:, 0])
        s2 = torch.einsum('thd,hnd->thn', q[:, :, 1], self.keys[:, 1])
        v1, i1 = s1.topk(self.top_k, -1)
        v2, i2 = s2.topk(self.top_k, -1)
        cand = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(-2)
        ids = (i1.unsqueeze(-1) * self.n_keys + i2.unsqueeze(-2)).flatten(-2)
        scores, pick = cand.topk(self.top_k, -1)
        expert_ids = ids.gather(-1, pick)
        if self.track_activations and self.training:
            flat_ids = expert_ids.reshape(-1)
            self.activation_counts.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.long))
        return scores, expert_ids

    def _remap(self, ids: Tensor) -> Tensor:
        if not self.compacted:
            return ids
        return self.id_map[ids]

    def _gather_table(self, ids: Tensor, which: str) -> Tensor:
        """Selected rows [T, H, k, d] from float, INT8, or low-rank storage."""
        ids = self._remap(ids)
        if self.low_rank:
            basis = getattr(self, f"{which}_basis")
            if self.quantized:
                code = getattr(self, f"{which}_code_q")
                scale = getattr(self, f"{which}_code_scale")
                gathered = code[ids].float() * scale[ids].unsqueeze(-1)
            else:
                gathered = getattr(self, f"{which}_code")[ids]
            return gathered @ basis
        if self.quantized:
            q = getattr(self, f"{which}_q")
            scale = getattr(self, f"{which}_scale")
            return q[ids].float() * scale[ids].unsqueeze(-1)
        table = self.down if which == "down" else self.up
        return table(ids)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        scores, ids = self.route(x)
        gate = scores.softmax(-1)
        down = self._gather_table(ids, "down")
        h = F.gelu(torch.einsum('td,thkd->thk', x, down)) * gate
        if self.low_rank or self.quantized or self.compacted:
            up = self._gather_table(ids, "up")
            out = torch.einsum('thk,thkd->td', h, up)
        else:
            T = ids.shape[0]
            out = F.embedding_bag(
                ids.reshape(T, -1), self.up.weight,
                per_sample_weights=h.reshape(T, -1),
                mode='sum', sparse=self.up.sparse,
            )
        return out.reshape(shape)

    def active_params_per_token(self, d_model: int) -> int:
        if self.low_rank:
            rank = self.down_basis.shape[0]
            return self.heads * self.top_k * 2 * rank + 2 * rank * d_model
        return self.heads * self.top_k * 2 * d_model

    def total_expert_params(self) -> int:
        if self.low_rank:
            return int(self.down_code.numel() + self.up_code.numel()
                       + self.down_basis.numel() + self.up_basis.numel())
        if self.compacted:
            return int(self.down.weight.numel() + self.up.weight.numel())
        return 2 * self.n_experts * self.down.embedding_dim

    def stored_bytes(self) -> int:
        """Bytes holding expert tables (not the router). Each tensor is counted once."""
        seen = set()
        n = 0
        names = [
            "down_q", "up_q", "down_scale", "up_scale",
            "down_code", "up_code", "down_code_q", "up_code_q",
            "down_code_scale", "up_code_scale",
            "down_basis", "up_basis", "id_map",
        ]
        if not self.quantized and not self.low_rank:
            names.extend(["down.weight", "up.weight"])
        for name in names:
            if "." in name:
                mod, attr = name.split(".")
                t = getattr(getattr(self, mod, None), attr, None)
            else:
                t = getattr(self, name, None)
            if t is None or not torch.is_tensor(t):
                continue
            key = t.data_ptr()
            if key in seen:
                continue
            seen.add(key)
            n += t.numel() * t.element_size()
        return int(n)

    def usage_stats(self):
        if not self.track_activations:
            return None
        counts = self.activation_counts
        used = (counts > 0).sum().item()
        return {
            "n_experts": self.n_experts,
            "used_experts": used,
            "unused_experts": self.n_experts - used,
            "max_activations": counts.max().item(),
            "mean_activations": counts.float().mean().item(),
        }

    def prune_unused(self, min_activations: int = 1) -> int:
        """How many experts fall below min_activations. Does not change weights."""
        if not self.track_activations:
            return 0
        return int((self.activation_counts < min_activations).sum().item())

    @torch.no_grad()
    def factor_low_rank_(self, rank: int, sample: Optional[int] = None) -> dict:
        """Replace expert tables with a shared basis plus per-expert codes.

        Reconstruction is W ≈ codes @ basis. Storage ratio is about d_model / rank
        when N is large. Call quantize_int8_() afterwards for another ~4x.
        """
        if self.quantized or self.compacted:
            raise RuntimeError("factor_low_rank_ expects the original float tables")
        rank = int(rank)
        d = self.down.embedding_dim
        if rank < 1 or rank >= d:
            raise ValueError(f"rank must be in 1..{d - 1}")

        def _factor(weight: Tensor):
            n = weight.shape[0]
            take = n if sample is None else min(n, int(sample))
            idx = torch.randperm(n, device=weight.device)[:take]
            _, _, vh = torch.linalg.svd(weight[idx].float(), full_matrices=False)
            basis = vh[:rank].contiguous()
            codes = (weight.float() @ basis.T).contiguous()
            err = (weight.float() - codes @ basis).pow(2).mean().sqrt()
            ref = weight.float().pow(2).mean().sqrt().clamp_min(1e-8)
            return codes, basis, float((err / ref).item())

        down_code, down_basis, down_err = _factor(self.down.weight.data)
        up_code, up_basis, up_err = _factor(self.up.weight.data)
        self.down_code = nn.Parameter(down_code)
        self.up_code = nn.Parameter(up_code)
        self.down_basis = nn.Parameter(down_basis)
        self.up_basis = nn.Parameter(up_basis)
        # Free the full float tables so size matches the factored form.
        self.down = nn.Embedding(1, 1)
        self.up = nn.Embedding(1, 1)
        self.low_rank = True
        return {"rank": rank, "down_rel_rmse": down_err, "up_rel_rmse": up_err,
                "stored_bytes": self.stored_bytes()}

    @torch.no_grad()
    def quantize_int8_(self) -> dict:
        """Pack stored expert rows or low-rank codes to per-row INT8. Inference only."""
        if self.quantized:
            return {"stored_bytes": self.stored_bytes(), "already": True}
        before = self.stored_bytes()
        if self.low_rank:
            dq, ds = _row_int8(self.down_code.data)
            uq, us = _row_int8(self.up_code.data)
            self.register_buffer("down_code_q", dq)
            self.register_buffer("up_code_q", uq)
            self.register_buffer("down_code_scale", ds)
            self.register_buffer("up_code_scale", us)
            del self.down_code
            del self.up_code
        else:
            dq, ds = _row_int8(self.down.weight.data)
            uq, us = _row_int8(self.up.weight.data)
            self.register_buffer("down_q", dq)
            self.register_buffer("up_q", uq)
            self.register_buffer("down_scale", ds)
            self.register_buffer("up_scale", us)
        self.quantized = True
        after = self.stored_bytes()
        return {"before_bytes": before, "stored_bytes": after, "ratio": before / max(after, 1)}

    @torch.no_grad()
    def compact_used_(self, min_activations: int = 1) -> dict:
        """Drop experts below min_activations. Routing ids stay valid via id_map.

        Index 0 of the new tables is a reserved zero row. Unused ids map to it.
        """
        if not self.track_activations:
            raise RuntimeError("compact_used_ needs track_activations=True")
        if self.low_rank or self.quantized or self.compacted:
            raise RuntimeError("compact_used_ expects the original float tables")
        keep = (self.activation_counts >= min_activations).nonzero().flatten()
        n_keep = int(keep.numel())
        d = self.down.embedding_dim
        device = self.down.weight.device
        id_map = torch.zeros(self.n_experts, dtype=torch.long, device=device)
        if n_keep:
            id_map[keep] = torch.arange(1, n_keep + 1, device=device)
        down_w = torch.zeros(n_keep + 1, d, device=device)
        up_w = torch.zeros(n_keep + 1, d, device=device)
        if n_keep:
            down_w[1:] = self.down.weight.data[keep]
            up_w[1:] = self.up.weight.data[keep]
        before = self.stored_bytes()
        self.down = nn.Embedding.from_pretrained(down_w, freeze=False, sparse=True)
        self.up = nn.Embedding.from_pretrained(up_w, freeze=False, sparse=True)
        self.register_buffer("id_map", id_map)
        self.compacted = True
        after = self.stored_bytes()
        return {
            "kept": n_keep,
            "dropped": self.n_experts - n_keep,
            "before_bytes": before,
            "stored_bytes": after,
            "ratio": before / max(after, 1),
        }



# ──  data/expert_dataset.py  ──
"""
Expert datasets
===============
Each expert has its own jsonl file:  <data_dir>/<domain>/<expert>.jsonl

Accepted row formats:
  {"messages": [{"role": "user", "content": ...}, {"role": "assistant", ...}]}
  {"text": "<|user|>\\n...\\n<|assistant|> ..."}            (legacy)
  {"text": "plain text"}                                    (pretraining)

Junk rows (template filler like "Sample 12 of 10000", "[SYNTHETIC ...]")
and exact duplicates are dropped at load time.
"""


import json
import random
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import torch

from Tantra.tokenizer import TantraTokenizer

JUNK = re.compile(r"Sample \d+ of \d+|\[SYNTHETIC|GENERATION NEEDED", re.I)
LEGACY = re.compile(r"<\|(user|assistant|system)\|>")


def _legacy_to_messages(text: str) -> Optional[List[Dict]]:
    parts = LEGACY.split(text)
    if len(parts) < 3:
        return None
    msgs = []
    for role, content in zip(parts[1::2], parts[2::2]):
        content = content.strip()
        if content:
            msgs.append({"role": role, "content": content})
    return msgs or None


def read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict]:
    """Load rows as {"messages": [...]} or {"text": ...}; filters junk + dups."""
    rows, seen = [], set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "messages" in item:
                row = {"messages": item["messages"]}
                if "experts" in item:  # mixed-intent row (used by eval / router)
                    row["experts"] = item["experts"]
            elif "text" in item:
                msgs = _legacy_to_messages(item["text"])
                row = {"messages": msgs} if msgs else {"text": item["text"]}
            else:
                continue
            key = json.dumps(row, ensure_ascii=False, sort_keys=True)
            if JUNK.search(key) or key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def find_expert_files(data_dirs: Sequence[str | Path]) -> Dict[str, Path]:
    """expert name -> jsonl path. Earlier dirs win (put curated data first)."""
    found: Dict[str, Path] = {}
    for d in data_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for f in sorted(d.glob("*/*.jsonl")):
            found.setdefault(f.stem, f)
    return found


def row_texts(row: Dict) -> Iterator[str]:
    if "messages" in row:
        for m in row["messages"]:
            yield m["content"]
    else:
        yield row["text"]


def encode_row(tok: TantraTokenizer, row: Dict, max_len: int, assistant_only: bool = True):
    if "messages" in row:
        ids, mask = tok.encode_chat(row["messages"])
        if not assistant_only:
            mask = [1] * len(ids)
    else:
        ids = [tok.bos_id] + tok.encode(row["text"]) + [tok.end_id]
        mask = [1] * len(ids)
    ids, mask = ids[:max_len], mask[:max_len]
    labels = [i if m else -100 for i, m in zip(ids, mask)]
    return ids, labels


class EncodedDataset:
    """Pre-tokenized samples (small data fits in RAM easily)."""

    def __init__(self, tok: TantraTokenizer, rows: List[Dict], max_len: int = 256,
                 assistant_only: bool = True, expert_id: int = -1):
        self.pad_id = tok.pad_id
        self.items = [encode_row(tok, r, max_len, assistant_only) for r in rows]
        self.items = [it for it in self.items if any(l != -100 for l in it[1][1:])]
        self.expert_id = expert_id

    def __len__(self) -> int:
        return len(self.items)

    def batches(self, batch_size: int, shuffle: bool = True, rng: Optional[random.Random] = None):
        idx = list(range(len(self.items)))
        if shuffle:
            (rng or random).shuffle(idx)
        for s in range(0, len(idx), batch_size):
            chunk = [self.items[i] for i in idx[s:s + batch_size]]
            yield collate(chunk, self.pad_id)


def collate(chunk, pad_id: int):
    L = max(len(ids) for ids, _ in chunk)
    ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
    labels = torch.full((len(chunk), L), -100, dtype=torch.long)
    mask = torch.zeros((len(chunk), L), dtype=torch.long)
    for i, (a, b) in enumerate(chunk):
        ids[i, :len(a)] = torch.tensor(a)
        labels[i, :len(b)] = torch.tensor(b)
        mask[i, :len(a)] = 1
    return ids, labels, mask


def first_user_text(row: Dict) -> str:
    """The text the router sees: the last user message (what the request is about)."""
    if "messages" in row:
        users = [m["content"] for m in row["messages"] if m["role"] == "user"]
        return users[-1] if users else ""
    return row["text"][:200]


def validate_expert_datasets(data_dirs: Sequence[str | Path], experts: Sequence[str]) -> Dict:
    files = find_expert_files(data_dirs)
    report = {"ok": {}, "missing": [], "low": {}}
    for e in experts:
        if e not in files:
            report["missing"].append(e)
            continue
        n = len(read_jsonl(files[e]))
        (report["ok"] if n >= 100 else report["low"])[e] = n
    return report



# ──  data/expert_workbench.py  ──
"""Dataset contracts, inventory and preparation for all taxonomy experts.

Tantra.workbench_cli --help
This module never trains or changes Model/. Generated examples are verifiable
bootstrap tasks; registered experts are not automatically trained experts.
"""

import argparse
import ast
import csv
import hashlib
import io
import json
import random
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

from Tantra import ROOT, get_preset, load_taxonomy, validate_taxonomy

VERSION = 1
DEFAULT_OUT = ROOT / 'Datasets' / 'workbench'
WB_JUNK = re.compile(r'sample\s+\d+\s+of\s+\d+|relates to sample\s+\d+|\[SYNTHETIC|GENERATION NEEDED|this is a sample text in', re.I)
PRIORITY = ['greeting', 'farewell', 'smalltalk', 'clarification', 'empathy', 'identity_self',
            'identity_capabilities', 'identity_limitations', 'safety_refusal', 'lang_hi']
LANGUAGES = dict(zip(
    'en zh es hi ar pt bn ru ja de ko fr tr vi it th pl nl id sv cs el he ro hu fi da no sk hr sl et lv lt mt ga cy eu ca gl is mk sq bs me sr bg uk be ka'.split(),
    ['English','Chinese','Spanish','Hindi','Arabic','Portuguese','Bengali','Russian','Japanese','German',
     'Korean','French','Turkish','Vietnamese','Italian','Thai','Polish','Dutch','Indonesian','Swedish',
     'Czech','Greek','Hebrew','Romanian','Hungarian','Finnish','Danish','Norwegian','Slovak','Croatian',
     'Slovenian','Estonian','Latvian','Lithuanian','Maltese','Irish','Welsh','Basque','Catalan','Galician',
     'Icelandic','Macedonian','Albanian','Bosnian','Montenegrin','Serbian','Bulgarian','Ukrainian','Belarusian','Georgian']))
GOALS = {
 'greeting':'Recognize greetings and return a concise, language-matched welcome.',
 'farewell':'Recognize conversation endings and respond without restarting the conversation.',
 'smalltalk':'Handle thanks and everyday social questions without inventing personal experiences.',
 'empathy':'Acknowledge feelings supportively without diagnosis or pretending to have feelings.',
 'clarification':'Ask the minimum useful question when the request lacks required context.',
 'apology':'Acknowledge a concrete error and correct it without repeated apologies.',
 'humor':'Produce relevant, harmless jokes and distinguish literal from humorous requests.',
 'identity_self':'Identify consistently as Tantra, a language model from Atulya AI.',
 'identity_capabilities':'Describe only capabilities available in the actual deployment.',
 'identity_limitations':'Explain uncertainty, access limits and possible errors honestly.',
 'identity_persona':'Maintain a helpful consistent voice without claiming human identity.',
 'identity_consistency':'Keep identity facts consistent across paraphrases and multi-turn challenges.',
 'identity_metadata':'Report supplied model metadata; do not invent version or training facts.',
 'tone_style':'Follow explicit tone and style constraints while preserving meaning.',
 'safety_refusal':'Refuse harmful requests briefly and offer a relevant safe alternative.',
 'logic':'Solve finite logical problems with checkable premises and conclusions.',
 'math':'Solve arithmetic and algebra with exact, programmatically checkable answers.',
 'coding':'Write small correct programs that satisfy supplied specifications and tests.',
 'reasoning_planning':'Explain a feasible plan with explicit assumptions and dependencies.',
 'debugging':'Identify the actual cause in a supplied failing program and propose a minimal fix.',
 'algorithm_design':'Choose and explain an algorithm with correct complexity and edge cases.',
 'code_review':'Find actionable defects in supplied code, with a concrete failure example.',
 'architecture':'Compare system designs against stated constraints without inventing requirements.',
 'optimization':'Optimize measured bottlenecks while preserving tested behavior.',
 'verification':'Check a proposed answer or proof against its stated requirements.',
 'symbolic_reasoning':'Apply explicit symbolic rules without changing their meaning.',
 'causal_inference':'Distinguish correlation from causation and identify necessary assumptions.',
 'pii':'Identify sensitive personal data and transform it according to a stated privacy policy.',
 'toxicity':'Recognize abusive content and provide a non-abusive response or rewrite.',
 'jailbreak':'Keep instruction boundaries when untrusted text asks to override them.',
 'bias':'Identify unsupported stereotypes and respond fairly using relevant evidence.',
 'medical_safety':'Handle medical questions cautiously, with uncertainty and appropriate escalation.',
 'financial_safety':'Explain financial concepts without guaranteed returns or fabricated certainty.',
 'legal_safety':'Distinguish general legal information from jurisdiction-specific professional advice.',
 'child_safety':'Respond appropriately to child-related risk without sexual or exploitative material.',
 'self_harm':'Respond compassionately to distress; never generate harmful procedural instructions.',
 'misinformation':'Identify unsupported claims and separate evidence from speculation.',
 'summarization':'Summarize supplied text accurately with requested length and coverage.',
 'translation':'Translate between explicitly specified languages without adding facts.',
 'qa':'Answer only what the supplied evidence supports and abstain when evidence is missing.',
 'creative_writing':'Produce original writing satisfying explicit genre, tone and length constraints.',
 'analysis':'Analyze supplied material using supported observations and stated assumptions.',
 'extraction':'Extract requested fields exactly from supplied records or text.',
 'formatting':'Change representation without changing the underlying data.',
 'rewriting':'Rewrite supplied text while preserving specified facts and constraints.',
 'classification':'Assign only labels from a supplied ontology and explain ambiguous cases.',
 'comparison':'Compare options on explicit criteria without inventing attributes.',
 'synthesis':'Combine supplied sources while preserving disagreements and attribution.',
 'evaluation':'Assess supplied work against an explicit rubric.',
 'planning':'Produce executable steps with prerequisites, success checks and stop conditions.',
 'task_decomposition':'Break a request into non-overlapping tasks with dependencies.',
 'tool_selection':'Choose an available tool from its actual schema; abstain if none fits.',
 'state_tracking':'Update an explicit task state consistently across turns.',
 'error_recovery':'Select a bounded retry or rollback based on a supplied failure.',
 'multi_step_reasoning':'Complete linked steps while retaining intermediate constraints.',
 'api_schema':'Emit calls matching a supplied API schema without inventing fields.',
 'function_composition':'Connect tool outputs to compatible downstream inputs.',
 'parallel_calls':'Parallelize independent calls while preserving dependency order.',
 'streaming_results':'Assemble partial results and distinguish partial from completed output.',
 'auth_handling':'Handle authorization errors without exposing or fabricating credentials.',
 'tool_error_handling':'Interpret structured tool failures and choose a justified response.',
 'json_schema':'Emit syntactically valid JSON satisfying a supplied schema.',
 'xml_format':'Emit well-formed XML with required structure and correct escaping.',
 'markdown_tables':'Render supplied records as correct, escaped Markdown tables.',
 'csv_export':'Serialize supplied records as CSV with correct quoting and column order.',
 'regex_validation':'Match or validate strings using a stated pattern and check edge cases.',
 'query_rewrite':'Rewrite a search query without introducing unsupported entities or claims.',
 'doc_ranking':'Rank supplied documents by their evidence for a given query.',
 'citation_extraction':'Attach claims to exact supplied source spans and identifiers.',
 'hallucination_check':'Flag claims unsupported by supplied evidence and propose grounded corrections.',
 'context_compression':'Compress supplied evidence while retaining facts required by a question.',
 'repo_navigation':'Find relevant files and symbols from a supplied repository snapshot.',
 'test_generation':'Create tests that fail for a concrete supplied defect.',
 'debug_loop':'Use a supplied failure/test result to propose and validate the next fix.',
 'refactor_planning':'Plan behavior-preserving changes with dependency and test coverage.',
 'code_search':'Locate code matching a query and cite the actual supplied paths.',
 'dependency_analysis':'Explain actual dependencies and cycles from supplied manifests or code.',
 'translate_pivot':'Translate through a pivot language while checking meaning preservation.',
 'translate_quality':'Evaluate adequacy and fluency against a supplied source and rubric.',
 'translate_terminology':'Preserve a supplied terminology glossary consistently.',
 'translate_formality':'Control register and honorifics without changing meaning.',
 'translate_domain':'Translate domain text with appropriate terminology and uncertainty.',
 'translate_literary':'Translate literary meaning, voice and imagery with documented tradeoffs.',
 'translate_technical':'Translate technical instructions while preserving identifiers and numbers.',
 'translate_low_resource':'Handle supported low-resource language pairs with native-speaker review.',
 'retrieval':'Retrieve exact evidence from long supplied documents.',
 'hierarchy':'Organize long material into a faithful hierarchical outline.',
 'compression':'Compress long material under a budget while preserving required information.',
 'rnn_memory':'Maintain explicit sequential state over a supplied long sequence.'
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def normalized(text):
    return unicodedata.normalize('NFC', text)


def split_for(family):
    n = int(digest(family)[:12], 16) % 100
    return 'train' if n < 90 else 'val' if n < 95 else 'test'


def family_key(row):
    """Keep reviewed sheet translations in the original English split namespace.

    Source attribution remains unchanged. Restrict the alias to the known sheet
    source and family format; unrelated publishers retain independent families.
    Existing English split hashes are preserved.
    """
    source = row['source']['id']
    family = row['family_id']
    if (source == 'tantra-conversation-human-v1'
            and family.startswith(row['expert'] + ':draft-v1:')):
        source = 'tantra-conversation-drafts-v1'
    return source + ':' + family


def catalog(taxonomy=None, config=None):
    taxonomy = taxonomy or load_taxonomy()
    errors = validate_taxonomy(taxonomy)
    if errors:
        raise ValueError('; '.join(errors))
    cfg = config or get_preset('small')
    cfg.vocab_size = 10277
    d, kv, h = cfg.d_model, cfg.d_model // cfg.n_heads * cfg.n_kv_heads, cfg.mlp_hidden
    shapes = {'q': (d,d), 'k': (d,kv), 'v': (d,kv), 'o': (d,d),
              'gate': (d,h), 'up': (d,h), 'down': (h,d)}
    result = {}
    for domain, info in taxonomy.items():
        for name in info['experts']:
            lang = name[5:] if name.startswith('lang_') else None
            goal = (f'Understand and respond naturally in {LANGUAGES[lang]}, with native-speaker review; English text with a language tag is not valid evidence.'
                    if lang else GOALS.get(name))
            if name.startswith('gk_'):
                topic = name[3:].replace('_',' ')
                if name=='gk_math': topic='mathematical definitions and facts (arithmetic computation belongs to math)'
                goal = f'Answer {topic} questions with checkable references, explicit uncertainty and dated sources for changing facts.'
            if not goal:
                raise ValueError(f'Missing expert specification: {name}')
            rank = int(info['rank'])
            result[name] = {'expert': name, 'domain': domain, 'goal': goal,
                            'languages': [lang] if lang else ['hi','hinglish','en'],
                            'target_rank': rank,
                            'adapter_parameters_small': cfg.n_layers * rank * sum(sum(shapes[t]) for t in cfg.lora_targets),
                            'minimum': {'train':1000,'val':100,'test':100},
                            'priority': PRIORITY.index(name) if name in PRIORITY else 100,
                            'deployment':'disabled_until_independent_held_out_gain', 'pilot_rank':8}
    return result


def initialize(out=DEFAULT_OUT, docs=ROOT/'docs'/'experts'):
    out, docs = Path(out), Path(docs)
    specs = catalog()
    for name, item in specs.items():
        for split in ('incoming',):
            (out / split / item['domain']).mkdir(parents=True, exist_ok=True)
        card = docs / item['domain'] / f'{name}.md'
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text(f"# {name}\n\n{item['goal']}\n\n"
                        f"- Domain: {item['domain']}; languages: {', '.join(item['languages'])}.\n"
                        f"- Incoming: `Datasets/workbench/incoming/{item['domain']}/{name}.jsonl`.\n"
                        f"- Taxonomy target rank: {item['target_rank']}; small-model adapter parameters: {item['adapter_parameters_small']:,}.\n"
                        "- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.\n"
                        "- Minimum independent examples: 1,000 train / 100 validation / 100 test.\n"
                        "- Follow the [shared dataset contract](../../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.\n"
                        "- A card is a capability specification, not a trained or validated expert.\n", encoding='utf-8')
    write_json(docs.parent/'expert_catalog.json', {'version':VERSION,'experts':specs})
    write_json(docs.parent/'dataset_record.schema.json',dataset_schema())
    return specs


def messages_of(row):
    if not isinstance(row,dict):
        raise ValueError('row_not_object')
    messages = row.get('messages')
    if messages is None and isinstance(row.get('text'),str):
        messages = _legacy_to_messages(row['text'].replace('\\n','\n'))
    if not isinstance(messages,list) or not messages:
        raise ValueError('missing_chat_messages')
    if any(not isinstance(m,dict) or m.get('role') not in ('system','user','assistant')
           or not isinstance(m.get('content'),str) or not m['content'].strip() for m in messages):
        raise ValueError('invalid_messages')
    if not any(m['role']=='user' for m in messages) or messages[-1]['role']!='assistant':
        raise ValueError('missing_user_or_final_assistant')
    if any(WB_JUNK.search(m['content']) for m in messages):
        raise ValueError('placeholder')
    return [{'role':m['role'],'content':m['content'].strip()} for m in messages]


def verify_auto(row):
    v = row.get('verification') or {}
    answer = row['messages'][-1]['content'].strip()
    kind = v.get('kind')
    if kind=='arithmetic':
        a,b,op = v['a'],v['b'],v['op']
        if not isinstance(a,int) or not isinstance(b,int): return False
        expected = {'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b}.get(op)
        return (expected is not None and answer==str(expected()) and re.search(r'(?<!\d)'+re.escape(str(a))+r'\s*'+re.escape(op)+r'\s*'+re.escape(str(b))+r'(?!\d)',row['messages'][0]['content']) is not None)
    if kind=='json':
        return json.loads(answer)==v['expected'] and json.loads(row['messages'][0]['content'].split('\n',1)[1])==v['expected']
    if kind=='csv':
        expected = [[str(x) for x in r] for r in v['expected']]
        return list(csv.reader(io.StringIO(answer)))==expected and json.loads(row['messages'][0]['content'].split('\n',1)[1])==v['expected']
    return False


def validate_record(row, expert, strict=True):
    if WB_JUNK.search(json.dumps(row,ensure_ascii=False)):
        raise ValueError('placeholder')
    messages = messages_of(row)
    clean = {**row,'messages':messages,'expert':expert}
    if not strict:
        return clean
    if row.get('schema_version')!=VERSION or row.get('expert')!=expert:
        raise ValueError('schema_or_expert_mismatch')
    if not isinstance(row.get('language'),str) or not row['language']:
        raise ValueError('missing_language')
    if not isinstance(row.get('template_id'),str) or not row['template_id']:
        raise ValueError('missing_template')
    if row['language'] not in set(LANGUAGES)|{'hinglish'}:
        raise ValueError('unsupported_language')
    if expert.startswith('lang_') and row['language']!=expert[5:]:
        raise ValueError('expert_language_mismatch')
    if not isinstance(row.get('family_id'),str) or not row['family_id']:
        raise ValueError('missing_family')
    source,review = row.get('source'),row.get('review')
    if not isinstance(source,dict) or any(not source.get(k) for k in ('id','revision','license','kind')):
        raise ValueError('missing_provenance')
    if source['license'].lower() in ('unknown','todo','unverified'):
        raise ValueError('unknown_license')
    if not isinstance(review,dict) or not review.get('reviewer'):
        raise ValueError('missing_review')
    if review.get('status')=='verified_auto':
        owner={'arithmetic':'math','json':'json_schema','csv':'csv_export'}.get((row.get('verification') or {}).get('kind'))
        if owner!=expert:
            raise ValueError('wrong_auto_task_owner')
        if source['id']!='tantra-verifiable-v1' or not verify_auto(clean):
            raise ValueError('verification_failed')
    elif review.get('status')!='reviewed':
        raise ValueError('review_required')
    return clean


def source_files(inputs):
    return sorted({p.resolve() for base in inputs for p in Path(base).rglob('*.jsonl')})


def audit(inputs=None, out=DEFAULT_OUT, model_dir=ROOT/'Model'):
    specs = catalog()
    inputs = inputs or [ROOT/'Datasets'/'curated', ROOT/'Datasets'/'experts',Path(out)/'incoming']
    stats = {n:{'files':[],'raw_rows':0,'usable_chat_rows':0,'reviewed_rows':0,'rejected':{},
                'weights_present':(Path(model_dir)/'experts'/f'{n}.pt').is_file()} for n in specs}
    extras = []
    for path in source_files(inputs):
        name = file_expert(path,specs)
        if name not in specs:
            extras.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
            continue
        record = stats[name]
        record['files'].append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
        rejects = Counter(record['rejected'])
        with path.open(encoding='utf-8-sig') as f:
            for line in f:
                if not line.strip(): continue
                record['raw_rows'] += 1
                # Skip known filler before JSON parsing; preserve exact counts.
                if WB_JUNK.search(line):
                    rejects['placeholder'] += 1
                    continue
                try:
                    row = json.loads(line)
                    validate_record(row,name,False)
                    record['usable_chat_rows'] += 1
                    try:
                        validate_record(row,name,True)
                        record['reviewed_rows'] += 1
                    except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
                        rejects['contract:'+str(exc)] += 1
                except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
                    rejects[str(exc)] += 1
        record['rejected'] = dict(rejects)
    extra_weights = [p.stem for p in (Path(model_dir)/'experts').glob('*.pt') if p.stem not in specs]
    result = {'version':VERSION,'experts':stats,'supplementary_data_files':extras,
              'supplementary_weights':extra_weights,
              'summary':{'registered':len(specs),'taxonomy_weights_present':sum(s['weights_present'] for s in stats.values()),
                         'raw_rows':sum(s['raw_rows'] for s in stats.values()),
                         'usable_chat_rows':sum(s['usable_chat_rows'] for s in stats.values()),
                         'reviewed_rows':sum(s['reviewed_rows'] for s in stats.values()),
                         'placeholder_rows':sum(s['rejected'].get('placeholder',0) for s in stats.values())}}
    write_json(Path(out)/'audit.json',result)
    return result


def generated_rows(count=5000):
    for index in range(count):
        a,b = index*37%1999-999,index*53%997-498
        op = ['+','-','*'][index%3]
        expected = {'+':a+b,'-':a-b,'*':a*b}[op]
        lang = ['en','hi','hinglish'][(index//3)%3]
        question = {'en':f'Calculate {a} {op} {b}. Return only the integer.',
                    'hi':f'{a} {op} {b} की गणना करें। केवल पूर्णांक लिखें।',
                    'hinglish':f'{a} {op} {b} ka jawab do. Sirf integer likho.'}[lang]
        common = {'schema_version':VERSION,'language':lang,'family_id':f'arithmetic:{op}:{a}:{b}',
                  'template_id':f'arithmetic:{op}','source':{'id':'tantra-verifiable-v1','revision':'1','license':'MIT','kind':'synthetic_verified'},
                  'review':{'status':'verified_auto','reviewer':'exact arithmetic/serialization verifier'}}
        for name in ('math',):
            yield {**common,'expert':name,'messages':[{'role':'user','content':question},{'role':'assistant','content':str(expected)}],
                   'verification':{'kind':'arithmetic','a':a,'b':b,'op':op}}
        obj = {'id':index,'active':bool(index%2),'values':list(range(index%5+1))}
        payload = json.dumps(obj,ensure_ascii=False)
        yield {**common,'expert':'json_schema','language':'en','family_id':f'json-record:{index}','template_id':'json-serialize',
               'messages':[{'role':'user','content':'Return the following object as valid JSON, with no Markdown.\n'+payload},
                           {'role':'assistant','content':payload}], 'verification':{'kind':'json','expected':obj}}
        table = [['id','name','value'],[index,f'example,{index}',a],[index+1,f'quote"{index}',b]]
        stream = io.StringIO(newline='')
        csv.writer(stream,lineterminator='\n').writerows(table)
        yield {**common,'expert':'csv_export','language':'en','family_id':f'csv-record:{index}','template_id':'csv-serialize',
               'messages':[{'role':'user','content':'Convert these records to CSV, preserving row and column order.\n'+json.dumps(table)},
                           {'role':'assistant','content':stream.getvalue().rstrip('\n')}],
               'verification':{'kind':'csv','expected':table}}


def seed(out=DEFAULT_OUT, count=5000):
    if count < 1: raise ValueError('count must be positive')
    specs = catalog()
    handles, counts = {},Counter()
    try:
        for row in generated_rows(count):
            name = row['expert']
            validate_record(row,name)
            if name not in handles:
                path = Path(out)/'incoming'/specs[name]['domain']/f'verified-{name}.jsonl'
                path.parent.mkdir(parents=True,exist_ok=True)
                handles[name] = path.open('w',encoding='utf-8')
            handles[name].write(json.dumps(row,ensure_ascii=False)+'\n')
            counts[name] += 1
    finally:
        for f in handles.values(): f.close()
    return dict(counts)


def file_expert(path, specs):
    name = path.stem.removeprefix('verified-')
    return name if name in specs else None


def prepare(out=DEFAULT_OUT):
    out = Path(out)
    specs = catalog()
    manifest_path = out/'prepared'/'manifest.json'
    if manifest_path.exists():
        raise ValueError('Prepared bundle is immutable; use --out for a new dataset version')
    out.mkdir(parents=True,exist_ok=True)
    rejects,duplicates = Counter(),0
    parent = {}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        parent[max(a,b)]=min(a,b)
    with tempfile.TemporaryDirectory(prefix='tantra-dataset-') as temporary:
        db=sqlite3.connect(Path(temporary)/'dedup.sqlite')
        db.execute('CREATE TABLE records(expert TEXT, hash TEXT, family TEXT, payload TEXT, UNIQUE(expert,hash))')
        db.execute('CREATE TABLE documents(hash TEXT PRIMARY KEY, family TEXT)')
        db.execute('CREATE INDEX expert_index ON records(expert)')
        quarantine = out/'quarantine.jsonl'
        with quarantine.open('w',encoding='utf-8') as bad:
            for path in source_files([out/'incoming']):
                name=file_expert(path,specs)
                if not name:
                    rejects['unknown_expert_file']+=1
                    continue
                with path.open(encoding='utf-8-sig') as f:
                    for number,line in enumerate(f,1):
                        if not line.strip(): continue
                        try:
                            row=validate_record(json.loads(line),name)
                            key=digest(normalized(json.dumps(row['messages'],ensure_ascii=False,sort_keys=True)))
                            family=family_key(row)
                            prior=db.execute('SELECT family FROM documents WHERE hash=?',(key,)).fetchone()
                            if prior: union(family,prior[0])
                            else: db.execute('INSERT INTO documents VALUES (?,?)',(key,family))
                            find(family)
                            inserted=db.execute('INSERT OR IGNORE INTO records VALUES (?,?,?,?)',
                                                (name,key,family,json.dumps(row,ensure_ascii=False))).rowcount
                            duplicates+=not inserted
                        except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
                            rejects[str(exc)]+=1
                            bad.write(json.dumps({'file':str(path),'line':number,'reason':str(exc),'sample':line[:500]},ensure_ascii=False)+'\n')
        db.commit()
        result={}
        for name,item in specs.items():
            counts=Counter(); families={s:set() for s in ('train','val','test')}; languages=Counter(); templates=set(); files={}
            writers={}
            try:
                for split in families:
                    path=out/'prepared'/split/item['domain']/f'{name}.jsonl'
                    if db.execute('SELECT 1 FROM records WHERE expert=? LIMIT 1',(name,)).fetchone():
                        path.parent.mkdir(parents=True,exist_ok=True)
                        writers[split]=path.open('w',encoding='utf-8'); files[split]=str(path.relative_to(out/'prepared'))
                for h,family,payload in db.execute('SELECT hash,family,payload FROM records WHERE expert=? ORDER BY hash',(name,)):
                    row=json.loads(payload); group=find(family); split=split_for(group)
                    row['record_id']=h; row['split_group']=group; row['split']=split
                    writers[split].write(json.dumps(row,ensure_ascii=False)+'\n')
                    counts[split]+=1; families[split].add(group); languages[row['language']]+=1
                    templates.add(row.get('template_id','not_supplied'))
            finally:
                for f in writers.values(): f.close()
            ready=all(counts[s]>=item['minimum'][s] and len(families[s])>=item['minimum'][s] for s in families)
            result[name]={'state':'data_ready' if ready else 'needs_data','rows':dict(counts),
                          'families':{s:len(v) for s,v in families.items()},'languages':dict(languages),
                          'template_count':len(templates),'files':{s:{'path':p,'sha256':hashlib.sha256((out/'prepared'/p).read_bytes()).hexdigest()} for s,p in files.items()},
                          'minimum':item['minimum'], 'trained':False, 'quality_validated':False}
        db.close()
    manifest={'version':VERSION,'seed':42,'split_policy':'sha256(source-family) modulo 100: 90/5/5; duplicate-connected families unioned globally',
              'experts':result,'rejected':dict(rejects),'duplicates_removed':duplicates,
              'summary':{'registered':len(specs),'data_ready':sum(v['state']=='data_ready' for v in result.values()),
                         'accepted_rows':sum(sum(v['rows'].values()) for v in result.values())}}
    write_json(manifest_path,manifest)
    return manifest


def training_plan(out=DEFAULT_OUT):
    out=Path(out)
    manifest=json.loads((out/'prepared'/'manifest.json').read_text(encoding='utf-8'))
    specs=catalog(); tasks=[]
    for name,item in sorted(specs.items(),key=lambda pair:(pair[1]['priority'],pair[0])):
        state=manifest['experts'][name]
        tasks.append({'expert':name,'domain':item['domain'],'state':state['state'],'goal':item['goal'],
                      'blocked_on':[] if state['state']=='data_ready' else ['reviewed independent train/val/test examples'],
                      'train_file':str(out/'prepared'/state['files']['train']['path']) if 'train' in state['files'] else None,
                      'validation_file':str(out/'prepared'/state['files']['val']['path']) if 'val' in state['files'] else None,
                      'test_file':str(out/'prepared'/state['files']['test']['path']) if 'test' in state['files'] else None,
                      'adapter_parameters_at_target_rank':item['adapter_parameters_small'],
                      'instruction':'Run the workbench train command for a data_ready expert. Keep frozen splits; validate answers and router separately. Export to a new directory.'})
    write_json(out/'training_plan.json',{'tasks':tasks,'warning':'A plan is not a trained model. The workbench train command preserves frozen splits; the legacy trainer re-splits and must not consume the bundle directly.'})
    return tasks



def dataset_schema():
    text={'type':'string','minLength':1}
    return {'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Tantra expert record v1','type':'object',
            'required':['schema_version','expert','language','family_id','template_id','messages','source','review'],
            'properties':{'schema_version':{'const':VERSION},'expert':{'enum':list(catalog())},
             'language':{'enum':list(LANGUAGES)+['hinglish']},'family_id':text,'template_id':text,
             'messages':{'type':'array','minItems':2,'items':{'type':'object','required':['role','content'],
               'properties':{'role':{'enum':['system','user','assistant']},'content':text}}},
             'source':{'type':'object','required':['id','revision','license','kind'],
               'properties':{k:text for k in ['id','revision','license','kind','url']}},
             'review':{'type':'object','required':['status','reviewer'],
               'properties':{'status':{'enum':['reviewed','verified_auto']},'reviewer':text}},
             'verification':{'type':'object'}},'additionalProperties':True}


def report(out=DEFAULT_OUT, docs=ROOT/'docs'):
    out,docs=Path(out),Path(docs)
    a=json.loads((out/'audit.json').read_text(encoding='utf-8'))
    m=json.loads((out/'prepared'/'manifest.json').read_text(encoding='utf-8'))
    specs=catalog()
    lines=['# Expert coverage snapshot','',
           'Regenerate with `Tantra.workbench_cli report`. This is a local-data snapshot, not live telemetry.','',
           '## Summary','',
           f"- Registered capabilities: {len(specs)}; taxonomy weight files: {a['summary']['taxonomy_weights_present']}; supplementary: {', '.join(a['supplementary_weights']) or 'none'}.",
           f"- Audited rows: {a['summary']['raw_rows']:,}; rejected filler: {a['summary']['placeholder_rows']:,}.",
           f"- Structurally usable chat: {a['summary']['usable_chat_rows']:,}; reviewed/verifiable: {a['summary']['reviewed_rows']:,}.",
           f"- Prepared examples: {m['summary']['accepted_rows']:,}; pilot-ready experts: {m['summary']['data_ready']}.",
           '- Legacy examples without provenance/review remain excluded. Saved weights are not proof of answer quality.',
           '- Generated task templates are narrow bootstrap coverage, not production-ready capabilities. Keep unsupported or redundant experts disabled.','',
           '## Prepared bootstrap','',
           '| Expert | Train | Validation | Test | Templates | Languages |',
           '| --- | ---: | ---: | ---: | ---: | --- |']
    for name,item in m['experts'].items():
        if item['state']=='data_ready':
            r=item['rows']; lines.append(f"| {name} | {r.get('train',0)} | {r.get('val',0)} | {r.get('test',0)} | {item['template_count']} | {', '.join(item['languages'])} |")
    lines += ['', '## Expert inventory','',
              '| Expert | Domain | Raw rows | Usable chat | Reviewed | Prepared state | Weights |',
              '| --- | --- | ---: | ---: | ---: | --- | --- |']
    for name,spec in specs.items():
        item=a['experts'][name]; state=m['experts'][name]['state']
        lines.append(f"| [{name}](experts/{spec['domain']}/{name}.md) | {spec['domain']} | {item['raw_rows']} | {item['usable_chat_rows']} | {item['reviewed_rows']} | {state} | {'present, unvalidated' if item['weights_present'] else 'absent'} |")
    lines += ['', '## Continue','',
              'Prioritize reviewed Hindi/Hinglish/English conversation and identity examples, expand task diversity, then train isolated candidates. Read HANDOFF.md and DATASET_SPEC.md before adding data. Use new immutable bundle directories for new versions. Evaluate answer quality, router behavior and regressions before deployment.','']
    docs.mkdir(parents=True,exist_ok=True)
    (docs/'EXPERT_STATUS.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(docs/'dataset_record.schema.json',dataset_schema())
    return {'registered':len(specs),'pilot_ready':m['summary']['data_ready'],'report':str(docs/'EXPERT_STATUS.md')}


def workbench_cli(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('init'); p=sub.add_parser('audit'); p.add_argument('--inputs',nargs='+',type=Path)
    p=sub.add_parser('seed'); p.add_argument('--count',type=int,default=5000)
    sub.add_parser('prepare'); sub.add_parser('plan'); sub.add_parser('status'); sub.add_parser('report')
    p=sub.add_parser('train'); p.add_argument('--expert',required=True); p.add_argument('--model-dir',type=Path,default=ROOT/'Model'); p.add_argument('--output',type=Path,required=True); p.add_argument('--steps',type=int,default=100); p.add_argument('--rank',type=int,default=8); p.add_argument('--threads',type=int,default=4); p.add_argument('--release-check',action='store_true'); p.add_argument('--small-pilot',action='store_true',help='Explicitly allow an undersized reviewed bundle; never counts as release-ready')
    args=parser.parse_args(argv)
    if args.command=='init': result={'registered':len(initialize(args.out))}
    elif args.command=='audit': result=audit(args.inputs,args.out)['summary']
    elif args.command=='seed': result=seed(args.out,args.count)
    elif args.command=='prepare': result=prepare(args.out)['summary']
    elif args.command=='plan': result={'tasks':len(training_plan(args.out))}
    elif args.command=='report': result=report(args.out)
    elif args.command=='train':
        result=train_one(args.out,args.expert,args.model_dir,args.output,args.steps,args.rank,args.threads,args.release_check,args.small_pilot)
    else:
        path=args.out/'prepared'/'manifest.json'
        result=json.loads(path.read_text(encoding='utf-8'))['summary'] if path.exists() else {'state':'not_prepared','registered':len(catalog())}
    print(json.dumps(result,indent=2,ensure_ascii=True))
    return 0



# ──  data/train_prepared.py  ──
"""Train one prepared adapter without loading or rewriting all other experts."""

import json
import math
import os
import random
import time
from pathlib import Path



def load_split(root, item, expert):
    import hashlib
    # Manifests written on Windows store backslash paths; accept them on every OS.
    path=Path(root)/'prepared'/Path(*item['path'].replace('\\','/').split('/'))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        raise ValueError('Prepared split checksum mismatch: '+str(path))
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(validate_record(json.loads(line),expert))
    return rows


def train_one(out, expert, model_dir, output, steps=100, rank=8, threads=4, release_check=False, small_pilot=False):
    import hashlib
    import torch
    from Tantra import ModelConfig
    from Tantra.tokenizer import TantraTokenizer
    out,model_dir,output=Path(out).resolve(),Path(model_dir).resolve(),Path(output).resolve()
    if output==model_dir or model_dir.is_relative_to(output) or output.is_relative_to(model_dir):
        raise ValueError('Output must be separate from the source Model directory')
    if steps<1 or rank<1 or threads<1: raise ValueError('steps, rank and threads must be positive')
    manifest_path=out/'prepared'/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if expert not in manifest['experts']: raise ValueError('Unknown expert: '+expert)
    record=manifest['experts'][expert]
    undersized=record['state']!='data_ready'
    if undersized and not small_pilot: raise ValueError('Expert data is not ready: '+expert)
    if undersized and release_check: raise ValueError('An undersized pilot cannot request release validation')
    if (output/f'{expert}.pt').exists(): raise ValueError('Candidate already exists; use a new output directory')
    rows={s:load_split(out,record['files'][s],expert) for s in ('train','val','test')}
    if any(not values for values in rows.values()): raise ValueError('Every pilot split must contain reviewed examples')
    groups={s:{row['split_group'] for row in values} for s,values in rows.items()}
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Cross-split family leakage')
    torch.set_num_threads(threads); torch.manual_seed(42)
    tok=TantraTokenizer.load(model_dir/'tokenizer.json')
    checkpoint=torch.load(model_dir/'base.pt',map_location='cpu',weights_only=False)
    cfg=ModelConfig.from_dict(checkpoint['config'])
    if cfg.vocab_size!=tok.vocab_size: raise ValueError('Base/tokenizer vocabulary mismatch')
    model=TantraModel(cfg)
    base={k:v for k,v in checkpoint['state'].items() if not k.startswith('router.') and 'lora_' not in k}
    missing,unexpected=model.load_state_dict(base,strict=False)
    if unexpected or any(not k.startswith('router.') for k in missing):
        raise ValueError('Incompatible base state')
    del checkpoint,base
    for p in model.parameters(): p.requires_grad=False
    model.add_expert(expert,rank)
    parameters=model.expert_parameters(expert)
    model.set_active([(expert,1.)])
    datasets={s:EncodedDataset(tok,values,max_len=min(256,cfg.max_seq_len)) for s,values in rows.items()}
    if any(not len(dataset) for dataset in datasets.values()): raise ValueError('No assistant labels after truncation')
    def loss_on(dataset):
        model.eval(); total=0.; count=0
        with torch.no_grad():
            for ids,labels,_ in dataset.batches(4,shuffle=False):
                n=int((labels[:,1:]!=-100).sum())
                value=float(model.loss(ids,labels))
                if not math.isfinite(value): raise FloatingPointError('Non-finite validation loss')
                total+=value*n; count+=n
        return total/count
    model.set_active([]); before=loss_on(datasets['val']); model.set_active([(expert,1.)])
    optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=.01)
    rng=random.Random(42); step=0; started=time.perf_counter(); peak=0
    import psutil
    while step<steps:
        for ids,labels,_ in datasets['train'].batches(4,rng=rng):
            memory=psutil.virtual_memory()
            rss=psutil.Process().memory_info().rss; peak=max(peak,rss)
            if rss>8*2**30 or memory.available<2*2**30:
                raise MemoryError('Resource limit reached; source weights remain unchanged')
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss=model.loss(ids,labels)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite training loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not bool(torch.isfinite(norm)): raise FloatingPointError('Non-finite gradient')
            optimizer.step(); step+=1
            if step>=steps: break
    after=loss_on(datasets['val'])
    # No test-set answers are generated until the optional release check.
    answer_accuracy={}
    if release_check and after<before:
        for split in ('val','test'):
            checked=correct=0
            for row in rows[split][:100]:
                ids,_=tok.encode_chat(row['messages'][:-1],add_generation_prompt=True)
                if len(ids)+64>cfg.max_seq_len: continue
                tokens=list(model.generate(torch.tensor([ids]),max_new_tokens=64,temperature=0,stop_ids=(tok.end_id,tok.user_id)))
                candidate={**row,'messages':row['messages'][:-1]+[{'role':'assistant','content':tok.decode(tokens)}]}
                if row.get('verification'):
                    checked+=1
                    try: correct+=bool(verify_auto(candidate))
                    except (ValueError,TypeError,KeyError): pass
            answer_accuracy[split]={'checked':checked,'correct':correct,'accuracy':correct/checked if checked else None}
    validated=(release_check and after<before and all(answer_accuracy.get(s,{}).get('checked',0)>=100 and
                (answer_accuracy[s]['accuracy'] or 0)>=.9 for s in ('val','test')))
    output.mkdir(parents=True,exist_ok=True)
    payload={'rank':rank,'state':model.expert_state(expert),'base_sha256':hashlib.sha256((model_dir/'base.pt').read_bytes()).hexdigest(),
             'tokenizer_sha256':hashlib.sha256((model_dir/'tokenizer.json').read_bytes()).hexdigest(),
             'dataset_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    temp=output/f'{expert}.tmp'; torch.save(payload,temp); os.replace(temp,output/f'{expert}.pt')
    report={'expert':expert,'state':'quality_validated' if validated else ('undersized_pilot_unvalidated' if undersized else 'pilot_unvalidated'),'steps':step,'rank':rank,
            'adapter_parameters':sum(p.numel() for p in parameters),'validation_loss_base':before,'validation_loss_adapter':after,
            'answer_accuracy':answer_accuracy,'seconds':time.perf_counter()-started,'peak_rss_bytes':peak,
            'source_model_changed':False,'router_retrained':False,
            'data_thresholds_met':not undersized,'split_examples':{s:len(v) for s,v in rows.items()},
            'split_families':{s:len(v) for s,v in groups.items()},
            'training_languages':sorted({row['language'] for row in rows['train']}),
            'note':'Candidate only. Keep paired with exact base/tokenizer hashes; router integration and regression checks are still required.'}
    write_json(output/f'{expert}.report.json',report)
    return report



# ──  data/persona_rules.py  ──
"""Tantra persona rules shared by the audit script and bulk filters (owner, 2026-10-19).

Mention being an AI only when the user asks about it; mention Atul Vij / Atulya AI only
when asked who made Tantra, its company or origin; never use service wording unless the
user brings it up. Each assistant turn is judged against the user turn right before it,
so one early question does not license every later reply.
"""
import re

ASKS_AI = re.compile(r'\b(ai|a\.i\.|human|person|people|real|robot|bot|chatbot|machine|program|alive|living|conscious|'
                     r'feelings|chatgpt|gpt|gemini|siri|alexa|claude|copilot|bard|llama|jarvis|model|computer|'
                     r'what (exactly )?are you|what is this|someone (reading|typing)|customer care|meet( up)?|'
                     r'turing|are you (him|her)|grandma|can we meet|pick up)\b', re.I)
ASKS_CREATOR = re.compile(r'\b(made|maker|creat\w*|built|build|develop\w*|coded|programm\w*|trained|invent\w*|designed|'
                          r'behind (you|tantra)|owns? you|owner|company|firm|startup|brand|organi[sz]ation|'
                          r'founder|founded|your dev|origin|atul|atulya|microsoft|amazon|apple|meta|openai|facebook|'
                          r'tata|reliance|government|university|credit|responsible for you|thank for you|boss|'
                          r'gave you life|into this world|your (father|parent|story)|raised you|your existence|'
                          r'mind behind|where (are|do) you (come )?from|who (are|r) (you|u)|introduce yourself|'
                          r'about (yourself|you)|who is this|identify yourself|(make|made|brought|built) you|(did|do) you come from|'
                          r'are you (from|by|made)|google|jarvis|chatgpt|gemini|siri|alexa|claude|copilot|bard)\b', re.I)
SAYS_AI = re.compile(r"\b(an ai|i'm an ai|i am an ai|ai assistant|language model|computer program|chatbot|"
                     r"i'm a bot|i am a bot|virtual (assistant|agent)|a piece of software)\b", re.I)
SAYS_CREATOR = re.compile(r'\b(atul vij|atulya( ai)?|atul)\b', re.I)
SERVICE = re.compile(r'\b(assistant|how can i help|how may i assist|as an ai language model)\b', re.I)
ASKS_SERVICE = re.compile(r'\bassistant\b', re.I)


def _previous_user(messages, i):
    for m in reversed(messages[:i]):
        if m['role'] == 'user':
            return m['content']
    return ''


def turn_violations(messages, i):
    """Problems in assistant turn i, judged against the user turn right before it."""
    if messages[i]['role'] != 'assistant':
        return []
    asked = _previous_user(messages, i)
    reply = messages[i]['content']; found = []
    if SAYS_AI.search(reply) and not ASKS_AI.search(asked): found.append('unasked AI mention')
    # Continuity: once the creator came up legitimately in this conversation, later turns may refer to him.
    earlier = any(m['role'] == 'assistant' and SAYS_CREATOR.search(m['content']) for m in messages[:i])
    if SAYS_CREATOR.search(reply) and not (ASKS_CREATOR.search(asked) or earlier): found.append('unasked creator mention')
    if SERVICE.search(reply) and not ASKS_SERVICE.search(asked): found.append('service wording')
    return found


def violations(messages):
    """Persona problems across every assistant turn, whatever the role order (system turns included)."""
    return [v for i, m in enumerate(messages) if m['role'] == 'assistant' for v in turn_violations(messages, i)]



# ──  data/auto_growth.py  ──
"""Bounded expert expansion: one isolated candidate per invocation."""
import argparse
import json
import os
from pathlib import Path
from Tantra import ROOT


def select_expert(manifest, attempted):
    specs = catalog()
    order = list(dict.fromkeys(PRIORITY + ['math', 'json_schema', 'csv_export'] + list(specs)))
    for name in order:
        spec = specs[name]
        if name.startswith('lang_') and name not in ('lang_hi', 'lang_en'):
            continue
        if any(language not in ('hi', 'en', 'hinglish') for language in spec['languages']):
            continue
        if name not in attempted and manifest['experts'].get(name, {}).get('state') == 'data_ready':
            return name
    return None


def run_one(bundle=DEFAULT_OUT, model_dir=ROOT/'Model', output=ROOT/'runs'/'expert-growth', steps=100):
    bundle, output = Path(bundle), Path(output)
    manifest = json.loads((bundle/'prepared'/'manifest.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive file creation prevents concurrent invocations training two candidates.
    lock = output/'growth.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(str(os.getpid()))
    try:
        attempted = {p.name for p in output.iterdir() if p.is_dir()}
        name = select_expert(manifest, attempted)
        if name is None:
            return {'state': 'blocked', 'reason': 'No unattempted data-ready expert in initial language scope'}
        candidate = output/name
        candidate.mkdir()  # Retain failed attempts too; retries require explicit review.
        try:
            result = train_one(bundle, name, model_dir, candidate/'candidate', steps=steps, rank=8, threads=4)
            event = {'expert': name, 'state': 'candidate_trained', 'deployed': False, 'result': result,
                     'next_gate': 'Independent answer evaluation, router macro-F1 >=0.90 and conversation regression checks'}
        except Exception as exc:
            event = {'expert': name, 'state': 'failed', 'deployed': False,
                     'error': type(exc).__name__ + ': ' + str(exc)}
            (candidate/'event.json').write_text(json.dumps(event, indent=2), encoding='utf-8')
            raise
        (candidate/'event.json').write_text(json.dumps(event, indent=2), encoding='utf-8')
        return event
    finally:
        lock.unlink()


def auto_growth_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--model-dir', type=Path, default=ROOT/'Model')
    parser.add_argument('--output', type=Path, default=ROOT/'runs'/'expert-growth')
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run_one(args.out, args.model_dir, args.output, args.steps), indent=2))


if __name__ == '__main__':
    main()
