"""
Tantra/train.py — The training loop.

NeuroTrainer does four things:
  train_step()      one micro-batch: forward, loss, backward, (every N micro-batches) optimizer step
  fit()             the loop: logging, validation, probe/eval callback, checkpoints, optional auto-growth
  save_checkpoint() / load_checkpoint()   full-precision, atomic, resumable (optimizer + step + tokens)
  train_dpo()       preference tuning from real chosen/rejected pairs

Learning rate = warmup then cosine decay to 10%, computed from the global step,
so stopping and resuming never resets or jumps the schedule.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Tantra.dataset import IGNORE_INDEX
from Tantra.utils import get_logger, safe_load_checkpoint, unwrap_model

log = get_logger("tantra.train")

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Model")
STATUS_PATH = os.path.join(_MODEL_DIR, "training_status.json")
STOP_FILE = os.path.join(_MODEL_DIR, "STOP")   # create this file to stop training cleanly (WebUI does it)


# ── Optimizer & schedule ─────────────────────────────────────────────────────

class Lion(torch.optim.Optimizer):
    """Lion (sign momentum). One state buffer instead of AdamW's two. Use ~1/5 of the AdamW LR."""

    def __init__(self, params, lr: float = 3e-5, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                st = self.state[p]
                if not st:
                    st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                m = st["exp_avg"]
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(m.mul(b1).add_(g, alpha=1 - b1).sign_().to(p.dtype), alpha=-group["lr"])
                m.mul_(b2).add_(g, alpha=1 - b2)


def param_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    """No weight decay on norms, biases and gates."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 or "norm" in name or name.endswith("bias") else decay).append(p)
    if not decay and not no_decay:
        raise ValueError("No trainable parameters.")
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(name: str, groups: Any, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    name = (name or "adamw").lower()
    if name == "lion":
        return Lion(groups, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay,
                             fused=torch.cuda.is_available())


def lr_at(step: int, peak: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return peak * max(0.05, (step + 1) / max(1, warmup))
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def fmt_duration(seconds: float) -> str:
    s = int(max(0, seconds))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{d}d {h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"


def _logits(out: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    first = out[0] if isinstance(out, (tuple, list)) else out
    if isinstance(first, (tuple, list)):
        return first[0], first[1]
    return first, None


# ── Trainer ──────────────────────────────────────────────────────────────────

class NeuroTrainer:
    def __init__(self, model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.1,
                 optimizer_name: str = "adamw", total_steps: int = 20000, warmup_steps: int = 500,
                 grad_accumulation_steps: int = 1, use_mtp_loss: bool = False, mtp_loss_weight: float = 0.3,
                 max_grad_norm: float = 1.0, label_smoothing: float = 0.0, **_ignored: Any):
        self.model = model
        self.device = next(model.parameters()).device
        self.lr, self.weight_decay, self.optimizer_name = float(lr), float(weight_decay), optimizer_name
        self.total_steps, self.warmup_steps = int(total_steps), int(warmup_steps)
        self.grad_accumulation_steps = max(1, int(grad_accumulation_steps))
        self.use_mtp_loss, self.mtp_loss_weight = use_mtp_loss, float(mtp_loss_weight)
        self.max_grad_norm = float(max_grad_norm)
        self.label_smoothing = float(label_smoothing)
        self.optimizer = build_optimizer(optimizer_name, param_groups(model, self.weight_decay), self.lr, self.weight_decay)

        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        self.step_count = 0
        self.total_tokens = 0
        self.total_training_seconds = 0.0
        self.best_val_loss = float("inf")
        self.ema_loss: Optional[float] = None
        self.last_validation: Dict[str, float] = {}
        self._micro = 0
        self._set_lr()

    # ── core step ──
    def _set_lr(self) -> float:
        lr = lr_at(self.step_count, self.lr, self.warmup_steps, self.total_steps)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def refresh_optimizer(self) -> None:
        """Register parameters added by growth / category layers; keeps existing momentum."""
        known = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
        new = [p for p in self.model.parameters() if p.requires_grad and id(p) not in known]
        if new:
            self.optimizer.add_param_group({"params": [p for p in new if p.ndim >= 2], "weight_decay": self.weight_decay})
            self.optimizer.add_param_group({"params": [p for p in new if p.ndim < 2], "weight_decay": 0.0})
            self._set_lr()

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        self.model.train()
        x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            main, mtp = _logits(self.model(token_ids=x, return_mtp=self.use_mtp_loss, use_latent_reasoning=False))
            flat_y = y.reshape(-1)
            keep = flat_y != IGNORE_INDEX
            flat = main.reshape(-1, main.size(-1))
            if keep.any():
                loss = F.cross_entropy(flat[keep].float(), flat_y[keep], label_smoothing=self.label_smoothing)
            else:
                loss = flat.sum() * 0.0
            raw = unwrap_model(self.model)
            if hasattr(raw, "get_aux_loss"):
                loss = loss + raw.get_aux_loss()
            if mtp is not None and self.mtp_loss_weight > 0 and y.size(1) > 1:
                y2 = y[:, 1:].reshape(-1)
                k2 = y2 != IGNORE_INDEX
                if k2.any():
                    m2 = mtp[:, :-1].reshape(-1, mtp.size(-1))[k2].float()
                    loss = loss + self.mtp_loss_weight * F.cross_entropy(m2, y2[k2])

        if not torch.isfinite(loss):
            log.warning("Non-finite loss — batch skipped.")
            self.optimizer.zero_grad(set_to_none=True)
            self._micro = 0
            return {"loss": float("nan"), "stepped": False}

        self.scaler.scale(loss / self.grad_accumulation_steps).backward()
        self._micro += 1
        n_tok = int(keep.sum())
        self.total_tokens += x.numel()
        with torch.no_grad():
            acc = float((flat[keep].argmax(-1) == flat_y[keep]).float().mean()) * 100 if n_tok else 0.0

        stepped, grad_norm = False, 0.0
        if self._micro >= self.grad_accumulation_steps:
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self._micro = 0
            self.step_count += 1
            self._set_lr()
            stepped = True

        lv = float(loss.detach())
        self.ema_loss = lv if self.ema_loss is None else 0.98 * self.ema_loss + 0.02 * lv
        return {"loss": lv, "acc": acc, "grad_norm": grad_norm, "stepped": stepped}

    # ── validation ──
    def evaluate(self, val_loader: Iterable, max_batches: int = 50) -> Dict[str, float]:
        from Tantra.eval_suite import validation_metrics
        m = validation_metrics(unwrap_model(self.model), val_loader, max_batches=max_batches)
        self.last_validation = m
        return m

    # ── main loop ──
    def fit(self, train_loader: Iterable, max_steps: int, log_every: int = 50, eval_every: int = 250,
            val_loader: Optional[Iterable] = None, val_batches: int = 50,
            on_eval: Optional[Callable[[int, Dict[str, float]], None]] = None,
            checkpoint_every: int = 1000, on_checkpoint: Optional[Callable[[int], None]] = None,
            growth: Any = None, early_stopping_patience: int = 0) -> None:
        self.total_steps = max(self.total_steps, max_steps)
        start_step, t0 = self.step_count, time.time()
        session_tokens0 = self.total_tokens
        window: List[float] = []
        accs: List[float] = []
        bad_evals = 0
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
        self._status("running", max_steps, t0, start_step, session_tokens0)
        log.info(f"Training from step {self.step_count:,} to {max_steps:,} "
                 f"(effective batch = {self.grad_accumulation_steps} micro-batches)")
        for x, y in train_loader:
            if self.step_count >= max_steps:
                break
            r = self.train_step(x, y)
            if math.isfinite(r["loss"]):
                window.append(r["loss"])
                accs.append(r["acc"])
            if not r["stepped"]:
                continue
            s = self.step_count

            if s % log_every == 0 or s == start_step + 1:
                done = s - start_step
                spent = time.time() - t0
                eta = spent / max(done, 1) * (max_steps - s)
                tok_s = (self.total_tokens - session_tokens0) / max(spent, 1e-6)
                log.info(f"step {s:,}/{max_steps:,} | loss {sum(window)/max(len(window),1):.4f} | "
                         f"acc {sum(accs)/max(len(accs),1):.1f}% | lr {self.optimizer.param_groups[0]['lr']:.2e} | "
                         f"grad {r['grad_norm']:.2f} | {tok_s:,.0f} tok/s | ETA {fmt_duration(eta)}")
                self._status("running", max_steps, t0, start_step, session_tokens0,
                             loss=sum(window) / max(len(window), 1), accuracy=sum(accs) / max(len(accs), 1))
                window.clear()
                accs.clear()

            if growth is not None and self.ema_loss is not None:
                if growth.observe(float(self.ema_loss), unwrap_model(self.model), optimizer=None):
                    self.refresh_optimizer()

            if eval_every and s % eval_every == 0:
                metrics: Dict[str, float] = {}
                if val_loader is not None:
                    metrics = self.evaluate(val_loader, val_batches)
                    improved = metrics["loss"] < self.best_val_loss - 1e-4
                    if improved:
                        self.best_val_loss = metrics["loss"]
                        bad_evals = 0
                    else:
                        bad_evals += 1
                    metrics["is_best"] = improved
                    log.info(f"  [val @ {s:,}] loss {metrics['loss']:.4f} | ppl {metrics['perplexity']:.1f} | "
                             f"top1 {metrics['top1_accuracy_percent']:.1f}% | top5 {metrics['top5_accuracy_percent']:.1f}%"
                             f"{'  (new best)' if improved else ''}")
                if metrics:
                    self._val_history().append({"step": s, "loss": metrics["loss"],
                                                "top1": metrics["top1_accuracy_percent"]})
                if on_eval is not None:
                    on_eval(s, metrics)
                if early_stopping_patience and bad_evals >= early_stopping_patience:
                    log.warning(f"Validation did not improve for {bad_evals} evals — stopping early.")
                    break

            if checkpoint_every and s % checkpoint_every == 0 and on_checkpoint is not None:
                on_checkpoint(s)

            if os.path.exists(STOP_FILE):
                os.remove(STOP_FILE)
                log.warning("Stop requested — saving and exiting.")
                break

        self.total_training_seconds += time.time() - t0
        self._status("complete" if self.step_count >= max_steps else "stopped", max_steps, t0, start_step, session_tokens0)

    # ── live status for the WebUI (Model/training_status.json) ──
    def _load_history(self) -> None:
        """Continue the loss curves of an earlier session (points up to the current step)."""
        self._hist: Dict[str, List[dict]] = {"train": [], "val": []}
        try:
            with open(STATUS_PATH, encoding="utf-8") as f:
                old = json.load(f).get("history") or {}
            for k in self._hist:
                self._hist[k] = [p for p in old.get(k, []) if p.get("step", 1e18) <= self.step_count]
        except (OSError, ValueError, AttributeError):
            pass

    def _val_history(self) -> List[dict]:
        if not hasattr(self, "_hist"):
            self._load_history()
        return self._hist["val"]

    def _status(self, state: str, target: int, t0: float, start_step: int, tok0: int, **extra: Any) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return
        if not hasattr(self, "_hist"):
            self._load_history()
        spent = time.time() - t0
        done = max(self.step_count - start_step, 1)
        tok_s = round((self.total_tokens - tok0) / max(spent, 1e-6), 1)
        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else None
        if extra.get("loss") is not None:
            self._last_train = {"loss": extra["loss"], "accuracy": extra.get("accuracy")}
            self._hist["train"].append({"step": self.step_count, "loss": round(extra["loss"], 4),
                                        "acc": round(extra.get("accuracy") or 0.0, 2), "lr": lr, "tok_s": tok_s})
        for k, cap in (("train", 2000), ("val", 500)):   # keep the file small
            if len(self._hist[k]) > cap:
                self._hist[k] = self._hist[k][::2]
        data = {"status": state, "step": self.step_count, "target_steps": target, "start_step": start_step,
                "ema_loss": self.ema_loss, "total_tokens": self.total_tokens, "lr": lr,
                "tok_s": tok_s, "elapsed": fmt_duration(spent),
                "eta": fmt_duration(spent / done * max(target - self.step_count, 0)),
                "validation": self.last_validation, "best_val_loss": self.best_val_loss
                if math.isfinite(self.best_val_loss) else None, "pid": os.getpid(),
                "updated_at": time.time(), **getattr(self, "_last_train", {}), **extra,
                "history": self._hist}
        if data.get("loss") is not None:
            data["ppl"] = round(math.exp(min(20.0, data["loss"])), 2)
        try:
            os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
            with open(STATUS_PATH + ".tmp", "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(STATUS_PATH + ".tmp", STATUS_PATH)
        except OSError:
            pass

    # ── checkpoints ──
    def save_checkpoint(self, path: str, save_optimizer: bool = True, extra: Optional[dict] = None) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        raw = unwrap_model(self.model)
        if hasattr(raw, "config"):
            raw.config.block.num_layers = len(raw.layers)
        data = {
            "model_state_dict": {k: v.detach().cpu() for k, v in raw.state_dict().items()},  # full fp32
            "config": copy.deepcopy(getattr(raw, "config", None)),
            "use_mtp": bool(getattr(raw, "use_mtp", False)),
            "step_count": self.step_count,
            "total_tokens": self.total_tokens,
            "total_training_seconds": self.total_training_seconds,
            "best_val_loss": self.best_val_loss,
            "ema_loss": self.ema_loss,
            "last_validation": dict(self.last_validation),
            "peak_lr": self.lr, "warmup_steps": self.warmup_steps, "total_steps": self.total_steps,
            **(extra or {}),
        }
        if save_optimizer:
            data["optimizer_state_dict"] = self.optimizer.state_dict()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        torch.save(data, tmp)
        os.replace(tmp, path)
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({"step_count": self.step_count, "num_layers": len(getattr(raw, "layers", [])),
                       "total_tokens": self.total_tokens, "best_val_loss": self.best_val_loss,
                       "last_validation": self.last_validation,
                       "training_time": fmt_duration(self.total_training_seconds)}, f, indent=2)
        log.info(f"Checkpoint saved: {path} (step {self.step_count:,})")

    def load_checkpoint(self, path: str, reset_optimizer: bool = False) -> dict:
        ckpt = safe_load_checkpoint(path, map_location="cpu")
        raw = unwrap_model(self.model)
        state = {k: v for k, v in ckpt.get("model_state_dict", ckpt).items()}
        missing, unexpected = raw.load_state_dict(state, strict=False)
        if missing or unexpected:
            log.warning(f"Checkpoint/model mismatch: {len(missing)} missing, {len(unexpected)} unexpected tensors "
                        f"(e.g. {(missing or unexpected)[:3]})")
        self.step_count = int(ckpt.get("step_count", 0))
        self.total_tokens = int(ckpt.get("total_tokens", 0))
        self.total_training_seconds = float(ckpt.get("total_training_seconds", 0.0))
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        self.ema_loss = ckpt.get("ema_loss")
        self.last_validation = dict(ckpt.get("last_validation", {}) or {})
        if not reset_optimizer and "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                for st in self.optimizer.state.values():
                    for k, v in st.items():
                        if torch.is_tensor(v):
                            st[k] = v.to(self.device)
            except (ValueError, KeyError) as exc:
                log.warning(f"Optimizer state not restored ({exc}); continuing with fresh momentum.")
        self._set_lr()
        log.info(f"Resumed {path} at step {self.step_count:,} ({self.total_tokens/1e6:.1f}M tokens seen)")
        return ckpt

    # ── DPO ──
    @staticmethod
    def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        lab = labels[:, 1:]
        mask = lab != IGNORE_INDEX
        tok = torch.gather(logp, 2, lab.clamp(min=0).unsqueeze(2)).squeeze(2)
        return (tok * mask).sum(1)

    def train_dpo(self, loader: Iterable, max_steps: int, beta: float = 0.1, log_every: int = 25,
                  on_checkpoint: Optional[Callable[[int], None]] = None, checkpoint_every: int = 250) -> None:
        ref = copy.deepcopy(unwrap_model(self.model)).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        target = self.step_count + max_steps
        self.total_steps = max(self.total_steps, target)
        it = iter(loader)
        while self.step_count < target:
            self.optimizer.zero_grad(set_to_none=True)
            stats = []
            for _ in range(self.grad_accumulation_steps):
                try:
                    b = next(it)
                except StopIteration:
                    it = iter(loader)
                    b = next(it)
                b = {k: v.to(self.device) for k, v in b.items()}
                pc = self.sequence_logprob(_logits(self.model(b["chosen_input_ids"]))[0], b["chosen_labels"])
                pr = self.sequence_logprob(_logits(self.model(b["rejected_input_ids"]))[0], b["rejected_labels"])
                with torch.no_grad():
                    rc = self.sequence_logprob(_logits(ref(b["chosen_input_ids"]))[0], b["chosen_labels"])
                    rr = self.sequence_logprob(_logits(ref(b["rejected_input_ids"]))[0], b["rejected_labels"])
                margin = beta * ((pc - pr) - (rc - rr))
                loss = -F.logsigmoid(margin).mean()
                (loss / self.grad_accumulation_steps).backward()
                stats.append((float(loss), float((margin > 0).float().mean())))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.step_count += 1
            self._set_lr()
            if self.step_count % log_every == 0:
                log.info(f"dpo step {self.step_count:,}/{target:,} | loss {sum(s[0] for s in stats)/len(stats):.4f} | "
                         f"chosen wins {100*sum(s[1] for s in stats)/len(stats):.0f}%")
            if on_checkpoint and checkpoint_every and self.step_count % checkpoint_every == 0:
                on_checkpoint(self.step_count)
