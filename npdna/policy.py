"""Adaptive training policy utilities (train-less-but-smarter): step-adaptive
LR schedule with warmup + cosine decay, adaptive grad-clip scaling, checkpoint
and eval cadence, and an EMA self-distillation teacher trained from the model
on its own past, smoothed self (no external teacher model required)."""
from __future__ import annotations
import math
import re
import copy
import torch
from torch import nn


class AdaptiveTrainingPolicy:
    """Per-step LR multiplier, grad-clip scaling, and log/save/eval cadence."""

    def __init__(self, target_steps: int, warmup_ratio: float = 0.1,
                 base_lr: float = 3e-4, base_clip: float = 1.0,
                 log_every: int = 50, save_every: int = 500,
                 eval_every: int = 500, latest_every: int = 100):
        self.target_steps = max(1, int(target_steps))
        self.warmup_steps = max(1, int(self.target_steps * max(0.0, min(1.0, warmup_ratio))))
        self.base_lr = float(base_lr)
        self.base_clip = float(base_clip)
        self.log_every = max(1, int(log_every))
        self.save_every = max(1, int(save_every))
        self.eval_every = max(1, int(eval_every))
        self.latest_every = max(1, int(latest_every))

    def lr_factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return float((step + 1) / self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.target_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return float(0.5 * (1.0 + math.cos(math.pi * progress)))

    def lr_at(self, step: int) -> float:
        return self.base_lr * self.lr_factor(step)

    def clip_at(self, step: int) -> float:
        return self.base_clip * max(0.25, self.lr_factor(step))

    def should_log(self, step: int) -> bool:
        return step == 1 or step % self.log_every == 0

    def should_save_best(self, step: int) -> bool:
        return step % self.save_every == 0 or step == self.target_steps

    def should_save_latest(self, step: int) -> bool:
        return step % self.latest_every == 0 or step == self.target_steps

    def should_eval(self, step: int) -> bool:
        return step % self.eval_every == 0 or step == self.target_steps


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """In-place EMA update:  ema <- decay*ema + (1-decay)*student."""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            if ema_p.shape == p.shape:
                ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            if ema_b.shape == b.shape:
                ema_b.copy_(b.detach())


def compute_self_distill_loss(student_logits: torch.Tensor,
                              teacher_logits: torch.Tensor,
                              temperature: float = 2.0) -> torch.Tensor:
    """KL(student || teacher) over softmaxed temperature logits."""
    s = student_logits.float()
    t = teacher_logits.float()
    d = min(s.size(-1), t.size(-1))
    s = s[..., :d]
    t = t[..., :d]
    s_log = torch.log_softmax(s / temperature, dim=-1)
    t_p = torch.softmax(t / temperature, dim=-1)
    return nn.functional.kl_div(s_log, t_p, reduction="batchmean") * (temperature ** 2)


class SelfDistillationTeacher:
    """EMA shadow of the student that mentors it via KL self-distillation.

    Usage:
        teacher = SelfDistillationTeacher(student_model, decay=0.999, temperature=2.0,
                                          warmup_steps=200)
        for step in range(total):
            logits, _ = student(input_ids)
            distill_loss = teacher.distill(logits, input_ids, lambda m, x: m(x)) or 0.0
            loss = ce_loss + distill_weight * distill_loss
            (loss / grad_accum).backward()
            ...
            scaler.step(opt); scaler.update()
            teacher.step(student)   # refresh EMA after the real optimizer step
    """

    def __init__(self, model: nn.Module, decay: float = 0.999,
                 temperature: float = 2.0, warmup_steps: int = 200):
        self.decay = float(decay)
        self.temperature = float(temperature)
        self.warmup_steps = int(warmup_steps)
        self.model = copy.deepcopy(model)
        self.model.eval()

    def step(self, student: nn.Module) -> None:
        update_ema(self.model, student, self.decay)

    def active(self, step: int) -> bool:
        return step >= self.warmup_steps

    def distill(self, student_logits: torch.Tensor, input_ids: torch.Tensor,
                forward_fn) -> torch.Tensor:
        with torch.no_grad():
            ema_logits, _ = forward_fn(self.model, input_ids)
        return compute_self_distill_loss(student_logits, ema_logits, self.temperature)


_VOCAB_PREFIXES = ("Ġ", "▁", "##", "Ċ", "Ď")


def _vocab_token_norm(tok: str) -> str:
    """Normalize a BPE/WordPiece token string so tokens with different prefix
    markers but the same surface string compare equal across tokenizers."""
    t = tok
    for p in _VOCAB_PREFIXES:
        if t.startswith(p):
            t = t[len(p):]
    return t.strip()


def _tokenizer_vocab(tokenizer) -> dict:
    """Return {token_str: id} for either a HuggingFace-style tokenizer
    (get_vocab()) or the native AtulyaTokenizer (token_to_id)."""
    if hasattr(tokenizer, "get_vocab"):
        v = tokenizer.get_vocab()
        if v:
            return v
    if hasattr(tokenizer, "token_to_id") and tokenizer.token_to_id:
        return dict(tokenizer.token_to_id)
    raise ValueError(
        f"Unsupported tokenizer for vocab alignment: {type(tokenizer).__name__} "
        "(needs get_vocab() or token_to_id)"
    )


def build_vocab_map(student_tokenizer, teacher_tokenizer) -> torch.Tensor:
    """Return a LongTensor of shape (V_stu,): for each student token id, the
    teacher token id it corresponds to (matched by normalized surface string),
    or -1 if the token has no counterpart in the teacher's vocabulary.

    This is what makes cross-vocabulary distillation correct: instead of
    comparing logits at the same *index* across two different vocabularies (the
    old behavior, which is meaningless), we align by actual token string;
    student tokens with no teacher match receive no distillation gradient."""
    stu = _tokenizer_vocab(student_tokenizer)
    tea = _tokenizer_vocab(teacher_tokenizer)
    tea_by_norm: dict[str, int] = {}
    for s, i in tea.items():
        tea_by_norm.setdefault(_vocab_token_norm(s), i)
    out = torch.full((len(stu),), -1, dtype=torch.long)
    for s, i in stu.items():
        m = tea_by_norm.get(_vocab_token_norm(s))
        if m is not None:
            out[i] = m
    return out
