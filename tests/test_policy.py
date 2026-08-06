import math
import torch
from torch import nn
import pytest

from npdna.policy import (
    AdaptiveTrainingPolicy,
    SelfDistillationTeacher,
    build_vocab_map,
    compute_self_distill_loss,
    update_ema,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        out = self.lin(x)
        return out, torch.zeros(1)


def make_policy(steps=1000):
    return AdaptiveTrainingPolicy(target_steps=steps, warmup_ratio=0.1,
                                  base_lr=1.0, base_clip=1.0)


# ---- AdaptiveTrainingPolicy ----

def test_warmup_linear_ramp_to_one():
    p = make_policy(steps=1000)
    assert p.warmup_steps == 100
    f0 = p.lr_factor(0)
    f99 = p.lr_factor(99)
    f100 = p.lr_factor(100)
    assert f0 < f99
    assert f99 <= 1.0
    assert abs(f100 - 1.0) < 1e-6


def test_cosine_decay_to_zero_at_end():
    p = make_policy(steps=1000)
    assert abs(p.lr_factor(1000) - 0.0) < 1e-9      # exact zero at the configured end
    assert p.lr_factor(999) < 1e-3                  # near-zero at the last training step
    assert abs(p.lr_at(999) - 0.0) < 1e-2
    mid = p.lr_factor(549)
    assert 0.0 < mid < 1.0


def test_lr_monotonic_non_increasing_after_warmup():
    p = make_policy(steps=1000)
    prev = p.lr_factor(100)
    for s in range(101, 1000):
        cur = p.lr_factor(s)
        assert cur <= prev + 1e-9
        prev = cur


def test_clip_scales_with_lr_factor():
    p = make_policy(steps=1000)
    c_warm = p.clip_at(0)
    c_end = p.clip_at(999)
    assert c_end <= c_warm
    assert c_end >= 0.25 * p.base_clip
    assert c_warm <= p.base_clip


def test_cadence_flags():
    p = make_policy(steps=1000)
    assert p.should_log(1)
    assert not p.should_log(2)
    assert p.should_log(50)
    assert p.should_save_latest(500)
    assert p.should_save_latest(1000)
    assert p.should_save_best(1000)
    assert p.should_eval(1000)
    assert not p.should_eval(7)


# ---- update_ema ----

def test_update_ema_decay_zero_copies_student():
    student = nn.Linear(4, 4)
    ema = nn.Linear(4, 4)
    with torch.no_grad():
        ema.weight.copy_(torch.zeros(4, 4))
    update_ema(ema, student, decay=0.0)
    assert torch.allclose(ema.weight, student.weight)


def test_update_ema_decay_one_keeps_ema():
    student = nn.Linear(4, 4)
    ema = nn.Linear(4, 4)
    before = ema.weight.detach().clone()
    update_ema(ema, student, decay=1.0)
    assert torch.allclose(ema.weight, before)


def test_update_ema_interpolation():
    student = nn.Linear(3, 3)
    ema = nn.Linear(3, 3)
    with torch.no_grad():
        ema.weight.copy_(torch.zeros(3, 3))
        student.weight.copy_(torch.ones(3, 3))
    update_ema(ema, student, decay=0.5)
    # ema = 0.5*0 + 0.5*1 = 0.5 everywhere
    assert torch.allclose(ema.weight, torch.full((3, 3), 0.5))


# ---- compute_self_distill_loss ----

def test_distill_loss_zero_when_identical():
    m = TinyModel()
    logits, _ = m(torch.randn(2, 8))
    loss = compute_self_distill_loss(logits, logits, temperature=2.0)
    assert loss.item() < 1e-4


def test_distill_loss_finite_and_positive_when_different():
    a = torch.randn(4, 8)
    b = torch.randn(4, 8)
    loss = compute_self_distill_loss(a, b, temperature=2.0)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


# ---- SelfDistillationTeacher ----

def test_teacher_inactive_before_warmup():
    m = TinyModel()
    t = SelfDistillationTeacher(m, decay=0.999, warmup_steps=10)
    assert t.active(5) is False
    assert t.active(10) is True


def test_teacher_step_updates_ema():
    student = TinyModel()
    teacher = SelfDistillationTeacher(student, decay=0.0, warmup_steps=0)
    with torch.no_grad():
        student.lin.weight.copy_(torch.ones(8, 8))
    teacher.step(student)
    assert torch.allclose(teacher.model.lin.weight, student.lin.weight)


def test_teacher_distill_runs_and_is_finite():
    student = TinyModel()
    student.eval()
    teacher = SelfDistillationTeacher(student, decay=0.999, warmup_steps=0)
    with torch.no_grad():
        student.lin.weight.add_(torch.randn_like(student.lin.weight))  # non-uniform shift
    x = torch.randn(2, 8)
    s_logits, _ = student(x)
    loss = teacher.distill(s_logits, x, lambda m, inp: m(inp))
    assert torch.isfinite(loss)
    assert loss.item() > 0.0  # student != its frozen EMA shadow


# ---- build_vocab_map ----

class _FakeTok:
    def __init__(self, vocab):
        self.vocab = vocab  # {token_str: id}

    def get_vocab(self):
        return self.vocab


def test_vocab_map_identity_when_same_vocab():
    v = {"a": 0, "b": 1, "the": 2, "Ġworld": 3}
    m = build_vocab_map(_FakeTok(v), _FakeTok(v))
    assert torch.equal(m, torch.tensor([0, 1, 2, 3]))


def test_vocab_map_missing_tokens_map_to_minus_one():
    m = build_vocab_map(_FakeTok({"a": 0, "b": 1}), _FakeTok({"c": 0, "d": 1}))
    assert torch.all(m < 0)


def test_vocab_map_partial_overlap_and_prefix_normalization():
    stu = {"a": 0, "the": 1, "cat": 2}
    tea = {"Ġthe": 0, "Ġa": 1, "other": 2}
    m = build_vocab_map(_FakeTok(stu), _FakeTok(tea))
    # student 'a' -> teacher 'Ġa' (prefix-stripped 'a'), student 'the' -> 'Ġthe',
    # student 'cat' -> no match (-1)
    assert m[0].item() == 1
    assert m[1].item() == 0
    assert m[2].item() == -1


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
