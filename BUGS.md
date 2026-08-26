# 🐞 Tantra-LLM Bug Bounty & Issue Tracking Log

This document serves as the formal register of identified bugs, their root cause analysis, fix commits, and live verification evidence.

---

## 📋 Bug Registry

### 🟢 BUG-001: Model Weight Freezing / Static Tensor Bug
- **Found**: 2026-08-25 (Colab run & early step analysis)
- **Evidence**: Loss remained static or fluctuated on random token distributions without updating internal weights.
- **Root Cause**: Quantization clamping and optimizer graph detach prevented gradients from flowing into base linear layers.
- **Fix Commit**: [`0076a08`](https://github.com/atulyaai/Tantra-LLM/commit/0076a08)
- **Verification**: Verified gradient norm $>0.0$ and backprop loss reduction across 81 unit tests.
- **Status**: 🟢 **RESOLVED & VERIFIED**

---

### 🟢 BUG-002: Frobenius Norm Explosions on CPU Quantization
- **Found**: 2026-08-25, step 2,206
- **Evidence**: Loss spiked to `9.4` and gradients oscillated violently when ternary BitNet scales exceeded float32 thresholds.
- **Root Cause**: Unclamped scaling factors in `DynamicScaleNorm` and non-standard Frobenius normalization.
- **Fix Commit**: [`24ea2cf`](https://github.com/atulyaai/Tantra-LLM/commit/24ea2cf)
- **Verification**: Zero explosions across 18,000 continuous training steps.
- **Status**: 🟢 **RESOLVED & VERIFIED**

---

### 🟢 BUG-003: Training EMA Loss Masking True Validation Loss in `checkpoint_best.pt`
- **Found**: 2026-08-26, step 18,000 analysis
- **Evidence**: `checkpoint_best.pt` was saved with `best_loss=3.6406` when single-batch training EMA dipped, despite true held-out validation loss remaining at `~6.52`.
- **Root Cause**: `self.best_loss` in `train.py` was being updated by `self.ema_loss` on every micro-batch step rather than strictly on held-out validation evaluations.
- **Fix Commit**: [`98d467c`](https://github.com/atulyaai/Tantra-LLM/commit/98d467c)
- **Verification**: `self.best_loss` is strictly bound to `avg_val_loss` during `VAL EVAL`.
- **Status**: 🟢 **RESOLVED & VERIFIED**

---

### 🟢 BUG-004: Periodic Checkpoint Archive Overwriting `checkpoint_best.pt` & Unrestored `best_val_loss` on Resume
- **Found**: 2026-08-26, step 18,500 resume log
- **Evidence**: `checkpoint_best.pt` was saved at step 18,500 with `Val Loss: 6.5756`, which was higher than previous best `6.5013` (step 17,000).
- **Root Cause**:
  1. `main.py` had an archive condition `if is_new_best or step % (eval_every * 4) == 0:` that erroneously overwrote `best_ckpt` periodically regardless of validation loss improvement.
  2. Legacy checkpoints without explicit `best_val_loss` initialized `self.best_val_loss = float('inf')`, making any first validation evaluation count as a "new best".
- **Fix Commit**: [`7e2aac2`](https://github.com/atulyaai/Tantra-LLM/commit/7e2aac2)
- **Verification**: Tested `load_checkpoint` fallback and verified `checkpoint_best.pt` is only touched when `val_loss < historical_best`.
- **Status**: 🟢 **RESOLVED & VERIFIED**

---

### 🟢 BUG-005: Unbound `is_new_best` NameError in `main.py` `eval_callback`
- **Found**: 2026-08-26, deep code audit
- **Evidence**: `is_new_best` was referenced as a local variable inside `eval_callback()` in `main.py` without being passed in as an argument or resolved via trainer instance.
- **Root Cause**: Scope mismatch following the decoupling of periodic checkpoint archives from validation loss improvements.
- **Fix Commit**: [`2922d79`](https://github.com/atulyaai/Tantra-LLM/commit/2922d79)
- **Verification**: Replaced with safe attribute resolution `getattr(trainer, "is_new_best", False)` and verified callback execution.
- **Status**: 🟢 **RESOLVED & VERIFIED**

---

### 🟢 BUG-006: Optimizer Momentum & LR Scheduler Reset on Same-Stage Resume
- **Found**: 2026-08-27, Colab GPU resume log (Step 19,500 $\rightarrow$ 19,501)
- **Evidence**: Loss jumped from `6.3555` to `9.02` immediately upon resuming; LR was reset to `2.00e-06` (step 1 of warmup); text generation degraded due to zeroed momentum vectors.
- **Root Cause**: `main.py` unconditionally re-initialized `trainer.optimizer` and `trainer.scheduler` whenever `--training-stage sft` was specified, discarding the restored AdamW momentum buffers ($m_t, v_t$) and cosine decay position.
- **Fix Commit**: [Current Commit]
- **Verification**: Saved and restored `training_stage` in checkpoint state; only trigger re-initialization on genuine stage transition (`prev_stage != "sft"`). Verified via `test_optimizer_and_scheduler_continuity_on_resume` in `Tests/test_optimizers.py`.
- **Status**: 🟢 **RESOLVED & VERIFIED**

---

## 📊 Evaluation & Verification Protocol



When evaluating architectural changes or sequence length transitions:
1. **Tokens as Ground Truth**: Metrics must be plotted against **Cumulative Tokens Processed**, not raw step count.
2. **Multi-Point Verification**: Plateaus are evaluated across **$\ge 3$ consecutive `VAL EVAL` blocks** before making directional decisions.
3. **Generalization Overfitting Guard**: `checkpoint_best.pt` is solely updated when held-out validation loss beats the global historical best.
