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
- **Fix Commit**: [Pending Commit]
  1. Decoupled `checkpoint_best.pt` strictly to `is_new_best == True`.
  2. Enhanced `load_checkpoint` to restore `best_val_loss` from checkpoint metadata or fallback to existing `Model/Best/checkpoint_best.pt`.
- **Verification**: Tested `load_checkpoint` fallback and verified `checkpoint_best.pt` is only touched when `val_loss < historical_best`.
- **Status**: 🟢 **RESOLVED & VERIFIED**

---

## 📊 Evaluation & Verification Protocol

When evaluating architectural changes or sequence length transitions:
1. **Tokens as Ground Truth**: Metrics must be plotted against **Cumulative Tokens Processed**, not raw step count.
2. **Multi-Point Verification**: Plateaus are evaluated across **$\ge 3$ consecutive `VAL EVAL` blocks** before making directional decisions.
3. **Generalization Overfitting Guard**: `checkpoint_best.pt` is solely updated when held-out validation loss beats the global historical best.
