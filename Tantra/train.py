"""
tantra/train.py — Training pipeline for NeuroCore models.
"""
from __future__ import annotations

import os
import math
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Iterable, Tuple

from Tantra.utils import get_logger
from Tantra.evolution import SelfRepairEngine

log = get_logger(__name__)


def generate_synthetic_batch(vocab_size: int = 32000, batch_size: int = 2, seq_len: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic token sequences and auto-regressive targets."""
    vocab_size = max(2, vocab_size)
    batch_size = max(1, batch_size)
    seq_len = max(1, seq_len)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.roll(x, -1, dims=-1)
    y[:, -1] = 0
    return x, y


class NeuroTrainer:
    """Minimal, robust trainer for NeuroCore models."""

    def __init__(self, model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.01):
        self.model = model
        self.device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        log.info(f"  NeuroTrainer initialized on device: {self.device} (type={self.device.type})")
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.step_count = 0
        self.best_loss = float('inf')
        self.total_tokens = 0
        self._start_time = time.perf_counter()

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Single training step with bfloat16 autocast and MTP loss."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        x, y = x.to(self.device), y.to(self.device)
        
        # Use autocast based on actual model device, not global CUDA query
        device_type = self.device.type if self.device.type in ('cuda', 'mps') else 'cpu'
        autocast_enabled = device_type != 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
            out = self.model(x, return_mtp=True)
            if isinstance(out[0], tuple):
                logits_main, logits_mtp = out[0]
            else:
                logits_main = out[0]
                logits_mtp = None

            # Numerically stable logits clamping
            logits_flat = torch.clamp(logits_main.view(-1, logits_main.size(-1)), -50.0, 50.0)
            y_flat = torch.clamp(y.view(-1), 0, logits_main.size(-1) - 1)

            loss = self.criterion(logits_flat, y_flat)

            # Auxiliary MTP Loss (Multi-Token Prediction)
            if logits_mtp is not None and y.size(1) > 1:
                logits_mtp_flat = torch.clamp(logits_mtp[:, :-1, :].reshape(-1, logits_mtp.size(-1)), -50.0, 50.0)
                y_mtp_flat = torch.clamp(y[:, 1:].reshape(-1), 0, logits_mtp.size(-1) - 1)
                mtp_loss = self.criterion(logits_mtp_flat, y_mtp_flat)
                loss = loss + 0.25 * mtp_loss

        if torch.isnan(loss) or torch.isinf(loss):
            return 0.0, 0.0, 0.0, 0.0

        # Calculate Top-1 Accuracy
        with torch.no_grad():
            preds = logits_flat.argmax(dim=-1)
            correct = (preds == y_flat).float().sum()
            total = y_flat.numel()
            accuracy = (correct / max(total, 1)).item() * 100.0
            ppl = math.exp(min(loss.item(), 20.0))

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
        self.optimizer.step()

        self.step_count += 1
        self.total_tokens += x.numel()
        if loss < self.best_loss:
            self.best_loss = loss
        return loss.item(), accuracy, ppl, grad_norm

    def train_dataset(self, data_stream: Iterable[Tuple[torch.Tensor, torch.Tensor]], max_steps: int = 100, log_every: int = 1, eval_every: int = 0, eval_callback = None) -> list[float]:
        """Train over an iterable dataset stream (e.g. JSONLDataset)."""
        log.info(f"Starting dataset pre-training run (target steps: {max_steps})...")
        
        from Tantra.tokenjuice import TokenJuiceEngine
        juice = TokenJuiceEngine(entropy_threshold=0.3, enrichment_rate=0.1)
        
        losses = []
        for i, (x, y) in enumerate(data_stream):
            # TokenJuice: Enrich batch dynamically with synthetic high-signal logic/identity tokens
            x, y = juice.enrich_batch(x, y)
            
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if y.dim() == 1:
                y = y.unsqueeze(0)
                
            loss, acc, ppl, grad_norm = self.train_step(x, y)
            losses.append(loss)
            
            # Dynamic Self-Repair
            if math.isnan(loss) or loss > 15.0:
                log.warning(f"Loss instability detected (Loss: {loss:.4f}). Triggering dynamic Self-Repair...")
                repair = SelfRepairEngine()
                repair.scan_and_repair(self.model)
            
            if (self.step_count % log_every == 0) or (i == 0) or (self.step_count == max_steps):
                elapsed = time.perf_counter() - self._start_time
                if elapsed < 1.0:
                    tok_per_sec = 0.0  # Prevent first-step time glitch
                else:
                    tok_per_sec = (x.numel() * (i + 1)) / max(elapsed, 1e-6)
                
                # ETA Timer calculation
                avg_sec_per_step = elapsed / max(self.step_count, 1)
                remaining_steps = max(max_steps - self.step_count, 0)
                eta_sec = remaining_steps * avg_sec_per_step
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
                
                log.info(f"Step {self.step_count:>4d}/{max_steps} │ Loss: {loss:.4f} │ PPL: {ppl:.1f} │ Acc: {acc:.2f}% │ ∇: {grad_norm:.2f} │ ⚡ {tok_per_sec:.1f} tok/s │ Tokens: {self.total_tokens/1000:.1f}K │ ETA: {eta_str}")
                
            if eval_every > 0 and self.step_count % eval_every == 0:
                if eval_callback:
                    eval_callback(self.step_count)
                    
            if self.step_count >= max_steps:
                break
                
        log.info(f"Dataset pre-training run complete ({self.step_count} steps executed).")
        return losses

    def train_demo(self, steps: int = 20, batch_size: int = 2, seq_len: int = 64, vocab_size: int = 32000) -> list[float]:
        """Run quick training demo over synthetic batches."""
        log.info(f"Starting training run: {steps} steps (batch={batch_size}, seq_len={seq_len})...")
        losses = []
        for i in range(steps):
            x, y = generate_synthetic_batch(vocab_size, batch_size, seq_len)
            loss, acc, ppl, grad_norm = self.train_step(x, y)
            losses.append(loss)
            if (i + 1) % 5 == 0 or i == 0:
                elapsed = time.perf_counter() - self._start_time
                if elapsed < 1.0:
                    tok_per_sec = 0.0
                else:
                    tok_per_sec = (batch_size * seq_len * (i + 1)) / max(elapsed, 1e-6)
                log.info(f"Step {self.step_count:>4d}/{steps} | Loss: {loss:.4f} | PPL: {ppl:.1f} | Acc: {acc:.2f}% | GradNorm: {grad_norm:.2f} | Speed: {tok_per_sec:.1f} tok/s [Self-Repair: OK]")
        return losses

    def save_checkpoint(self, path: str, save_optimizer: bool = True, copy_tokenizer: bool = True) -> None:
        """Save model checkpoint with self-contained tokenizer and max 5 checkpoint history cleanup."""
        ckpt_data = {
            "model_state_dict": {k: v.half() if v.is_floating_point() else v for k, v in self.model.state_dict().items()},
            "step_count": self.step_count,
            "best_loss": getattr(self, "best_loss", float('inf')),
            "total_tokens": getattr(self, "total_tokens", 0),
            "training_hours": (time.perf_counter() - self._start_time) / 3600.0,
        }
        if save_optimizer:
            ckpt_data["optimizer_state_dict"] = self.optimizer.state_dict()

        target_dir = os.path.dirname(path) or "."
        os.makedirs(target_dir, exist_ok=True)
        torch.save(ckpt_data, path)
        log.info(f"Checkpoint saved -> {path}")

        # Copy tokenizer.pt into target directory if available
        if copy_tokenizer:
            dst_tok = os.path.join(target_dir, "tokenizer.pt")
            candidates = [
                os.path.join(os.path.dirname(target_dir), "tokenizer.pt"),
                os.path.join(target_dir, "..", "tokenizer.pt"),
                os.path.join(os.path.dirname(target_dir), "Model", "tokenizer.pt"),
            ]
            src_tok = None
            for cand in candidates:
                if os.path.exists(cand):
                    src_tok = cand
                    break
            if src_tok and os.path.abspath(src_tok) != os.path.abspath(dst_tok):
                import shutil
                try:
                    shutil.copy2(src_tok, dst_tok)
                    log.info(f"Self-contained tokenizer synced -> {dst_tok}")
                except Exception:
                    pass

        # Auto-prune Checkpoints folder to keep at most 5 latest step checkpoints side by side
        if "Checkpoints" in target_dir or "checkpoints" in target_dir:
            self.prune_checkpoint_history(target_dir, max_keep=5)

    @staticmethod
    def prune_checkpoint_history(checkpoints_dir: str, max_keep: int = 5) -> None:
        """Keep only the latest max_keep step checkpoints in checkpoints_dir and remove older ones."""
        if not os.path.exists(checkpoints_dir):
            return
        files = []
        for fname in os.listdir(checkpoints_dir):
            if fname.startswith("checkpoint_step_") and fname.endswith(".pt"):
                fpath = os.path.join(checkpoints_dir, fname)
                try:
                    step_num = int(fname.replace("checkpoint_step_", "").replace(".pt", ""))
                    files.append((step_num, fpath))
                except ValueError:
                    files.append((os.path.getmtime(fpath), fpath))
        
        if len(files) > max_keep:
            files.sort(key=lambda x: x[0])  # sort ascending by step/mtime
            to_remove = files[: len(files) - max_keep]
            for _, fpath in to_remove:
                try:
                    os.remove(fpath)
                    log.info(f"Pruned older checkpoint -> {fpath}")
                except Exception as e:
                    log.warning(f"Could not remove old checkpoint {fpath}: {e}")

    def load_checkpoint(self, path: str) -> None:
        """Load model + optimizer state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # Handle half-precision saved weights
        state_dict = ckpt["model_state_dict"]
        model_state = self.model.state_dict()
        for k, v in state_dict.items():
            if k in model_state and v.dtype != model_state[k].dtype:
                state_dict[k] = v.to(model_state[k].dtype)
        self.model.load_state_dict(state_dict)
        # Optimizer is optional (only saved when save_optimizer=True)
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                log.warning("Could not restore optimizer state — using fresh optimizer.")
        self.step_count = ckpt.get("step_count", 0)
        self.best_loss = ckpt.get("best_loss", float('inf'))
        self.total_tokens = ckpt.get("total_tokens", 0)
        log.info(f"Checkpoint loaded <- {path} (step {self.step_count}, best_loss={self.best_loss:.4f})")
