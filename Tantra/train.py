"""
tantra/train.py — Training pipeline for NeuroCore models.

Changes vs. the original:
  * CrossEntropyLoss uses ignore_index=IGNORE_INDEX so the assistant-only
    loss masking produced by Tantra/dataset.py actually takes effect.
  * autocast is only enabled on cuda/mps (bf16 autocast on plain CPU adds
    cast/dispatch overhead without a corresponding speedup on most CPUs).
  * grad_accumulation_steps is now actually used: gradients accumulate over
    N micro-batches before clip/step/scheduler-step, giving a larger,
    less-noisy effective batch without more RAM. self.step_count now counts
    real optimizer steps, not micro-batches.
  * The LR scheduler's state is saved/restored across checkpoints, so
    resuming training no longer resets LR back to the warmup start.
"""
from __future__ import annotations

import os
import math
import time
import json
import copy
import shutil
import threading
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Any, Iterable, Optional, Tuple


from Tantra.utils import get_logger
from Tantra.evolution import AutoGrowthController, SelfRepairEngine

log = get_logger(__name__)

IGNORE_INDEX = -100


def generate_synthetic_batch(vocab_size: int = 32000, batch_size: int = 2, seq_len: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic token sequences and auto-regressive targets."""
    vocab_size = max(2, vocab_size)
    batch_size = max(1, batch_size)
    seq_len = max(1, seq_len)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.roll(x, -1, dims=-1)
    y[:, -1] = 0
    return x, y


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.05,
    start_factor: float = 1e-3,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Creates a strictly clamped linear warmup + cosine decay LR schedule.

    CRITICAL FIX (Oscillation Prevention):
    Standard PyTorch CosineAnnealingLR is periodic (cos(pi * t / T_max)) and oscillates
    back up to peak LR when training steps exceed T_max (e.g. on resume or long runs).
    This function clamps progress at 1.0 once current_step >= total_steps, keeping the LR
    strictly locked at min_lr_ratio * base_lr without oscillation.
    """
    total_steps = max(1, int(total_steps))
    actual_warmup = max(1, min(int(warmup_steps), max(1, total_steps // 10)))
    decay_steps = max(1, total_steps - actual_warmup)

    def lr_lambda(step: int) -> float:
        if step < actual_warmup:
            return max(start_factor, float(step) / float(actual_warmup))
        if step >= total_steps:
            return min_lr_ratio
        progress = float(step - actual_warmup) / float(decay_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Lion(torch.optim.Optimizer):
    """
    Lion (EvoLved Sign Momentum) optimizer.
    Memory-efficient on CPU: uses only 1 momentum buffer (half the memory of AdamW's 2 buffers).
    References:
        Chen et al., "Symbolic Discovery of Optimization Algorithms", 2023.
    """
    def __init__(self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]

                # Perform decoupled weight decay
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Weight update: update = sign(beta1 * exp_avg + (1 - beta1) * grad)
                update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                p.data.add_(update, alpha=-lr)

                # Decay momentum: exp_avg = beta2 * exp_avg + (1 - beta2) * grad
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


def build_optimizer(
    optimizer_name: str,
    parameters: Any,
    lr: float,
    weight_decay: float
) -> torch.optim.Optimizer:
    name = (optimizer_name or "adamw").lower().strip()
    if name == "lion":
        try:
            from lion_pytorch import Lion as ExternalLion
            return ExternalLion(parameters, lr=lr, weight_decay=weight_decay)
        except ImportError:
            return Lion(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    return AdamW(parameters, lr=lr, weight_decay=weight_decay)


class NeuroTrainer:
    """Minimal, robust trainer for NeuroCore models."""

    def __init__(self, model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.01,
                 optimizer_name: str = "adamw",
                 total_steps: int = 100000, warmup_steps: int = 1000,
                 grad_accumulation_steps: int = 1, use_latent_reasoning: bool = True,
                 use_mtp_loss: bool = True):
        self.model = model
        self.optimizer_name = optimizer_name.lower().strip()
        self.use_latent_reasoning = use_latent_reasoning
        self.use_mtp_loss = use_mtp_loss
        self.device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        log.info(f"  NeuroTrainer initialized on device: {self.device} (type={self.device.type}) | Optimizer: {self.optimizer_name}")
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters are enabled.")
        self.optimizer = build_optimizer(self.optimizer_name, trainable_parameters, lr=lr, weight_decay=weight_decay)

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.weight_decay = weight_decay

        # Non-oscillating clamped linear warmup + cosine decay
        self.scheduler = create_lr_scheduler(self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=0.05)

        self.criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.step_count = 0
        self.best_loss = float('inf')
        self.ema_loss = None
        self.total_tokens = 0
        self._session_tokens = 0   # tokens in THIS training run only (not restored from checkpoint)
        self._start_time = time.perf_counter()
        self._status_history: list[dict] = []

        self.grad_accumulation_steps = max(1, grad_accumulation_steps)
        self._micro_step = 0
        if self.grad_accumulation_steps > 1:
            log.info(f"  Gradient accumulation enabled: {self.grad_accumulation_steps} micro-batches per optimizer step")
        if not self.use_mtp_loss:
            log.info("  MTP auxiliary loss DISABLED for this run (reduced CPU output-projection work).")

    def refresh_optimizer(self) -> None:
        """Rebuild the optimizer (and LR schedule) from the model's current
        trainable parameters. Call this after ``freeze_for_category`` or
        ``grow_category`` so a newly unfrozen/added layer is actually optimized.
        The schedule keeps its current step count.
        """
        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 1e-4
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters are enabled after refresh.")
        self.optimizer = build_optimizer(self.optimizer_name, trainable_parameters, lr=lr, weight_decay=self.weight_decay)
        self.scheduler = create_lr_scheduler(self.optimizer, warmup_steps=self.warmup_steps, total_steps=self.total_steps, min_lr_ratio=0.05)

    def _write_training_status(self, **status: Any) -> None:
        """Publish real training state for the local Web UI and recovery logs."""
        try:
            status_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Model", "training_status.json")
            os.makedirs(os.path.dirname(status_path), exist_ok=True)
            status["updated_at"] = time.time()
            self._status_history.append({
                "step": status.get("step", self.step_count), "loss": status.get("loss"),
                "ppl": status.get("ppl"), "tok_s": status.get("tok_s"),
            })
            status["history"] = self._status_history[-50:]
            temporary_path = status_path + ".tmp"
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(status, handle)
            os.replace(temporary_path, status_path)
        except Exception as exc:
            log.debug(f"Could not publish training status: {exc}")

    def train_step(self, x: torch.Tensor, y: torch.Tensor, use_latent_reasoning: Optional[bool] = None) -> Tuple[float, Optional[float], float, float, bool]:
        """One micro-batch. Optimizer/scheduler only advance every
        `grad_accumulation_steps` calls; returned metrics always describe
        this micro-batch's own loss/accuracy/grad-norm.

        `use_latent_reasoning`: per-call override. Defaults to None, which
        falls back to self.use_latent_reasoning (set at construction) —
        but an explicit True/False passed here always wins, so callers
        (e.g. train_dataset) can flip the flag mid-run without rebuilding
        the trainer/optimizer/scheduler state."""
        if use_latent_reasoning is None:
            use_latent_reasoning = self.use_latent_reasoning
        self.model.train()

        if self._micro_step % self.grad_accumulation_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        # Accuracy is reporting-only.  With accumulation, calculate it only
        # on the micro-batch that finishes an optimizer update.
        will_step = ((self._micro_step + 1) % self.grad_accumulation_steps == 0)

        x, y = x.to(self.device), y.to(self.device)
        raw_m = getattr(self.model, "_orig_mod", self.model)
        if hasattr(raw_m, "embed") and hasattr(raw_m.embed, "weight"):
            vsize = raw_m.embed.weight.size(0)
            x = torch.clamp(x, 0, vsize - 1)

        # Autocast only helps on accelerators with native low-precision
        # matmul paths. On plain CPU it adds per-op cast/dispatch overhead
        # for no speed benefit on most hardware, so it stays off there.
        device_type = self.device.type if self.device.type in ('cuda', 'mps') else 'cpu'

        autocast_enabled = device_type != 'cpu'
        amp_dtype = torch.bfloat16 if (self.device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=autocast_enabled):
            out = self.model(x, return_mtp=self.use_mtp_loss, use_latent_reasoning=use_latent_reasoning)
            if isinstance(out[0], tuple):
                logits_main, logits_mtp = out[0]
            else:
                logits_main = out[0]
                logits_mtp = None

            # Numerically stable logits clamping with contiguous reshape
            logits_flat = torch.clamp(logits_main.reshape(-1, logits_main.size(-1)), -50.0, 50.0)
            y_flat = self._safe_targets(y.reshape(-1), logits_main.size(-1))

            loss = self.criterion(logits_flat, y_flat)

            if hasattr(self.model, "get_aux_loss"):
                loss = loss + self.model.get_aux_loss()

            # Auxiliary MTP Loss (Multi-Token Prediction)
            if logits_mtp is not None and y.size(1) > 1:
                logits_mtp_flat = torch.clamp(logits_mtp[:, :-1, :].reshape(-1, logits_mtp.size(-1)), -50.0, 50.0)
                y_mtp_flat = self._safe_targets(y[:, 1:].reshape(-1), logits_mtp.size(-1))
                mtp_loss = self.criterion(logits_mtp_flat, y_mtp_flat)
                loss = loss + 0.25 * mtp_loss

        if torch.isnan(loss) or torch.isinf(loss):
            self._micro_step += 1
            return 0.0, 0.0, 0.0, 0.0, False

        # Argmax scans the full vocabulary and is expensive on CPU. It does
        # not alter gradients, so skip it for intermediate micro-batches.
        with torch.no_grad():
            if will_step:
                supervised = y_flat != IGNORE_INDEX
                total = supervised.sum().clamp(min=1)

                # Top-1 Next-Token Accuracy
                preds = logits_flat.argmax(dim=-1)
                correct = ((preds == y_flat) & supervised).float().sum()
                accuracy: Optional[float] = (correct / total).item() * 100.0

                # Top-5 Candidate Pool Accuracy (Standard LLM Benchmark)
                k_val = min(5, logits_flat.size(-1))
                _, top5_indices = torch.topk(logits_flat, k=k_val, dim=-1)
                correct_top5 = (top5_indices == y_flat.unsqueeze(-1)).any(dim=-1) & supervised
                self.last_top5_acc = (correct_top5.float().sum() / total).item() * 100.0
            else:
                accuracy = None
                self.last_top5_acc = None
            ppl = math.exp(min(loss.item(), 20.0))

        (loss / self.grad_accumulation_steps).backward()

        self._micro_step += 1

        at_boundary = (self._micro_step % self.grad_accumulation_steps == 0)
        if at_boundary:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
            self.optimizer.step()
            self.scheduler.step()
            self.step_count += 1
        else:
            grad_norm = 0.0

        self.total_tokens += x.numel()
        self._session_tokens += x.numel()
        loss_val = loss.item()
        if self.ema_loss is None:
            self.ema_loss = loss_val
        else:
            self.ema_loss = 0.95 * self.ema_loss + 0.05 * loss_val
        if self.ema_loss < self.best_loss:

            self.best_loss = self.ema_loss
        return loss_val, accuracy, ppl, grad_norm, at_boundary

    @staticmethod
    def _safe_targets(y_flat: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Clamp real token ids into vocab range while leaving IGNORE_INDEX
        untouched (a naive clamp(0, vocab-1) would corrupt -100 into 0)."""
        ignore_mask = y_flat == IGNORE_INDEX
        if not ignore_mask.any():
            return y_flat.clamp(0, vocab_size - 1)
        out = y_flat.clone()
        keep = ~ignore_mask
        out[keep] = out[keep].clamp(0, vocab_size - 1)
        return out

    def evaluate_validation(self, val_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]], max_val_batches: int = 20) -> dict:
        """Evaluate model on held-out validation data stream."""
        if val_loader is None:
            return {}
        raw_model = getattr(self.model, "_orig_mod", self.model)

        raw_model.eval()
        val_losses = []
        val_accs = []
        val_top5_accs = []
        val_ppls = []

        with torch.no_grad():
            batch_count = 0
            for vx, vy in val_loader:
                vx = vx.to(self.device)
                vy = vy.to(self.device)
                outputs = raw_model(vx)
                if isinstance(outputs, (tuple, list)):
                    first = outputs[0]
                    logits = first[0] if isinstance(first, (tuple, list)) else first
                else:
                    logits = outputs

                valid_mask = vy != IGNORE_INDEX
                if not valid_mask.any():
                    continue

                logits_flat = torch.clamp(logits.reshape(-1, logits.size(-1)), -50.0, 50.0)
                y_flat = self._safe_targets(vy.reshape(-1), logits.size(-1))
                loss = F.cross_entropy(logits_flat, y_flat, ignore_index=IGNORE_INDEX)
                val_losses.append(loss.item())

                # Top-1 Accuracy
                preds = logits.argmax(dim=-1)
                correct = (preds == vy) & valid_mask
                acc = correct.sum().float() / valid_mask.sum().float() * 100.0
                val_accs.append(acc.item())

                # Top-5 Accuracy
                k_val = min(5, logits_flat.size(-1))
                _, top5_indices = torch.topk(logits_flat, k=k_val, dim=-1)
                correct_top5 = (top5_indices == y_flat.unsqueeze(-1)).any(dim=-1) & (y_flat != IGNORE_INDEX)
                top5_acc = correct_top5.float().sum() / (y_flat != IGNORE_INDEX).sum().float().clamp(min=1) * 100.0
                val_top5_accs.append(top5_acc.item())



                ppl = math.exp(min(loss.item(), 20.0))
                val_ppls.append(ppl)

                batch_count += 1
                if batch_count >= max_val_batches:
                    break

        raw_model.train()
        if not val_losses:
            return {}

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_acc = sum(val_accs) / len(val_accs)
        avg_val_top5_acc = sum(val_top5_accs) / len(val_top5_accs) if val_top5_accs else 0.0
        avg_val_ppl = sum(val_ppls) / len(val_ppls)
        return {
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc,
            "val_top5_acc": avg_val_top5_acc,
            "val_ppl": avg_val_ppl,
        }


    def train_dataset(self, data_stream: Iterable[Tuple[torch.Tensor, torch.Tensor]], max_steps: int = 100, log_every: int = 10, eval_every: int = 0, eval_callback = None, checkpoint_every: int = 0, checkpoint_callback = None, tokenizer: Optional[Any] = None, enrichment_rate: float = 0.02, use_latent_reasoning: bool = True, auto_growth: bool = False, growth_patience: int = 1000, growth_min_delta: float = 0.005, max_layers: Optional[int] = None, val_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]] = None) -> list[float]:

        """Train over an iterable dataset stream (e.g. JSONLDataset).

        `tokenizer`: optional UnifiedTokenizer-like object (must expose
        `.encode(text)` returning a list[int]) used to tokenize the built-in
        TokenJuice identity/logic synthetic pairs. Without it, enrichment is
        skipped (TokenJuice requires raw token IDs, not raw text).

        NOTE: `max_steps` counts optimizer steps (self.step_count), not
        micro-batches — with grad_accumulation_steps > 1 this loop will
        consume grad_accumulation_steps times as many items from
        data_stream to reach max_steps.
        """
        log.info(f"Starting dataset pre-training run (target steps: {max_steps})...")
        # Reset session counters so tok/s and ETA reflect THIS run, not checkpoint history
        self._session_tokens = 0
        self._session_start_step = self.step_count
        self._start_time = time.perf_counter()
        log_every = max(1, int(log_every))
        checkpoint_every = max(0, int(checkpoint_every))
        growth_controller = None
        if auto_growth:
            growth_controller = AutoGrowthController(
                plateau_patience=max(20, int(growth_patience)),
                min_delta=max(0.0, float(growth_min_delta)),
                max_layers=max_layers,
            )
            log.info("  Auto-growth enabled: monitor every %d optimizer steps; max layers=%s.", growth_controller.plateau_patience, max_layers if max_layers is not None else "unbounded")
        if not use_latent_reasoning:
            log.info("  Latent CoT reasoning DISABLED for this run (~3x cheaper per step on that stage) "
                     "— re-enable for fine-tuning/reasoning-quality passes.")

        from Tantra.dataset import TokenJuiceEngine
        juice = TokenJuiceEngine(entropy_threshold=0.3, enrichment_rate=enrichment_rate)
        synthetic_qa_pairs = [
            ("What is Tantra?", "Tantra is a CPU-First Autonomous AI Engine."),
            ("Who created Tantra?", "Tantra LLM is created by the Tantra Engineering Team."),
            ("Explain artificial intelligence.", "AI is the simulation of human intelligence by computer systems."),
        ]
        if tokenizer is not None and enrichment_rate > 0.0:
            for question, answer in synthetic_qa_pairs:
                try:
                    q_ids = tokenizer.encode(question)
                    a_ids = tokenizer.encode(answer)
                    if q_ids and a_ids:
                        juice.register_synthetic_pair(q_ids, a_ids)
                except Exception as e:
                    log.warning(f"Could not tokenize TokenJuice synthetic pair ({question!r}): {e}")
        else:
            log.debug("TokenJuice enrichment disabled for this run (enrichment_rate <= 0 or no tokenizer).")

        losses = []
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
            has_rich = True
        except ImportError:
            has_rich = False

        if has_rich:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]Dataset Training[/bold cyan]"),
                BarColumn(bar_width=30),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TextColumn("[green]Loss: {task.fields[loss]:.4f}[/green]"),
                TextColumn("[yellow]Acc: {task.fields[acc]:.1f}%[/yellow]"),
                TextColumn("[blue]PPL: {task.fields[ppl]:.1f}[/blue]"),
                TextColumn("[dim]ETA: {task.fields[eta]}[/dim]"),
                TextColumn("[magenta]⚡ {task.fields[tok_s]:.1f} tok/s[/magenta]"),
            )
            task_id = progress.add_task("Train", total=max_steps, completed=self.step_count, loss=0.0, acc=0.0, ppl=0.0, tok_s=0.0, eta="estimating")
            progress.start()
        else:
            progress = None

        self._write_training_status(
            status="running", step=self.step_count, target_steps=max_steps,
            loss=None, ema_loss=self.ema_loss, accuracy=None, ppl=None,
            grad_norm=None, tok_s=0.0, session_tokens=0, eta="estimating",
            eta_seconds=None,
        )

        self._last_eval_step = getattr(self, "_last_eval_step", -1)
        window_losses: list[float] = []
        window_accs: list[float] = []
        window_top5_accs: list[float] = []
        last_accuracy = 0.0
        window_ppls: list[float] = []
        window_grad_norms: list[float] = []
        window_optimizer_steps = 0
        last_optimizer_time = self._start_time
        recent_step_seconds: list[float] = []

        try:
            for i, (x, y) in enumerate(data_stream):
                # TokenJuice: Enrich batch dynamically with synthetic high-signal logic/identity tokens
                x, y = juice.enrich_batch(x, y)

                if x.dim() == 1:
                    x = x.unsqueeze(0)
                if y.dim() == 1:
                    y = y.unsqueeze(0)

                loss, acc, ppl, grad_norm, at_boundary = self.train_step(x, y, use_latent_reasoning=use_latent_reasoning)
                losses.append(loss)
                window_losses.append(loss)
                if acc is not None:
                    last_accuracy = acc
                    window_accs.append(acc)
                if getattr(self, "last_top5_acc", None) is not None:
                    window_top5_accs.append(self.last_top5_acc)
                window_ppls.append(ppl)


                # Dynamic Self-Repair — rate-limited to at most once every 500 optimizer steps
                # to avoid scanning all parameters on every high-loss micro-batch at training start.
                _repair_interval = 500
                if math.isnan(loss) or loss > 15.0:
                    last_repair = getattr(self, "_last_repair_step", -_repair_interval)
                    if self.step_count - last_repair >= _repair_interval:
                        log.warning(f"Loss instability detected (Loss: {loss:.4f}) at step {self.step_count}. Triggering dynamic Self-Repair...")
                        repair = SelfRepairEngine()
                        repair.scan_and_repair(self.model)
                        self._last_repair_step = self.step_count

                elapsed = time.perf_counter() - self._start_time
                # Use session tokens (not total_tokens which includes checkpoint history)
                tok_per_sec = self._session_tokens / max(elapsed, 1e-6)

                # Rich redraws are UI-only. Update it when a real optimizer
                # step completes instead of every accumulation micro-batch.
                if progress and at_boundary:
                    progress.update(
                        task_id,
                        completed=min(self.step_count, max_steps),
                        loss=loss,
                        acc=last_accuracy,
                        ppl=ppl,
                        tok_s=tok_per_sec,
                    )

                if at_boundary and ((self.step_count == 1) or (self.step_count % log_every == 0) or (self.step_count == max_steps)):
                    # Use optimizer step count (not raw micro-batch index) so ETA is correct
                    # even when grad_accumulation_steps > 1.
                    session_steps = max(self.step_count - self._session_start_step, 1)
                    avg_sec_per_step = elapsed / session_steps
                    remaining_steps = max(max_steps - self.step_count, 0)
                    eta_sec = int(remaining_steps * avg_sec_per_step)

                    # Format ETA with days
                    d = eta_sec // 86400
                    h = (eta_sec % 86400) // 3600
                    m = (eta_sec % 3600) // 60
                    s = eta_sec % 60
                    if d > 0:
                        eta_str = f"{d}d {h:02d}:{m:02d}:{s:02d}"
                    else:
                        eta_str = f"{h:02d}:{m:02d}:{s:02d}"

                    if not progress:
                        log.info(f"Step {self.step_count:>4d}/{max_steps} │ Loss: {loss:.4f} │ PPL: {ppl:.1f} │ Acc: {last_accuracy:.2f}% │ ∇: {grad_norm:.2f} │ ⚡ {tok_per_sec:.1f} tok/s │ Tokens: {self.total_tokens/1000:.1f}K │ ETA: {eta_str}")

                # Emit a compact, rolling optimizer-step summary even while
                # Rich owns the progress bar.  This makes long CPU runs
                # auditable without flooding the terminal per micro-batch.
                if at_boundary:
                    window_optimizer_steps += 1
                    window_grad_norms.append(grad_norm)
                    now = time.perf_counter()
                    recent_step_seconds.append(now - last_optimizer_time)
                    recent_step_seconds = recent_step_seconds[-10:]
                    last_optimizer_time = now
                    rolling_eta = "estimating"
                    if len(recent_step_seconds) >= 3:
                        remaining_steps = max(max_steps - self.step_count, 0)
                        eta_sec = int(remaining_steps * (sum(recent_step_seconds) / len(recent_step_seconds)))
                        days, remainder = divmod(eta_sec, 86400)
                        hours, remainder = divmod(remainder, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        rolling_eta = f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}" if days else f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                    if progress:
                        progress.update(task_id, eta=rolling_eta)

                    rolling_eta_seconds = None
                    if len(recent_step_seconds) >= 3:
                        rolling_eta_seconds = int(max(max_steps - self.step_count, 0) * (
                            sum(recent_step_seconds) / len(recent_step_seconds)
                        ))
                    self._write_training_status(
                        status="running", step=self.step_count, target_steps=max_steps,
                        loss=loss, ema_loss=self.ema_loss, accuracy=last_accuracy, ppl=ppl,
                        grad_norm=grad_norm, tok_s=tok_per_sec,
                        session_tokens=self._session_tokens, eta=rolling_eta,
                        eta_seconds=rolling_eta_seconds,
                    )

                    if growth_controller is not None:
                        raw_model = getattr(self.model, "_orig_mod", self.model)
                        before = {id(param) for param in raw_model.parameters()}
                        if growth_controller.observe(float(self.ema_loss), raw_model):
                            new_params = [param for param in raw_model.parameters() if id(param) not in before]
                            if new_params:
                                # Existing parameters retain their Adam state;
                                # only the new layer starts with fresh state.
                                base_group = self.optimizer.param_groups[0]
                                self.optimizer.add_param_group({
                                    "params": new_params,
                                    "lr": base_group["lr"],
                                    "weight_decay": base_group.get("weight_decay", 0.0),
                                })
                                log.info("Auto-growth added %d parameters; optimizer now tracks %d layers.", sum(p.numel() for p in new_params), len(raw_model.layers))

                    session_steps = self.step_count - self._session_start_step
                    if session_steps == 1 or session_steps % log_every == 0 or self.step_count == max_steps:
                        first_step = self.step_count - window_optimizer_steps + 1
                        avg_loss = sum(window_losses) / max(len(window_losses), 1)
                        avg_acc = sum(window_accs) / max(len(window_accs), 1)
                        avg_top5_acc = sum(window_top5_accs) / max(len(window_top5_accs), 1) if window_top5_accs else 0.0
                        avg_ppl = sum(window_ppls) / max(len(window_ppls), 1)
                        avg_grad = sum(window_grad_norms) / max(len(window_grad_norms), 1)
                        pct = (self.step_count / max(max_steps, 1)) * 100.0
                        loss_arrow = "🔻" if (self.best_loss is None or avg_loss <= self.best_loss) else "🔸"
                        acc_arrow = "📈" if avg_top5_acc > 0 else ""

                        current_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
                        total_params = sum(p.numel() for p in self.model.parameters())

                        user_snippet = ""
                        asst_snippet = ""
                        if tokenizer is not None:
                            try:
                                sample_toks = [t for t in x[0].cpu().tolist() if t > 0]
                                decoded_text = tokenizer.decode(sample_toks)
                                
                                # Strip System Prompt if present
                                if "<|user|>" in decoded_text:
                                    after_user = decoded_text.split("<|user|>", 1)[1]
                                    if "<|assistant|>" in after_user:
                                        u_part, a_part = after_user.split("<|assistant|>", 1)
                                        user_snippet = u_part.replace("</s>", "").strip()[:85]
                                        asst_snippet = a_part.replace("</s>", "").strip()[:95]
                                    else:
                                        user_snippet = after_user.replace("</s>", "").strip()[:85]
                                else:
                                    # Fallback: remove standard system prompt text
                                    clean_d = decoded_text.replace("You are Tantra, a helpful, precise, and polite AI assistant created by Atulya AI. Answer clearly, accurately, and step-by-step.", "").strip()
                                    user_snippet = clean_d[:85]
                            except Exception:
                                pass

                        header = f"Step {self.step_count:,}/{max_steps:,} ({pct:.1f}%)"
                        metrics_line1 = f"Loss: {avg_loss:.4f} {loss_arrow} │ Top-1 Acc: {avg_acc:.1f}% │ Top-5 Acc: {avg_top5_acc:.1f}% {acc_arrow} │ PPL: {avg_ppl:.1f}"
                        metrics_line2 = f"Params: {total_params/1e6:.1f}M │ LR: {current_lr:.2e} │ Speed: {tok_per_sec:.1f} tok/s │ ETA: {rolling_eta}"
                        
                        log.info(f"┌── [{header}] ────────────────────────────────────────────────────────")
                        log.info(f"│ 📊 {metrics_line1}")
                        log.info(f"│ ⚙️  {metrics_line2}")
                        if user_snippet:
                            log.info(f"│ ❓ [User Prompt]  : {user_snippet}")
                        if asst_snippet:
                            log.info(f"│ 💡 [Target Answer]: {asst_snippet}")
                        log.info(f"└── Tokens: {self._session_tokens/1000:.1f}K processed ──────────────────────────────────────────")

                        window_losses.clear()
                        window_accs.clear()
                        window_top5_accs.clear()
                        window_ppls.clear()
                        window_grad_norms.clear()
                        window_optimizer_steps = 0

                if at_boundary and eval_every > 0 and (self.step_count % eval_every == 0) and (self.step_count != self._last_eval_step):
                    self._last_eval_step = self.step_count
                    if val_loader is not None:
                        val_res = self.evaluate_validation(val_loader)
                        if val_res:
                            v_loss = val_res["val_loss"]
                            v_acc = val_res["val_acc"]
                            v_top5 = val_res.get("val_top5_acc", 0.0)
                            v_ppl = val_res["val_ppl"]
                            log.info(
                                f"   [VAL EVAL @ Step {self.step_count}] "
                                f"Val Loss: {v_loss:.4f} │ Val Top-1: {v_acc:.2f}% │ Val Top-5: {v_top5:.2f}% │ Val PPL: {v_ppl:.1f}"
                            )
                            if (avg_acc - v_acc > 30.0) or (avg_loss > 0 and (v_loss / avg_loss) > 2.5 and avg_acc > 80.0):
                                log.warning(
                                    f"   ⚠️ [OVERFITTING DETECTED @ Step {self.step_count}] Training Accuracy ({avg_acc:.1f}%) "
                                    f"is significantly higher than Held-Out Validation Accuracy ({v_acc:.1f}%). "
                                    f"The model is memorizing training examples!"
                                )


                    if eval_callback is not None:
                        if progress:
                            progress.stop()
                        eval_callback(self.step_count)
                        if progress and self.step_count < max_steps:
                            progress.start()


                # Lightweight recovery checkpoint: this writes only the latest
                # resumable state.  Full sampled/archival checkpoints remain on
                # the much less frequent evaluation schedule.
                if at_boundary and checkpoint_every > 0 and (self.step_count % checkpoint_every == 0):
                    if checkpoint_callback is not None:
                        checkpoint_callback(self.step_count)

                if self.step_count >= max_steps:
                    break
        finally:
            if progress:
                progress.stop()

        self._write_training_status(
            status="complete" if self.step_count >= max_steps else "stopped",
            step=self.step_count, target_steps=max_steps,
            loss=losses[-1] if losses else None, ema_loss=self.ema_loss,
            accuracy=last_accuracy if losses else None, ppl=ppl if losses else None,
            grad_norm=grad_norm if losses else None, tok_s=tok_per_sec if losses else 0.0,
            session_tokens=self._session_tokens, eta="00:00:00" if self.step_count >= max_steps else "stopped",
            eta_seconds=0 if self.step_count >= max_steps else None,
        )
        log.info(f"Dataset pre-training run complete ({self.step_count} steps executed).")
        return losses

    def train_demo(self, steps: int = 20, batch_size: int = 2, seq_len: int = 64, vocab_size: int = 32000) -> list[float]:
        """Run quick training demo over synthetic batches."""
        log.info(f"Starting training run: {steps} steps (batch={batch_size}, seq_len={seq_len})...")
        losses = []
        for i in range(steps):
            x, y = generate_synthetic_batch(vocab_size, batch_size, seq_len)
            loss, acc, ppl, grad_norm, _ = self.train_step(x, y)
            losses.append(loss)
            if (i + 1) % 5 == 0 or i == 0:
                elapsed = time.perf_counter() - self._start_time
                if elapsed < 1.0:
                    tok_per_sec = 0.0
                else:
                    tok_per_sec = (batch_size * seq_len * (i + 1)) / max(elapsed, 1e-6)
                log.info(f"Step {self.step_count:>4d}/{steps} | Loss: {loss:.4f} | PPL: {ppl:.1f} | Acc: {acc:.2f}% | GradNorm: {grad_norm:.2f} | Speed: {tok_per_sec:.1f} tok/s [Self-Repair: OK]")
        return losses

    def save_checkpoint(self, path: str, save_optimizer: bool = True, async_write: bool = False) -> None:
        """Save model checkpoint with self-contained tokenizer and max 2 checkpoint history cleanup."""
        # Create an in-memory detached state dict clone to guarantee thread safety
        model_sd = {k: (v.clone().half() if v.is_floating_point() else v.clone()) for k, v in self.model.state_dict().items()}
        opt_sd = copy.deepcopy(self.optimizer.state_dict()) if save_optimizer else None

        ckpt_data = {
            "model_state_dict": model_sd,
            "config": getattr(self.model, "config", None),
            "step_count": self.step_count,
            "best_loss": getattr(self, "best_loss", float('inf')),
            "total_tokens": getattr(self, "total_tokens", 0),
            "training_hours": (time.perf_counter() - self._start_time) / 3600.0,
            "scheduler_state_dict": copy.deepcopy(self.scheduler.state_dict()),
            "total_steps": self.total_steps,
            "num_layers": len(getattr(getattr(self.model, "_orig_mod", self.model), "layers", [])),
        }
        if opt_sd is not None:
            ckpt_data["optimizer_state_dict"] = opt_sd

        step_num = self.step_count

        def _disk_writer(data, target_path, step):
            target_dir = os.path.dirname(target_path) or "."
            os.makedirs(target_dir, exist_ok=True)
            temporary_path = target_path + ".tmp"
            try:
                torch.save(data, temporary_path)
                os.replace(temporary_path, target_path)
                meta_path = target_path + ".meta.json"
                meta_temp = meta_path + ".tmp"
                with open(meta_temp, "w", encoding="utf-8") as handle:
                    json.dump({"num_layers": data["num_layers"], "step_count": step}, handle)
                os.replace(meta_temp, meta_path)
                # Save/copy updated tokenizer.json into checkpoint target_dir and root model_dir
                for cand in [os.path.join("Model", "tokenizer.json"), "tokenizer.json", os.path.join(target_dir, "..", "tokenizer.json")]:
                    if os.path.exists(cand):
                        try:
                            shutil.copy2(cand, os.path.join(target_dir, "tokenizer.json"))
                            root_d = os.path.dirname(target_dir)
                            if root_d and os.path.isdir(root_d):
                                shutil.copy2(cand, os.path.join(root_d, "tokenizer.json"))
                            break
                        except Exception:
                            pass
            except Exception as ex:

                try:
                    if os.path.exists(temporary_path):
                        os.remove(temporary_path)
                except OSError:
                    pass
                log.warning(f"Failed checkpoint write to {target_path}: {ex}")
                return
            log.info(f"Checkpoint saved -> {target_path}")

            if "Checkpoints" in target_dir or "checkpoints" in target_dir:
                NeuroTrainer.prune_checkpoint_history(target_dir, max_keep=2)

        if async_write:
            t = threading.Thread(target=_disk_writer, args=(ckpt_data, path, step_num), daemon=True)
            t.start()
        else:
            _disk_writer(ckpt_data, path, step_num)

    @staticmethod
    def prune_checkpoint_history(checkpoints_dir: str, max_keep: int = 2) -> None:
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
        """Load model + optimizer + scheduler state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"]
        raw_model = getattr(self.model, "_orig_mod", self.model)
        model_state = raw_model.state_dict()

        # Auto-align vocabulary shape mismatches (e.g. checkpoint saved at 32000, model configured for 65536)
        for k in ["embed.weight", "output_proj.weight", "mtp_head.weight"]:
            if k in state_dict and k in model_state:
                ckpt_w = state_dict[k]
                target_w = model_state[k]
                if ckpt_w.shape != target_w.shape:
                    log.info(f"Auto-aligning checkpoint tensor '{k}' shape {ckpt_w.shape} -> {target_w.shape}")
                    aligned_w = target_w.clone()
                    min_vocab = min(ckpt_w.size(0), target_w.size(0))
                    aligned_w[:min_vocab] = ckpt_w[:min_vocab]
                    state_dict[k] = aligned_w

        for k, v in state_dict.items():
            if k in model_state and v.dtype != model_state[k].dtype:
                state_dict[k] = v.to(model_state[k].dtype)
        # strict=False tolerates pre-gate checkpoints that lack category_gates.*
        # (and category_layers installed after a checkpoint was written); gates
        # for trained legacy categories are opened by the sync call below.
        raw_model.load_state_dict(state_dict, strict=False)
        raw_model.sync_category_gates_from_checkpoint(state_dict)
        # Optimizer is optional (only saved when save_optimizer=True)
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                log.warning("Could not restore optimizer state — using fresh optimizer.")
        self.step_count = ckpt.get("step_count", 0)
        self.best_loss = ckpt.get("best_loss", float('inf'))
        self.total_tokens = ckpt.get("total_tokens", 0)
        if "total_steps" in ckpt:
            self.total_steps = max(int(self.total_steps), int(ckpt["total_steps"]))
        if "warmup_steps" in ckpt:
            self.warmup_steps = int(ckpt["warmup_steps"])

        if "scheduler_state_dict" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                log.info("LR scheduler state restored — resuming at correct point in warmup/cosine schedule.")
            except Exception as e:
                log.warning(f"Could not restore scheduler state ({e}); fast-forwarding by step_count instead.")
                self._fast_forward_scheduler()
        elif self.step_count > 0:
            log.warning("Checkpoint has no scheduler_state_dict (older checkpoint) — "
                        "fast-forwarding scheduler by step_count so LR doesn't reset to warmup start.")
            self._fast_forward_scheduler()

        log.info(f"Checkpoint loaded <- {path} (step {self.step_count}, best_loss={self.best_loss:.4f})")

    def _fast_forward_scheduler(self) -> None:
        """Best-effort recovery for checkpoints saved before scheduler state
        was tracked: replay .step() step_count times so LR lands close to
        where it should be instead of resetting to the warmup start."""
        import warnings
        steps = min(self.step_count, self.total_steps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            for _ in range(steps):
                self.scheduler.step()
