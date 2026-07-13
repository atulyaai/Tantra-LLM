from __future__ import annotations

"""Train fusion projection layers while freezing base models."""

import torch
import torch.nn.functional as F
import logging
import time
from torch.nn import Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Any, Optional, Dict, List

from npdna.training.training_config import FusionTrainingConfig

logger = logging.getLogger(__name__)


class FusionProjector(Module):
    """Lightweight MLP projector with dropout for regularization."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        # Use a smaller hidden dim: 1/4 of output to keep params low
        hidden_dim = hidden_dim or max(256, output_dim // 4)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"[Projector] {input_dim} -> {hidden_dim} -> {output_dim} ({total:,} params, dropout={dropout})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionTrainer:
    """
    Train vision and audio projection layers (base model frozen).
    Features: contrastive loss, gradient clipping, LR warmup, early stopping.
    """

    def __init__(
        self,
        config: FusionTrainingConfig,
        vision_projector: Module,
        audio_projector: Module,
        base_model: Optional[Any] = None,
        use_compile: bool = False,
        patience: int = 3
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Optimize CPU threading
        if not torch.cuda.is_available():
            import os
            num_threads = os.cpu_count() or 4
            torch.set_num_threads(num_threads)
            logger.info(f"[CPU] Using {num_threads} threads")

        if use_compile and hasattr(torch, "compile"):
            try:
                self.vision_projector = torch.compile(vision_projector).to(self.device)
                self.audio_projector = torch.compile(audio_projector).to(self.device)
            except Exception:
                self.vision_projector = vision_projector.to(self.device)
                self.audio_projector = audio_projector.to(self.device)
        else:
            self.vision_projector = vision_projector.to(self.device)
            self.audio_projector = audio_projector.to(self.device)

        self.base_model = base_model
        self.patience = patience

        # Freeze base model
        if self.base_model:
            if hasattr(self.base_model, "to"):
                self.base_model = self.base_model.to(self.device)
            for param in self.base_model.parameters():
                param.requires_grad = False

        trainable_params = list(self.vision_projector.parameters()) + list(self.audio_projector.parameters())
        self.total_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"[Trainer] Trainable parameters: {self.total_trainable:,}")

        self.optimizer = AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = None  # Setup dynamically inside fit() based on dataset size

        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.global_step = 0

    def _warmup_lr(self):
        """Linear warmup for the first N steps."""
        if self.global_step < self.config.warmup_steps:
            warmup_factor = (self.global_step + 1) / self.config.warmup_steps
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.config.lr * warmup_factor

    def fit(self, train_dataset, val_dataset=None) -> Dict[str, List[float]]:
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False) if val_dataset else None

        # Setup Cosine Annealing scheduler based on actual step updates
        steps_per_epoch = (len(train_loader) + self.config.grad_accum - 1) // self.config.grad_accum
        total_steps = steps_per_epoch * self.config.epochs
        t_max = max(1, total_steps - self.config.warmup_steps)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=1e-6)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": [], "epoch_time_s": []}
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"  FUSION PROJECTOR TRAINING")
        print(f"  Device: {self.device} | Params: {self.total_trainable:,}")
        print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset) if val_dataset else 0}")
        print(f"  BS: {self.config.batch_size} | Epochs: {self.config.epochs} | LR: {self.config.lr}")
        print(f"  Warmup: {self.config.warmup_steps} steps | Grad Clip: {self.config.grad_clip}")
        print(f"  Grad Accum: {self.config.grad_accum} | Dropout: {self.config.dropout}")
        print(f"{'='*60}\n")

        for epoch in range(self.config.epochs):
            t0 = time.time()

            # -- Train --
            self.vision_projector.train()
            self.audio_projector.train()
            epoch_loss = 0.0
            steps = 0
            accum_count = 0  # Track microbatches accumulated since last optimizer step
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(train_loader):
                self._warmup_lr()

                vision_e = batch.get("vision_embeds")
                audio_e = batch.get("audio_embeds")
                targets = batch.get("target_ids")

                if vision_e is not None: vision_e = vision_e.to(self.device)
                if audio_e is not None: audio_e = audio_e.to(self.device)
                if targets is not None: targets = targets.to(self.device)

                device_type = "cuda" if self.use_amp else "cpu"
                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    loss = self._forward_step(vision_e, audio_e, targets)

                if loss is not None:
                    # Calculate group size at the start of each accumulation group
                    if accum_count == 0:
                        remaining = len(train_loader) - batch_idx
                        current_group_size = min(self.config.grad_accum, remaining)

                    accum_count += 1
                    is_last_batch = (batch_idx + 1) == len(train_loader)
                    scaled_loss = loss / current_group_size

                    if self.scaler:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    # Step optimizer on full accumulation boundary or final batch
                    if accum_count == self.config.grad_accum or is_last_batch:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.vision_projector.parameters()) + list(self.audio_projector.parameters()),
                            self.config.grad_clip
                        )
                        if self.scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        accum_count = 0

                        # Step cosine scheduler only after warmup
                        if self.global_step >= self.config.warmup_steps:
                            self.scheduler.step()

                        self.global_step += 1

                    epoch_loss += loss.item()
                    steps += 1

            avg_train = epoch_loss / max(steps, 1)
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(avg_train)
            history["lr"].append(current_lr)

            epoch_time = time.time() - t0
            history["epoch_time_s"].append(epoch_time)

            # -- Validate --
            if val_loader:
                avg_val = self._evaluate(val_loader)
                history["val_loss"].append(avg_val)
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs} | "
                      f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
                      f"LR: {current_lr:.2e} | {epoch_time:.2f}s")

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    patience_counter = 0
                    self.save_checkpoint("checkpoints/best_projectors.pt", include_states=False)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"\n  [Early Stop] epoch {epoch+1}, patience={self.patience}")
                        break
            else:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs} | "
                      f"Train: {avg_train:.6f} | LR: {current_lr:.2e} | {epoch_time:.2f}s")

        total_time = sum(history["epoch_time_s"])
        print(f"\n{'='*60}")
        print(f"  Done in {total_time:.2f}s | Final train: {history['train_loss'][-1]:.6f}")
        if history["val_loss"]:
            print(f"  Best val: {best_val_loss:.6f}")
        print(f"{'='*60}\n")

        # Save final light weights (excluding optimizer/scheduler states for inference/production)
        self.save_checkpoint("checkpoints/final_projectors.pt", include_states=False)
        return history

    def _evaluate(self, val_loader) -> float:
        self.vision_projector.eval()
        self.audio_projector.eval()
        total_loss = 0.0
        steps = 0
        device_type = "cuda" if self.use_amp else "cpu"

        with torch.no_grad():
            for batch in val_loader:
                v = batch.get("vision_embeds")
                a = batch.get("audio_embeds")
                t = batch.get("target_ids")
                if v is not None: v = v.to(self.device)
                if a is not None: a = a.to(self.device)
                if t is not None: t = t.to(self.device)

                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    loss = self._forward_step(v, a, t)
                if loss is not None:
                    total_loss += loss.item()
                    steps += 1

        return total_loss / max(steps, 1)

    def _forward_step(self, vision_embeds, audio_embeds, targets) -> Optional[torch.Tensor]:
        """Symmetric contrastive alignment loss combined with language model supervision loss."""
        # 1. Forward through projectors
        projected_v = self.vision_projector(vision_embeds) if vision_embeds is not None else None
        projected_a = self.audio_projector(audio_embeds) if audio_embeds is not None else None

        if projected_v is None and projected_a is None:
            return None

        # 2. Identify active (non-zero) modalities per sample in the batch
        # Zero-filled inputs represent missing/unpaired modalities and must not pollute alignment
        has_v = (vision_embeds.abs().sum(dim=-1) > 0.0) if vision_embeds is not None else torch.zeros(0, dtype=torch.bool, device=self.device)
        has_a = (audio_embeds.abs().sum(dim=-1) > 0.0) if audio_embeds is not None else torch.zeros(0, dtype=torch.bool, device=self.device)

        contrastive_loss = None
        # Cross-modal contrastive alignment only on samples that have BOTH modalities present
        if projected_v is not None and projected_a is not None:
            both_mask = has_v & has_a
            if both_mask.sum() > 1:
                v_aligned = projected_v[both_mask]
                a_aligned = projected_a[both_mask]
                v_norm = F.normalize(v_aligned, dim=-1)
                a_norm = F.normalize(a_aligned, dim=-1)
                logits = torch.matmul(v_norm, a_norm.T) / 0.07
                bs = logits.size(0)
                labels = torch.arange(bs, device=self.device)
                contrastive_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0

        # 3. Language Model Supervision
        lm_loss = None
        if self.base_model and targets is not None:
            try:
                # Resolve the embedding layer of the base model
                embed_layer = None
                for attr in ["get_input_embeddings", "embeddings", "wte"]:
                    if hasattr(self.base_model, attr):
                        if attr == "get_input_embeddings":
                            embed_layer = self.base_model.get_input_embeddings()
                        else:
                            embed_layer = getattr(self.base_model, attr)
                        break
                if not embed_layer and hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "wte"):
                    embed_layer = self.base_model.transformer.wte

                if embed_layer is not None:
                    # Embed targets: shape [BS, seq_len, model_dim]
                    target_embeds = embed_layer(targets)
                    inputs_embeds_list = []
                    
                    # Prepend projected modality features to targets
                    if projected_v is not None:
                        inputs_embeds_list.append(projected_v.unsqueeze(1))
                    if projected_a is not None:
                        inputs_embeds_list.append(projected_a.unsqueeze(1))
                    inputs_embeds_list.append(target_embeds)

                    inputs_embeds = torch.cat(inputs_embeds_list, dim=1) # [BS, prefix_len + seq_len, model_dim]

                    # Forward pass
                    outputs = self.base_model(inputs_embeds=inputs_embeds)
                    
                    # Align causal prediction shift
                    prefix_len = inputs_embeds.size(1) - targets.size(1)
                    logits = outputs.logits[:, prefix_len - 1 : -1, :] # shape [BS, seq_len, vocab_size]
                    
                    lm_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), 
                        targets.reshape(-1),
                        ignore_index=-100
                    )
            except Exception as e:
                # Log warning once per training session if base model interface fails
                logger.debug(f"Language model forward step warning: {e}")

        # 4. Multi-task loss combination
        loss_val = 0.0
        terms = 0
        if contrastive_loss is not None:
            loss_val += contrastive_loss
            terms += 1
        if lm_loss is not None:
            loss_val += lm_loss
            terms += 1

        if terms > 0:
            return loss_val / terms

        # Single-modality uniformity fallback
        proj = projected_v if projected_v is not None else projected_a
        if proj.size(0) <= 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        p_norm = F.normalize(proj, dim=-1)
        sq_pdist = torch.pdist(p_norm, p=2).pow(2)
        return sq_pdist.mul(-2).exp().mean().log()

    def save_checkpoint(self, path: str, include_states: bool = True):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convert state dicts to FP16 to reduce file size on disk
        vision_sd = {k: v.half() if torch.is_tensor(v) else v for k, v in self.vision_projector.state_dict().items()}
        audio_sd = {k: v.half() if torch.is_tensor(v) else v for k, v in self.audio_projector.state_dict().items()}
        
        payload = {
            "vision_projector": vision_sd,
            "audio_projector": audio_sd,
        }
        if include_states:
            payload["optimizer"] = self.optimizer.state_dict()
            if self.scheduler:
                payload["scheduler"] = self.scheduler.state_dict()
            payload["global_step"] = self.global_step
            
        torch.save(payload, path)
        logger.info(f"[Checkpoint] Saved to {path} (FP16, include_states={include_states})")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.vision_projector.load_state_dict(ckpt["vision_projector"])
        self.audio_projector.load_state_dict(ckpt["audio_projector"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt.get("global_step", 0)
        logger.info(f"[Checkpoint] Loaded from {path} (step {self.global_step})")
