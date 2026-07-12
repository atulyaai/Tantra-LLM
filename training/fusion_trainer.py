from __future__ import annotations

"""Train fusion projection layers while freezing base models, using performance optimizations."""

import torch
import torch.nn.functional as F
import logging
import time
from torch.nn import Module, CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Any, Optional, Dict, List

from training.training_config import FusionTrainingConfig

logger = logging.getLogger(__name__)


class FusionProjector(Module):
    """Simple MLP projector: maps encoder embedding space to base model hidden space."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or (input_dim + output_dim) // 2
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"[Projector] {input_dim} → {hidden_dim} → {output_dim} ({total:,} params)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionTrainer:
    """
    Train vision and audio projection layers while the base model remains frozen.
    Employs pre-computed embedding caching, mixed precision, cosine LR, and early stopping.
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
        
        # Compile projectors if requested and supported
        if use_compile and hasattr(torch, "compile"):
            try:
                logger.info("Compiling vision and audio projectors...")
                self.vision_projector = torch.compile(vision_projector).to(self.device)
                self.audio_projector = torch.compile(audio_projector).to(self.device)
            except Exception as e:
                logger.warning(f"Failed to compile projectors: {e}. Falling back to standard mode.")
                self.vision_projector = vision_projector.to(self.device)
                self.audio_projector = audio_projector.to(self.device)
        else:
            self.vision_projector = vision_projector.to(self.device)
            self.audio_projector = audio_projector.to(self.device)
            
        self.base_model = base_model
        self.loss_fn = CrossEntropyLoss()
        self.patience = patience
        
        # Freeze base model parameters
        if self.base_model:
            if hasattr(self.base_model, "to"):
                self.base_model = self.base_model.to(self.device)
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Optimize only projector parameters
        trainable_params = list(self.vision_projector.parameters()) + list(self.audio_projector.parameters())
        total_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"[Trainer] Total trainable parameters: {total_trainable:,}")
        
        self.optimizer = AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Setup Cosine Annealing scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config.epochs, 
            eta_min=1e-6
        )
        
        # Mixed precision setup
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def fit(self, train_dataset, val_dataset=None) -> Dict[str, List[float]]:
        """
        Runs the training loop over PyTorch Datasets.
        
        Args:
            train_dataset: Dataset yielding dicts with vision_embeds, audio_embeds, target_ids
            val_dataset: Optional validation dataset
            
        Returns:
            dict with train_loss and val_loss history
        """
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        history = {"train_loss": [], "val_loss": [], "lr": [], "epoch_time_s": []}
        best_val_loss = float("inf")
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print(f"  FUSION PROJECTOR TRAINING")
        print(f"  Device: {self.device}")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset) if val_dataset else 0}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  LR: {self.config.lr}  Weight Decay: {self.config.weight_decay}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # ── Train ──
            self.vision_projector.train()
            self.audio_projector.train()
            
            epoch_loss = 0.0
            steps = 0
            
            for batch in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                
                vision_embeds = batch.get("vision_embeds")
                audio_embeds = batch.get("audio_embeds")
                targets = batch.get("target_ids")
                
                if vision_embeds is not None:
                    vision_embeds = vision_embeds.to(self.device)
                if audio_embeds is not None:
                    audio_embeds = audio_embeds.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)

                device_type = "cuda" if self.use_amp else "cpu"
                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    loss = self._forward_step(vision_embeds, audio_embeds, targets)

                if loss is not None:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                        
                    epoch_loss += loss.item()
                    steps += 1
            
            avg_train_loss = epoch_loss / max(steps, 1)
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(avg_train_loss)
            history["lr"].append(current_lr)
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            history["epoch_time_s"].append(epoch_time)
            
            # -- Validate --
            if val_loader is not None:
                avg_val_loss = self._evaluate(val_loader)
                history["val_loss"].append(avg_val_loss)
                
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs} | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {avg_val_loss:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.2f}s")
                
                # Early Stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    self.save_checkpoint("checkpoints/best_projectors.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"\n  [Early Stopping] Triggered at epoch {epoch+1} (patience={self.patience})")
                        break
            else:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs} | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.2f}s")
        
        # Final summary
        total_time = sum(history["epoch_time_s"])
        print(f"\n{'='*60}")
        print(f"  Training Complete")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
        if history["val_loss"]:
            print(f"  Best Val Loss: {best_val_loss:.6f}")
        print(f"{'='*60}\n")
        
        self.save_checkpoint("checkpoints/final_projectors.pt")
        return history

    def _evaluate(self, val_loader) -> float:
        """Evaluates loss over a validation dataset."""
        self.vision_projector.eval()
        self.audio_projector.eval()
        
        total_loss = 0.0
        steps = 0
        device_type = "cuda" if self.use_amp else "cpu"
        
        with torch.no_grad():
            for batch in val_loader:
                vision_embeds = batch.get("vision_embeds")
                audio_embeds = batch.get("audio_embeds")
                targets = batch.get("target_ids")
                
                if vision_embeds is not None:
                    vision_embeds = vision_embeds.to(self.device)
                if audio_embeds is not None:
                    audio_embeds = audio_embeds.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                    
                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    loss = self._forward_step(vision_embeds, audio_embeds, targets)
                
                if loss is not None:
                    total_loss += loss.item()
                    steps += 1
                    
        return total_loss / max(steps, 1)

    def _forward_step(
        self, 
        vision_embeds: Optional[torch.Tensor], 
        audio_embeds: Optional[torch.Tensor], 
        targets: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Compute contrastive alignment loss between projected modality embeddings.
        
        The loss has two components:
        1. Cosine similarity alignment: vision and audio projections should map
           to similar points in the shared embedding space when from the same sample.
        2. Projection regularization: prevent degenerate collapse to zero.
        """
        projected_v = None
        projected_a = None
        
        if vision_embeds is not None:
            projected_v = self.vision_projector(vision_embeds)
            
        if audio_embeds is not None:
            projected_a = self.audio_projector(audio_embeds)

        if projected_v is None and projected_a is None:
            return None
        
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
        # ── Component 1: Cross-modal contrastive alignment ──
        # When both modalities are present, their projections should align
        if projected_v is not None and projected_a is not None:
            # Normalize to unit sphere
            v_norm = F.normalize(projected_v, dim=-1)
            a_norm = F.normalize(projected_a, dim=-1)
            
            # Cosine similarity matrix [batch, batch]
            logits = torch.matmul(v_norm, a_norm.T) / 0.07  # temperature=0.07
            
            # Contrastive target: diagonal entries should be most similar
            batch_size = logits.size(0)
            labels = torch.arange(batch_size, device=self.device)
            
            # Symmetric contrastive loss (CLIP-style)
            loss_v2a = F.cross_entropy(logits, labels)
            loss_a2v = F.cross_entropy(logits.T, labels)
            loss = (loss_v2a + loss_a2v) / 2.0
        
        # ── Component 2: Single-modality reconstruction target ──
        elif projected_v is not None:
            # Self-supervised: predict a simple target from the projection
            v_norm = F.normalize(projected_v, dim=-1)
            # Uniformity loss: projections should not collapse
            sq_pdist = torch.pdist(v_norm, p=2).pow(2)
            uniformity_loss = sq_pdist.mul(-2).exp().mean().log()
            loss = uniformity_loss
            
        elif projected_a is not None:
            a_norm = F.normalize(projected_a, dim=-1)
            sq_pdist = torch.pdist(a_norm, p=2).pow(2)
            uniformity_loss = sq_pdist.mul(-2).exp().mean().log()
            loss = uniformity_loss
                
        return loss

    def save_checkpoint(self, path: str):
        """Saves projector weights."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "vision_projector": self.vision_projector.state_dict(),
            "audio_projector": self.audio_projector.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }, path)
        logger.info(f"[Checkpoint] Saved to {path}")
