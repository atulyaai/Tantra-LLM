from __future__ import annotations

"""Train fusion projection layers while freezing base models."""

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import AdamW
from typing import Any, Optional

from training.training_config import FusionTrainingConfig


class FusionTrainer:
    """Train projectors while base model stays frozen."""

    def __init__(self, config: FusionTrainingConfig, vision_projector: Module, audio_projector: Module, base_model: Any):
        self.config = config
        self.vision_projector = vision_projector
        self.audio_projector = audio_projector
        self.base_model = base_model
        self.loss_fn = CrossEntropyLoss()
        
        # Freeze base model
        if self.base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Only train projectors
        self.optimizer = AdamW(
            list(vision_projector.parameters()) + list(audio_projector.parameters()),
            lr=config.lr
        )

    def fit(self, dataset):
        """Training loop: batch → forward → loss → backward → step."""
        if not dataset:
            return
        
        for epoch in range(self.config.epochs):
            for batch in dataset:
                # Forward through projectors
                vision_embeds = batch.get("vision_embeds")
                audio_embeds = batch.get("audio_embeds")
                targets = batch.get("target_ids")

                 # Compute loss (simplified stub)
                loss = None
                proj_sum = 0.0
                has_proj = False
                if vision_embeds is not None and hasattr(self, 'vision_projector'):
                    projected_v = self.vision_projector(vision_embeds)
                    proj_sum += projected_v.sum()
                    has_proj = True
                if audio_embeds is not None and hasattr(self, 'audio_projector'):
                    projected_a = self.audio_projector(audio_embeds)
                    proj_sum += projected_a.sum()
                    has_proj = True

                if has_proj:
                    loss = proj_sum * 1e-4
                    
                    if targets is not None and self.base_model:
                        try:
                            # Try model-specific forward pass if base_model supports inputs_embeds
                            pass
                        except Exception:
                            pass

                if loss is not None:
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


            # Save checkpoint after each epoch
            self.save_checkpoint(f"checkpoints/epoch_{epoch}.pt")

    def save_checkpoint(self, path: str):
        """Save projector weights."""
        torch.save({
            "vision_projector": self.vision_projector.state_dict(),
            "audio_projector": self.audio_projector.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)


