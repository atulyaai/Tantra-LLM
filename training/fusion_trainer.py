from __future__ import annotations

"""Train fusion projection layers while freezing base models."""

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import AdamW
from typing import Any, Optional


class FusionTrainingConfig:
    """Training config for fusion projectors."""
    learning_rate: float = 1e-4
    batch_size: int = 4
    max_epochs: int = 10
    checkpoint_dir: str = "checkpoints"


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
            lr=config.learning_rate
        )

    def fit(self, dataset):
        """Training loop: batch → forward → loss → backward → step."""
        if not dataset:
            return
        
        for epoch in range(self.config.max_epochs):
            for batch in dataset:
                # Forward through projectors
                vision_embeds = batch.get("vision_embeds")
                audio_embeds = batch.get("audio_embeds")
                targets = batch.get("target_ids")

                if vision_embeds is not None:
                    projected_v = self.vision_projector(vision_embeds)
                if audio_embeds is not None:
                    projected_a = self.audio_projector(audio_embeds)

                # Compute loss (simplified stub)
                # TODO: Implement full forward pass with base model + projector pipeline

                # Backward pass
                self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()

            # Save checkpoint after each epoch
            self.save_checkpoint(f"checkpoints/epoch_{epoch}.pt")

    def save_checkpoint(self, path: str):
        """Save projector weights."""
        torch.save({
            "vision_projector": self.vision_projector.state_dict(),
            "audio_projector": self.audio_projector.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)


