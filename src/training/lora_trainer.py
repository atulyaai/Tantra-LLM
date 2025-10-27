"""
LoRA (Low-Rank Adaptation) Training Module for Tantra
Efficient fine-tuning with low-rank adaptation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA training"""
    
    # LoRA parameters
    rank: int = 16  # LoRA rank
    alpha: float = 32.0  # LoRA scaling factor
    dropout: float = 0.1  # LoRA dropout
    target_modules: List[str] = None  # Modules to apply LoRA to
    
    # Training parameters
    learning_rate: float = 1e-4
    lora_learning_rate: float = 1e-3  # Higher LR for LoRA parameters
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # LoRA-specific settings
    lora_plus: bool = True  # Use LoRA+ scaling
    lora_plus_alpha: float = 16.0
    gradient_checkpointing: bool = True
    
    def __post_init__(self):
        """Initialize default values"""
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]


class LoRALayer(nn.Module):
    """LoRA layer implementation"""
    
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer"""
        return self.dropout(self.lora_B(self.lora_A(x))) * self.scaling


class LoRALinear(nn.Module):
    """LoRA-enhanced linear layer"""
    
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank, alpha, dropout
        )
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation"""
        return self.original_layer(x) + self.lora(x)
    
    def get_lora_parameters(self) -> List[torch.Tensor]:
        """Get LoRA parameters for optimization"""
        return list(self.lora.parameters())


class LoRATransformerBlock(nn.Module):
    """LoRA-enhanced transformer block"""
    
    def __init__(self, original_block, lora_config: LoRAConfig):
        super().__init__()
        self.original_block = original_block
        self.lora_config = lora_config
        
        # Apply LoRA to target modules
        self._apply_lora_to_modules()
    
    def _apply_lora_to_modules(self):
        """Apply LoRA to target modules in the transformer block"""
        for name, module in self.original_block.named_modules():
            if any(target in name for target in self.lora_config.target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA-enhanced version
                    lora_module = LoRALinear(
                        module, 
                        self.lora_config.rank,
                        self.lora_config.alpha,
                        self.lora_config.dropout
                    )
                    # Replace in parent module
                    parent_name = '.'.join(name.split('.')[:-1])
                    if parent_name:
                        parent_module = self.original_block
                        for part in parent_name.split('.'):
                            parent_module = getattr(parent_module, part)
                        setattr(parent_module, name.split('.')[-1], lora_module)
                    else:
                        # Root level module
                        setattr(self.original_block, name, lora_module)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through LoRA-enhanced transformer block"""
        return self.original_block(x, **kwargs)
    
    def get_lora_parameters(self) -> List[torch.Tensor]:
        """Get all LoRA parameters from this block"""
        lora_params = []
        for module in self.original_block.modules():
            if isinstance(module, LoRALinear):
                lora_params.extend(module.get_lora_parameters())
        return lora_params


class LoRATrainer:
    """LoRA trainer for efficient fine-tuning"""
    
    def __init__(self, model: nn.Module, config: LoRAConfig, base_config=None):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Apply LoRA to model
        self._apply_lora_to_model()
        
        # Setup optimizers
        self._setup_optimizers()
        
        # Setup logging
        self.writer = SummaryWriter("logs/lora_training")
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        logger.info(f"LoRATrainer initialized with rank {config.rank}")
    
    def _apply_lora_to_model(self):
        """Apply LoRA to the model"""
        # Find transformer blocks and apply LoRA
        for name, module in self.model.named_modules():
            if 'transformer' in name.lower() or 'block' in name.lower():
                if hasattr(module, 'named_children'):
                    # This is likely a transformer block
                    lora_block = LoRATransformerBlock(module, self.config)
                    # Replace in parent module
                    parent_name = '.'.join(name.split('.')[:-1])
                    if parent_name:
                        parent_module = self.model
                        for part in parent_name.split('.'):
                            parent_module = getattr(parent_module, part)
                        setattr(parent_module, name.split('.')[-1], lora_block)
                    else:
                        # Root level module
                        setattr(self.model, name, lora_block)
    
    def _setup_optimizers(self):
        """Setup optimizers for LoRA training"""
        # Get LoRA parameters
        lora_params = []
        base_params = []
        
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                lora_params.append(param)
            else:
                base_params.append(param)
        
        # LoRA optimizer (higher learning rate)
        self.lora_optimizer = optim.AdamW(
            lora_params,
            lr=self.config.lora_learning_rate,
            weight_decay=0.01
        )
        
        # Base model optimizer (lower learning rate)
        self.base_optimizer = optim.AdamW(
            base_params,
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Schedulers
        self.lora_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.lora_optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.lora_learning_rate * 0.1
        )
        
        self.base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.base_optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.learning_rate * 0.1
        )
        
        logger.info(f"Setup optimizers: {len(lora_params)} LoRA params, {len(base_params)} base params")
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Prepare batch
                loss = self._train_step(batch)
                
                total_loss += loss
                num_batches += 1
                
                # Log step
                if self.global_step % 100 == 0:
                    self.writer.add_scalar('Train/Loss', loss, self.global_step)
                    self.writer.add_scalar('Train/LoRALR', 
                                         self.lora_optimizer.param_groups[0]['lr'], 
                                         self.global_step)
                    self.writer.add_scalar('Train/BaseLR', 
                                         self.base_optimizer.param_groups[0]['lr'], 
                                         self.global_step)
                
                self.global_step += 1
                
            except Exception as e:
                logger.error(f"Error in training step {batch_idx}: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Single training step"""
        # Prepare inputs
        inputs = batch.get('input_ids', batch.get('text', ''))
        targets = batch.get('labels', batch.get('target', ''))
        
        # Forward pass
        if hasattr(self.model, 'forward'):
            outputs = self.model.forward(inputs)
        else:
            # Fallback for different model interfaces
            outputs = self.model(inputs)
        
        # Calculate loss
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('text_logits'))
        else:
            logits = outputs
        
        # Simple loss calculation (adapt based on your model)
        if isinstance(targets, str):
            # Convert string target to tensor (simplified)
            target_tensor = torch.tensor([ord(c) for c in targets[:100]], 
                                       dtype=torch.long, device=self.device)
            if len(target_tensor) > logits.shape[-1]:
                target_tensor = target_tensor[:logits.shape[-1]]
            else:
                padding = torch.zeros(logits.shape[-1] - len(target_tensor),
                                    dtype=torch.long, device=self.device)
                target_tensor = torch.cat([target_tensor, padding])
            
            loss = nn.CrossEntropyLoss()(logits.squeeze(), target_tensor)
        else:
            loss = nn.MSELoss()(logits, targets)
        
        # Backward pass
        self.lora_optimizer.zero_grad()
        self.base_optimizer.zero_grad()
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        # Update parameters
        self.lora_optimizer.step()
        self.base_optimizer.step()
        
        return loss.item()
    
    def train(self, train_loader, val_loader=None):
        """Main training loop"""
        logger.info("Starting LoRA training...")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = 0.0
            if val_loader:
                val_loss = self.validate_epoch(val_loader)
            
            # Update schedulers
            self.lora_scheduler.step()
            self.base_scheduler.step()
            
            # Log epoch
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(f"best_lora_model.pt")
    
    def validate_epoch(self, val_loader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    # Prepare inputs
                    inputs = batch.get('input_ids', batch.get('text', ''))
                    targets = batch.get('labels', batch.get('target', ''))
                    
                    # Forward pass
                    if hasattr(self.model, 'forward'):
                        outputs = self.model.forward(inputs)
                    else:
                        outputs = self.model(inputs)
                    
                    # Calculate loss
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs.get('text_logits'))
                    else:
                        logits = outputs
                    
                    if isinstance(targets, str):
                        target_tensor = torch.tensor([ord(c) for c in targets[:100]], 
                                                   dtype=torch.long, device=self.device)
                        if len(target_tensor) > logits.shape[-1]:
                            target_tensor = target_tensor[:logits.shape[-1]]
                        else:
                            padding = torch.zeros(logits.shape[-1] - len(target_tensor),
                                                dtype=torch.long, device=self.device)
                            target_tensor = torch.cat([target_tensor, padding])
                        
                        loss = nn.CrossEntropyLoss()(logits.squeeze(), target_tensor)
                    else:
                        loss = nn.MSELoss()(logits, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in validation step: {e}")
                    continue
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, path: str):
        """Save LoRA checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'lora_optimizer_state_dict': self.lora_optimizer.state_dict(),
            'base_optimizer_state_dict': self.base_optimizer.state_dict(),
            'lora_scheduler_state_dict': self.lora_scheduler.state_dict(),
            'base_scheduler_state_dict': self.base_scheduler.state_dict(),
            'config': self.config.__dict__,
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss
        }
        
        torch.save(checkpoint, path)
        logger.info(f"LoRA checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load LoRA checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.lora_optimizer.load_state_dict(checkpoint['lora_optimizer_state_dict'])
        self.base_optimizer.load_state_dict(checkpoint['base_optimizer_state_dict'])
        self.lora_scheduler.load_state_dict(checkpoint['lora_scheduler_state_dict'])
        self.base_scheduler.load_state_dict(checkpoint['base_scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"LoRA checkpoint loaded from {path}")
    
    def get_lora_parameters_count(self) -> int:
        """Get count of LoRA parameters"""
        lora_count = 0
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                lora_count += param.numel()
        return lora_count
    
    def get_total_parameters_count(self) -> int:
        """Get total parameter count"""
        return sum(p.numel() for p in self.model.parameters())
    
    def get_parameter_efficiency(self) -> float:
        """Get parameter efficiency (LoRA params / total params)"""
        lora_count = self.get_lora_parameters_count()
        total_count = self.get_total_parameters_count()
        return lora_count / total_count if total_count > 0 else 0.0