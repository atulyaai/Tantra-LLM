"""
Conversational Training Module for Tantra
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm
import time
from pathlib import Path

from ..core.tantra_llm import TantraLLM, TantraConfig
from .data_loader import ConversationDataLoader
from .training_config import TrainingConfig

logger = logging.getLogger(__name__)


class ConversationalTrainer:
    """Trainer for conversational capabilities"""
    
    def __init__(self, config: TrainingConfig, model: Optional[TantraLLM] = None):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        if model is None:
            tantra_config = TantraConfig(
                d_model=config.conversation_max_length // 4,
                n_layers=12,
                n_heads=8,
                vocab_size=50000,
                max_seq_length=config.conversation_max_length
            )
            self.model = TantraLLM(tantra_config)
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # Initialize data loader
        self.data_loader = ConversationDataLoader(config)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.1
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Initialize logging
        self.writer = SummaryWriter(config.logs_path)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"ConversationalTrainer initialized with device: {self.device}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting conversational training...")
        
        train_loader = self.data_loader.get_train_loader()
        val_loader = self.data_loader.get_val_loader()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self._log_metrics(epoch, train_loss, val_loss)
            
            # Save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)
            
            if epoch % self.config.save_steps == 0:
                self._save_checkpoint(epoch, val_loss, is_best=False)
            
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        logger.info("Conversational training completed!")
        self._save_final_model()
    
    def _train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Prepare batch
                loss = self._train_step(batch)
                
                total_loss += loss
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}'
                })
                
                # Log step
                if self.global_step % self.config.logging_steps == 0:
                    self.writer.add_scalar('Train/Loss', loss, self.global_step)
                    self.writer.add_scalar('Train/LearningRate', 
                                         self.optimizer.param_groups[0]['lr'], 
                                         self.global_step)
                
                self.global_step += 1
                
            except Exception as e:
                logger.error(f"Error in training step {batch_idx}: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Single training step"""
        # Prepare inputs
        user_messages = batch["user_messages"]
        assistant_responses = batch["assistant_responses"]
        contexts = batch["contexts"]
        personalities = batch["personalities"]
        
        # Create input for model
        inputs = {
            'text': user_messages[0] if user_messages else "",
            'speech': None,
            'image': None
        }
        
        # Forward pass
        outputs = self.model.forward(inputs)
        
        # Prepare targets (simplified - in practice you'd tokenize properly)
        target_text = assistant_responses[0] if assistant_responses else ""
        
        # Create simple loss (this is a simplified version)
        # In practice, you'd use proper tokenization and language modeling loss
        if target_text:
            # Simple character-level loss for demonstration
            target_tensor = torch.tensor([ord(c) for c in target_text[:100]], 
                                       dtype=torch.long, device=self.device)
            
            # Pad or truncate to match output size
            if len(target_tensor) > outputs['text_logits'].shape[-1]:
                target_tensor = target_tensor[:outputs['text_logits'].shape[-1]]
            else:
                padding = torch.zeros(outputs['text_logits'].shape[-1] - len(target_tensor),
                                    dtype=torch.long, device=self.device)
                target_tensor = torch.cat([target_tensor, padding])
            
            # Calculate loss
            loss = self.criterion(outputs['text_logits'].squeeze(), target_tensor)
        else:
            # If no target, use a small regularization loss
            loss = torch.tensor(0.1, device=self.device, requires_grad=True)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        self.optimizer.step()
        
        return loss.item()
    
    def _validate_epoch(self, val_loader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    # Prepare inputs (same as training)
                    user_messages = batch["user_messages"]
                    assistant_responses = batch["assistant_responses"]
                    
                    inputs = {
                        'text': user_messages[0] if user_messages else "",
                        'speech': None,
                        'image': None
                    }
                    
                    # Forward pass
                    outputs = self.model.forward(inputs)
                    
                    # Calculate loss (same as training)
                    target_text = assistant_responses[0] if assistant_responses else ""
                    
                    if target_text:
                        target_tensor = torch.tensor([ord(c) for c in target_text[:100]], 
                                                   dtype=torch.long, device=self.device)
                        
                        if len(target_tensor) > outputs['text_logits'].shape[-1]:
                            target_tensor = target_tensor[:outputs['text_logits'].shape[-1]]
                        else:
                            padding = torch.zeros(outputs['text_logits'].shape[-1] - len(target_tensor),
                                                dtype=torch.long, device=self.device)
                            target_tensor = torch.cat([target_tensor, padding])
                        
                        loss = self.criterion(outputs['text_logits'].squeeze(), target_tensor)
                    else:
                        loss = torch.tensor(0.1, device=self.device)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in validation step: {e}")
                    continue
        
        return total_loss / max(num_batches, 1)
    
    def _log_metrics(self, epoch: int, train_loss: float, val_loss: float):
        """Log training metrics"""
        self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
        self.writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
        self.writer.add_scalar('Epoch/LearningRate', 
                             self.optimizer.param_groups[0]['lr'], epoch)
        
        # Log model parameters
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Parameters/{name}', param, epoch)
                self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_path, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_path, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
    
    def _save_final_model(self):
        """Save final trained model"""
        final_model_path = os.path.join(self.config.model_output_path, 
                                       f'{self.config.model_name}_conversational.pt')
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'training_info': {
                'final_epoch': self.current_epoch,
                'best_val_loss': self.best_val_loss,
                'total_steps': self.global_step
            }
        }
        
        torch.save(model_state, final_model_path)
        logger.info(f"Final conversational model saved to: {final_model_path}")
    
    def generate_response(self, user_message: str, context: str = "", 
                         personality: str = "helpful") -> str:
        """Generate conversational response"""
        self.model.eval()
        
        with torch.no_grad():
            inputs = {
                'text': user_message,
                'speech': None,
                'image': None
            }
            
            # Add context and personality to memory
            if context:
                self.model.add_to_memory(context, "conversation_context", 0.8)
            
            if personality:
                self.model.add_to_memory(f"Personality: {personality}", "personality", 0.9)
            
            # Generate response
            response = self.model.generate_response(inputs, user_message)
            
            return response
    
    def evaluate_conversation_quality(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate conversation quality metrics"""
        self.model.eval()
        
        metrics = {
            'response_length': [],
            'response_time': [],
            'relevance_score': [],
            'coherence_score': []
        }
        
        with torch.no_grad():
            for sample in test_data:
                start_time = time.time()
                
                user_message = sample.get('user_message', '')
                context = sample.get('context', '')
                personality = sample.get('personality', 'helpful')
                
                # Generate response
                response = self.generate_response(user_message, context, personality)
                
                response_time = time.time() - start_time
                
                # Calculate metrics
                metrics['response_length'].append(len(response))
                metrics['response_time'].append(response_time)
                
                # Simple relevance score (word overlap)
                user_words = set(user_message.lower().split())
                response_words = set(response.lower().split())
                if user_words:
                    relevance = len(user_words.intersection(response_words)) / len(user_words)
                else:
                    relevance = 0.0
                metrics['relevance_score'].append(relevance)
                
                # Simple coherence score (sentence structure)
                sentences = response.split('.')
                coherence = min(len(sentences) / 3.0, 1.0)  # Normalize to 0-1
                metrics['coherence_score'].append(coherence)
        
        # Calculate averages
        avg_metrics = {}
        for key, values in metrics.items():
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
        
        return avg_metrics