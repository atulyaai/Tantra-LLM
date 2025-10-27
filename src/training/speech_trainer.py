"""
Speech Training Module for Tantra
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import librosa
import soundfile as sf
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from tqdm import tqdm
import time
from pathlib import Path

from ..core.tantra_llm import TantraLLM, TantraConfig
from .data_loader import SpeechDataLoader
from .training_config import TrainingConfig

logger = logging.getLogger(__name__)


class SpeechEncoder(nn.Module):
    """Speech encoder for converting audio to embeddings"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Convolutional layers for audio processing
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=5, padding=7),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=15, stride=5, padding=7),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=15, stride=5, padding=7),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Projection to model dimension
        self.projection = nn.Linear(1024, config.conversation_max_length // 4)
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass for speech encoding"""
        # audio shape: [batch, time]
        batch_size = audio.shape[0]
        
        # Add channel dimension
        audio = audio.unsqueeze(1)  # [batch, 1, time]
        
        # Convolutional processing
        conv_out = self.conv_layers(audio)  # [batch, 256, time']
        
        # Transpose for LSTM
        conv_out = conv_out.transpose(1, 2)  # [batch, time', 256]
        
        # LSTM processing
        lstm_out, _ = self.lstm(conv_out)  # [batch, time', 1024]
        
        # Global average pooling
        pooled = torch.mean(lstm_out, dim=1)  # [batch, 1024]
        
        # Project to model dimension
        output = self.projection(pooled)  # [batch, d_model]
        
        return output


class SpeechDecoder(nn.Module):
    """Speech decoder for converting embeddings to audio"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Projection from model dimension
        self.projection = nn.Linear(config.conversation_max_length // 4, 512)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Convolutional layers for audio generation
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=15, stride=5, padding=7),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.ConvTranspose1d(256, 128, kernel_size=15, stride=5, padding=7),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 1, kernel_size=15, stride=5, padding=7),
            nn.Tanh()
        )
        
    def forward(self, embeddings: torch.Tensor, target_length: int) -> torch.Tensor:
        """Forward pass for speech decoding"""
        batch_size = embeddings.shape[0]
        
        # Project embeddings
        projected = self.projection(embeddings)  # [batch, 512]
        
        # Expand to sequence
        projected = projected.unsqueeze(1).repeat(1, target_length, 1)  # [batch, time, 512]
        
        # LSTM processing
        lstm_out, _ = self.lstm(projected)  # [batch, time, 512]
        
        # Transpose for conv layers
        lstm_out = lstm_out.transpose(1, 2)  # [batch, 512, time]
        
        # Convolutional processing
        audio = self.conv_layers(lstm_out)  # [batch, 1, time]
        
        return audio.squeeze(1)  # [batch, time]


class SpeechTrainer:
    """Trainer for speech capabilities"""
    
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
        
        # Initialize speech components
        self.speech_encoder = SpeechEncoder(config).to(self.device)
        self.speech_decoder = SpeechDecoder(config).to(self.device)
        
        # Initialize data loader
        self.data_loader = SpeechDataLoader(config)
        
        # Initialize optimizers
        self.model_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        self.speech_optimizer = optim.AdamW(
            list(self.speech_encoder.parameters()) + list(self.speech_decoder.parameters()),
            lr=config.learning_rate * 2,  # Higher learning rate for speech components
            weight_decay=0.01
        )
        
        # Initialize schedulers
        self.model_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.model_optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.1
        )
        
        self.speech_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.speech_optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.1
        )
        
        # Initialize loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Initialize logging
        self.writer = SummaryWriter(os.path.join(config.logs_path, "speech"))
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"SpeechTrainer initialized with device: {self.device}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting speech training...")
        
        train_loader = self.data_loader.get_train_loader()
        val_loader = self.data_loader.get_val_loader()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader)
            
            # Update learning rates
            self.model_scheduler.step()
            self.speech_scheduler.step()
            
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
        
        logger.info("Speech training completed!")
        self._save_final_model()
    
    def _train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        self.speech_encoder.train()
        self.speech_decoder.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Speech Training Epoch {self.current_epoch+1}")
        
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
                    self.writer.add_scalar('Train/ModelLR', 
                                         self.model_optimizer.param_groups[0]['lr'], 
                                         self.global_step)
                    self.writer.add_scalar('Train/SpeechLR', 
                                         self.speech_optimizer.param_groups[0]['lr'], 
                                         self.global_step)
                
                self.global_step += 1
                
            except Exception as e:
                logger.error(f"Error in speech training step {batch_idx}: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Single training step"""
        audio = batch["audio"].to(self.device)
        texts = batch["texts"]
        mel_spectrograms = batch["mel_spectrograms"].to(self.device)
        
        # Encode speech to embeddings
        speech_embeddings = self.speech_encoder(audio)
        
        # Process through Tantra model
        inputs = {
            'text': texts[0] if texts else "",
            'speech': audio[0].cpu().numpy() if len(audio) > 0 else None,
            'image': None
        }
        
        model_outputs = self.model.forward(inputs)
        model_embeddings = model_outputs['embeddings']
        
        # Combine speech and model embeddings
        combined_embeddings = speech_embeddings + model_embeddings.squeeze()
        
        # Decode back to speech
        target_length = audio.shape[1]
        reconstructed_audio = self.speech_decoder(combined_embeddings, target_length)
        
        # Calculate losses
        # Reconstruction loss
        recon_loss = self.mse_loss(reconstructed_audio, audio)
        
        # Spectral loss (mel spectrogram)
        if mel_spectrograms.shape[1] > 0:
            # Convert reconstructed audio to mel spectrogram
            recon_mel = self._audio_to_mel(reconstructed_audio)
            spectral_loss = self.mse_loss(recon_mel, mel_spectrograms)
        else:
            spectral_loss = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = recon_loss + 0.1 * spectral_loss
        
        # Backward pass
        self.model_optimizer.zero_grad()
        self.speech_optimizer.zero_grad()
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(
            list(self.speech_encoder.parameters()) + list(self.speech_decoder.parameters()),
            self.config.max_grad_norm
        )
        
        self.model_optimizer.step()
        self.speech_optimizer.step()
        
        return total_loss.item()
    
    def _validate_epoch(self, val_loader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        self.speech_encoder.eval()
        self.speech_decoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    audio = batch["audio"].to(self.device)
                    texts = batch["texts"]
                    mel_spectrograms = batch["mel_spectrograms"].to(self.device)
                    
                    # Encode speech
                    speech_embeddings = self.speech_encoder(audio)
                    
                    # Process through model
                    inputs = {
                        'text': texts[0] if texts else "",
                        'speech': audio[0].cpu().numpy() if len(audio) > 0 else None,
                        'image': None
                    }
                    
                    model_outputs = self.model.forward(inputs)
                    model_embeddings = model_outputs['embeddings']
                    
                    # Combine embeddings
                    combined_embeddings = speech_embeddings + model_embeddings.squeeze()
                    
                    # Decode speech
                    target_length = audio.shape[1]
                    reconstructed_audio = self.speech_decoder(combined_embeddings, target_length)
                    
                    # Calculate losses
                    recon_loss = self.mse_loss(reconstructed_audio, audio)
                    
                    if mel_spectrograms.shape[1] > 0:
                        recon_mel = self._audio_to_mel(reconstructed_audio)
                        spectral_loss = self.mse_loss(recon_mel, mel_spectrograms)
                    else:
                        spectral_loss = torch.tensor(0.0, device=self.device)
                    
                    total_loss += (recon_loss + 0.1 * spectral_loss).item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in speech validation step: {e}")
                    continue
        
        return total_loss / max(num_batches, 1)
    
    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel spectrogram"""
        # Convert to numpy for librosa
        audio_np = audio.cpu().numpy()
        
        mel_specs = []
        for i in range(audio_np.shape[0]):
            mel_spec = librosa.feature.melspectrogram(
                y=audio_np[i],
                sr=self.config.speech_sample_rate,
                n_fft=self.config.speech_n_fft,
                hop_length=self.config.speech_hop_length,
                n_mels=self.config.speech_n_mels,
                fmin=self.config.speech_f_min,
                fmax=self.config.speech_f_max
            )
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_specs.append(mel_spec)
        
        # Stack and convert back to tensor
        mel_specs = np.stack(mel_specs)
        return torch.tensor(mel_specs, dtype=torch.float32, device=self.device)
    
    def _log_metrics(self, epoch: int, train_loss: float, val_loss: float):
        """Log training metrics"""
        self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
        self.writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
        self.writer.add_scalar('Epoch/ModelLR', 
                             self.model_optimizer.param_groups[0]['lr'], epoch)
        self.writer.add_scalar('Epoch/SpeechLR', 
                             self.speech_optimizer.param_groups[0]['lr'], epoch)
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'speech_encoder_state_dict': self.speech_encoder.state_dict(),
            'speech_decoder_state_dict': self.speech_decoder.state_dict(),
            'model_optimizer_state_dict': self.model_optimizer.state_dict(),
            'speech_optimizer_state_dict': self.speech_optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_path, f'speech_checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_path, 'best_speech_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"New best speech model saved with validation loss: {val_loss:.4f}")
    
    def _save_final_model(self):
        """Save final trained model"""
        final_model_path = os.path.join(self.config.model_output_path, 
                                       f'{self.config.model_name}_speech.pt')
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'speech_encoder_state_dict': self.speech_encoder.state_dict(),
            'speech_decoder_state_dict': self.speech_decoder.state_dict(),
            'config': self.config.to_dict(),
            'training_info': {
                'final_epoch': self.current_epoch,
                'best_val_loss': self.best_val_loss,
                'total_steps': self.global_step
            }
        }
        
        torch.save(model_state, final_model_path)
        logger.info(f"Final speech model saved to: {final_model_path}")
    
    def text_to_speech(self, text: str, voice_style: str = "neutral") -> np.ndarray:
        """Convert text to speech"""
        self.model.eval()
        self.speech_encoder.eval()
        self.speech_decoder.eval()
        
        with torch.no_grad():
            # Process text through model
            inputs = {
                'text': text,
                'speech': None,
                'image': None
            }
            
            model_outputs = self.model.forward(inputs)
            text_embeddings = model_outputs['embeddings'].squeeze()
            
            # Add voice style to embeddings
            if voice_style == "excited":
                style_vector = torch.ones_like(text_embeddings) * 0.5
            elif voice_style == "calm":
                style_vector = torch.ones_like(text_embeddings) * -0.3
            else:  # neutral
                style_vector = torch.zeros_like(text_embeddings)
            
            combined_embeddings = text_embeddings + style_vector
            
            # Generate speech
            target_length = int(self.config.speech_sample_rate * 3.0)  # 3 seconds
            audio = self.speech_decoder(combined_embeddings.unsqueeze(0), target_length)
            
            return audio[0].cpu().numpy()
    
    def speech_to_text(self, audio: np.ndarray) -> str:
        """Convert speech to text"""
        self.model.eval()
        self.speech_encoder.eval()
        
        with torch.no_grad():
            # Convert to tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32, device=self.device)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Encode speech
            speech_embeddings = self.speech_encoder(audio_tensor)
            
            # Process through model
            inputs = {
                'text': "",
                'speech': audio,
                'image': None
            }
            
            model_outputs = self.model.forward(inputs)
            
            # Generate text response
            response = self.model.generate_response(inputs, "Transcribe the speech")
            
            return response
    
    def evaluate_speech_quality(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate speech quality metrics"""
        self.model.eval()
        self.speech_encoder.eval()
        self.speech_decoder.eval()
        
        metrics = {
            'reconstruction_mse': [],
            'spectral_distance': [],
            'inference_time': []
        }
        
        with torch.no_grad():
            for sample in test_data:
                start_time = time.time()
                
                audio = sample.get('audio', np.array([]))
                text = sample.get('text', '')
                
                if len(audio) == 0:
                    continue
                
                # Convert to tensor
                audio_tensor = torch.tensor(audio, dtype=torch.float32, device=self.device)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                # Encode and decode
                speech_embeddings = self.speech_encoder(audio_tensor)
                reconstructed = self.speech_decoder(speech_embeddings, len(audio))
                
                # Calculate metrics
                mse = self.mse_loss(reconstructed, audio_tensor).item()
                metrics['reconstruction_mse'].append(mse)
                
                # Spectral distance
                original_mel = self._audio_to_mel(audio_tensor)
                recon_mel = self._audio_to_mel(reconstructed)
                spectral_dist = self.mse_loss(recon_mel, original_mel).item()
                metrics['spectral_distance'].append(spectral_dist)
                
                inference_time = time.time() - start_time
                metrics['inference_time'].append(inference_time)
        
        # Calculate averages
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)
            else:
                avg_metrics[f'avg_{key}'] = 0.0
                avg_metrics[f'std_{key}'] = 0.0
        
        return avg_metrics