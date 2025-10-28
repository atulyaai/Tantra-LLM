"""
Multimodal Fusion System
Handles fusion of text, vision, and audio modalities for SpikingBrain.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """Base class for modality encoders."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VisionEncoder(ModalityEncoder):
    """Vision encoder using CNN + Transformer."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_patches: int = 196):
        super().__init__(input_dim, hidden_dim)
        self.num_patches = num_patches
        
        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Patch embedding
        self.patch_embed = nn.Linear(256, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, channels, height, width]
        batch_size = x.size(0)
        
        # Extract features
        features = self.conv_layers(x)  # [batch_size, 256, H', W']
        
        # Reshape to patches
        features = features.flatten(2).transpose(1, 2)  # [batch_size, num_patches, 256]
        
        # Project to hidden dimension
        patches = self.patch_embed(features)  # [batch_size, num_patches, hidden_dim]
        
        # Add positional embeddings
        patches = patches + self.pos_embed
        
        # Apply transformer
        encoded = self.transformer(patches)  # [batch_size, num_patches, hidden_dim]
        
        # Global average pooling
        pooled = encoded.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Output projection
        output = self.output_proj(pooled)
        
        return output


class AudioEncoder(ModalityEncoder):
    """Audio encoder using CNN + Transformer."""
    
    def __init__(self, input_dim: int, hidden_dim: int, sequence_length: int = 1000):
        super().__init__(input_dim, hidden_dim)
        self.sequence_length = sequence_length
        
        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        # Calculate output sequence length
        conv_output_length = sequence_length
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv1d):
                conv_output_length = (conv_output_length + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, conv_output_length, hidden_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, sequence_length] or [batch_size, 1, sequence_length]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        batch_size = x.size(0)
        
        # Extract features
        features = self.conv_layers(x)  # [batch_size, 256, conv_output_length]
        features = features.transpose(1, 2)  # [batch_size, conv_output_length, 256]
        
        # Project to hidden dimension
        projected = F.linear(features, torch.randn(256, self.hidden_dim, device=x.device))
        
        # Add positional embeddings
        projected = projected + self.pos_embed[:, :projected.size(1), :]
        
        # Apply transformer
        encoded = self.transformer(projected)  # [batch_size, conv_output_length, hidden_dim]
        
        # Global average pooling
        pooled = encoded.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Output projection
        output = self.output_proj(pooled)
        
        return output


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections
        q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Output projection
        output = self.output_proj(context)
        
        return output


class MultimodalFusion(nn.Module):
    """Multimodal fusion system for SpikingBrain."""
    
    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        audio_dim: int,
        hidden_dim: int,
        num_fusion_layers: int = 4
    ):
        super().__init__()
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        
        # Modality encoders
        self.vision_encoder = VisionEncoder(vision_dim, hidden_dim)
        self.audio_encoder = AudioEncoder(audio_dim, hidden_dim)
        
        # Projection layers
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Cross-modal attention layers
        self.fusion_layers = nn.ModuleList([
            CrossModalAttention(hidden_dim) for _ in range(num_fusion_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, text_dim)
    
    def forward(
        self,
        text_embeds: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = text_embeds.size(0)
        
        # Project all modalities to hidden dimension
        text_proj = self.text_proj(text_embeds)
        
        # Process vision if available
        if vision_embeds is not None:
            vision_proj = self.vision_encoder(vision_embeds)
        else:
            vision_proj = torch.zeros(batch_size, self.hidden_dim, device=text_embeds.device)
        
        # Process audio if available
        if audio_embeds is not None:
            audio_proj = self.audio_encoder(audio_embeds)
        else:
            audio_proj = torch.zeros(batch_size, self.hidden_dim, device=text_embeds.device)
        
        # Fuse modalities using cross-modal attention
        fused = text_proj
        
        for fusion_layer in self.fusion_layers:
            # Text attends to vision
            text_vision = fusion_layer(fused, vision_proj, vision_proj)
            fused = fused + text_vision
            
            # Text attends to audio
            text_audio = fusion_layer(fused, audio_proj, audio_proj)
            fused = fused + text_audio
        
        # Project back to text dimension
        output = self.output_proj(fused)
        
        return output


class MultimodalTokenizer:
    """Tokenizer for multimodal inputs."""
    
    def __init__(self, text_tokenizer, vision_token_id: int = 50258, audio_token_id: int = 50259):
        self.text_tokenizer = text_tokenizer
        self.vision_token_id = vision_token_id
        self.audio_token_id = audio_token_id
        
        # Add special tokens
        special_tokens = ["<vision>", "<audio>", "<vision_end>", "<audio_end>"]
        self.text_tokenizer.add_tokens(special_tokens)
    
    def encode_multimodal(
        self,
        text: str,
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode multimodal input."""
        # Encode text
        text_tokens = self.text_tokenizer.encode(text, add_special_tokens=True)
        
        # Add modality tokens
        input_ids = text_tokens.copy()
        modality_embeds = []
        
        if vision_embeds is not None:
            vision_start_id = self.text_tokenizer.convert_tokens_to_ids("<vision>")
            vision_end_id = self.text_tokenizer.convert_tokens_to_ids("<vision_end>")
            input_ids.append(vision_start_id)
            modality_embeds.append(vision_embeds)
            input_ids.append(vision_end_id)
        
        if audio_embeds is not None:
            audio_start_id = self.text_tokenizer.convert_tokens_to_ids("<audio>")
            audio_end_id = self.text_tokenizer.convert_tokens_to_ids("<audio_end>")
            input_ids.append(audio_start_id)
            modality_embeds.append(audio_embeds)
            input_ids.append(audio_end_id)
        
        return torch.tensor(input_ids, dtype=torch.long), modality_embeds