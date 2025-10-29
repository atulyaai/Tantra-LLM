"""
Unified Multimodal Fusion System
Single, consistent approach to multimodal fusion for SpikingBrain.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityProjector(nn.Module):
    """Projects modality embeddings to model dimension."""
    
    def __init__(self, input_dim: int, model_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.projection(x)


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism."""
    
    def __init__(self, model_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(model_dim, model_dim)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections
        q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)
        
        # Output projection
        output = self.out_proj(context)
        
        return output


class FusionLayer(nn.Module):
    """Single fusion layer combining modalities."""
    
    def __init__(self, model_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(model_dim, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        
    def forward(
        self, 
        text_embeds: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Start with text embeddings
        fused = text_embeds
        
        # Fuse with vision if available
        if vision_embeds is not None:
            # Text attends to vision
            vision_context = self.cross_attention(fused, vision_embeds, vision_embeds)
            fused = self.norm1(fused + vision_context)
        
        # Fuse with audio if available
        if audio_embeds is not None:
            # Text attends to audio
            audio_context = self.cross_attention(fused, audio_embeds, audio_embeds)
            fused = self.norm1(fused + audio_context)
        
        # Feed-forward network
        ffn_out = self.ffn(fused)
        fused = self.norm2(fused + ffn_out)
        
        return fused


class UnifiedMultimodalFusion(nn.Module):
    """Unified multimodal fusion system."""
    
    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        audio_dim: int,
        model_dim: int,
        num_fusion_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.model_dim = model_dim
        
        # Modality projectors
        self.vision_projector = ModalityProjector(vision_dim, model_dim, dropout)
        self.audio_projector = ModalityProjector(audio_dim, model_dim, dropout)
        self.text_projector = ModalityProjector(text_dim, model_dim, dropout)
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            FusionLayer(model_dim, num_heads, dropout)
            for _ in range(num_fusion_layers)
        ])
        
        # Output projection back to text dimension
        self.output_proj = nn.Linear(model_dim, text_dim)
        
    def forward(
        self,
        text_embeds: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = text_embeds.size(0)
        
        # Project all modalities to model dimension
        text_proj = self.text_projector(text_embeds)
        
        # Project vision if available
        if vision_embeds is not None:
            vision_proj = self.vision_projector(vision_embeds)
        else:
            vision_proj = None
        
        # Project audio if available
        if audio_embeds is not None:
            audio_proj = self.audio_projector(audio_embeds)
        else:
            audio_proj = None
        
        # Apply fusion layers
        fused = text_proj
        for fusion_layer in self.fusion_layers:
            fused = fusion_layer(fused, vision_proj, audio_proj)
        
        # Project back to text dimension
        output = self.output_proj(fused)
        
        return output


class TokenStreamFusion(nn.Module):
    """Token-level fusion for streaming generation."""
    
    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__()
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        
        # Special token embeddings
        self.vision_start_embed = nn.Parameter(torch.randn(model_dim))
        self.vision_end_embed = nn.Parameter(torch.randn(model_dim))
        self.audio_start_embed = nn.Parameter(torch.randn(model_dim))
        self.audio_end_embed = nn.Parameter(torch.randn(model_dim))
        
        # Modality gates
        self.vision_gate = nn.Linear(model_dim, 1)
        self.audio_gate = nn.Linear(model_dim, 1)
        
    def forward(
        self,
        text_embeds: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return fused embeddings and attention mask."""
        batch_size, seq_len, model_dim = text_embeds.size()
        
        # Create attention mask
        attention_mask = torch.ones(batch_size, seq_len, device=text_embeds.device)
        
        # Insert vision embeddings if available
        if vision_embeds is not None:
            # Add vision start token
            vision_start = self.vision_start_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, model_dim)
            text_embeds = torch.cat([text_embeds, vision_start], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=text_embeds.device)], dim=1)
            
            # Add vision embeddings
            text_embeds = torch.cat([text_embeds, vision_embeds], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, vision_embeds.size(1), device=text_embeds.device)], dim=1)
            
            # Add vision end token
            vision_end = self.vision_end_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, model_dim)
            text_embeds = torch.cat([text_embeds, vision_end], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=text_embeds.device)], dim=1)
        
        # Insert audio embeddings if available
        if audio_embeds is not None:
            # Add audio start token
            audio_start = self.audio_start_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, model_dim)
            text_embeds = torch.cat([text_embeds, audio_start], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=text_embeds.device)], dim=1)
            
            # Add audio embeddings
            text_embeds = torch.cat([text_embeds, audio_embeds], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, audio_embeds.size(1), device=text_embeds.device)], dim=1)
            
            # Add audio end token
            audio_end = self.audio_end_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, model_dim)
            text_embeds = torch.cat([text_embeds, audio_end], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=text_embeds.device)], dim=1)
        
        return text_embeds, attention_mask


class FusionOrchestrator:
    """Orchestrates multimodal fusion with proper error handling."""
    
    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        audio_dim: int,
        model_dim: int,
        fusion_type: str = "unified"
    ):
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.model_dim = model_dim
        self.fusion_type = fusion_type
        
        # Initialize fusion system
        if fusion_type == "unified":
            self.fusion = UnifiedMultimodalFusion(
                text_dim, vision_dim, audio_dim, model_dim
            )
        elif fusion_type == "stream":
            self.fusion = TokenStreamFusion(model_dim, 50257)  # Default vocab size
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def fuse(
        self,
        text_embeds: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Fuse multimodal embeddings."""
        try:
            if self.fusion_type == "unified":
                return self.fusion(text_embeds, vision_embeds, audio_embeds)
            elif self.fusion_type == "stream":
                return self.fusion(text_embeds, vision_embeds, audio_embeds)
        except Exception as e:
            # Fallback to text-only
            if self.fusion_type == "stream":
                return text_embeds, torch.ones(text_embeds.size(0), text_embeds.size(1), device=text_embeds.device)
            else:
                return text_embeds