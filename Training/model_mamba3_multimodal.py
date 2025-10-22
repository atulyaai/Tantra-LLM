"""
Multi-Modal Mamba 3 Architecture with MoE
Supports Audio/Speech → Text → Vision priority with dynamic vocabulary and compression
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass


@dataclass
class Mamba3Config:
    """Configuration for Mamba 3 Multi-Modal model"""
    # Core architecture
    d_model: int = 768
    n_layers: int = 16
    d_state: int = 64
    d_conv: int = 4
    dropout: float = 0.1
    
    # Multi-modal dimensions
    audio_dim: int = 128
    text_dim: int = 768
    vision_dim: int = 512
    
    # Dynamic vocabulary
    initial_vocab_size: int = 32000
    max_vocab_size: int = 100000
    vocab_growth_threshold: float = 0.8
    
    # MoE configuration
    num_experts: int = 8
    expert_capacity: int = 64
    expert_categories: List[str] = None
    routing_strategy: str = "category_based"  # category_based, learned, random
    
    # Compression
    quantization_bits: int = 8
    pruning_ratio: float = 0.1
    distillation_alpha: float = 0.3
    
    # Modality priority (audio -> text -> vision)
    modality_priority: List[str] = None
    
    def __post_init__(self):
        if self.expert_categories is None:
            self.expert_categories = [
                "audio_processing", "speech_recognition", "text_generation", 
                "text_understanding", "vision_analysis", "multimodal_fusion",
                "reasoning", "general"
            ]
        if self.modality_priority is None:
            self.modality_priority = ["audio", "text", "vision"]


class DynamicVocabulary:
    """Dynamic vocabulary that grows during training"""
    
    def __init__(self, initial_vocab_size: int = 32000, max_vocab_size: int = 100000):
        self.initial_vocab_size = initial_vocab_size
        self.max_vocab_size = max_vocab_size
        self.current_vocab_size = initial_vocab_size
        self.token_frequencies = {}
        self.growth_threshold = 0.8
        
    def update_frequencies(self, tokens: List[int]):
        """Update token frequencies for vocabulary growth decisions"""
        for token in tokens:
            self.token_frequencies[token] = self.token_frequencies.get(token, 0) + 1
    
    def should_grow_vocab(self) -> bool:
        """Determine if vocabulary should grow based on usage patterns"""
        if self.current_vocab_size >= self.max_vocab_size:
            return False
        
        # Check if we're using most of the current vocabulary
        used_tokens = len(self.token_frequencies)
        usage_ratio = used_tokens / self.current_vocab_size
        return usage_ratio > self.growth_threshold
    
    def grow_vocabulary(self, growth_size: int = 1000):
        """Grow vocabulary by adding new tokens"""
        old_size = self.current_vocab_size
        self.current_vocab_size = min(
            self.current_vocab_size + growth_size, 
            self.max_vocab_size
        )
        return self.current_vocab_size - old_size


class QuantizedLinear(nn.Module):
    """Quantized linear layer for compression"""
    
    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Quantized weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(out_features))
        self.zero_point = nn.Parameter(torch.zeros(out_features))
        
    def quantize_weights(self):
        """Quantize weights to specified bit width"""
        with torch.no_grad():
            # Min-max quantization
            w_min = self.weight.min()
            w_max = self.weight.max()
            
            # Scale and zero point
            scale = (w_max - w_min) / (2**self.bits - 1)
            zero_point = -w_min / scale
            
            # Quantize
            q_weights = torch.round(self.weight / scale + zero_point)
            q_weights = torch.clamp(q_weights, 0, 2**self.bits - 1)
            
            # Dequantize for forward pass
            self.weight.data = (q_weights - zero_point) * scale
            self.scale.data = scale
            self.zero_point.data = zero_point
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.quantize_weights()
        return F.linear(x, self.weight)


class ExpertRouter(nn.Module):
    """Mixture of Experts router with category-based routing"""
    
    def __init__(self, d_model: int, num_experts: int, expert_categories: List[str]):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_categories = expert_categories
        
        # Category embeddings
        self.category_embeddings = nn.Embedding(len(expert_categories), d_model)
        
        # Routing network
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Expert capacity
        self.expert_capacity = 64
        
    def forward(self, x: torch.Tensor, modality: str = "text") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route inputs to experts based on modality and content
        Returns: (expert_weights, expert_indices)
        """
        batch_size, seq_len, _ = x.shape
        
        # Get modality category
        category_idx = self.expert_categories.index(f"{modality}_processing") if f"{modality}_processing" in self.expert_categories else 0
        category_emb = self.category_embeddings(torch.tensor(category_idx, device=x.device))
        
        # Combine input with modality information
        modality_info = category_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        combined_input = x + modality_info
        
        # Get expert weights
        expert_weights = self.router(combined_input)  # [B, T, num_experts]
        
        # Select top-k experts
        top_k = min(2, self.num_experts)  # Use top 2 experts
        expert_indices = torch.topk(expert_weights, top_k, dim=-1).indices  # [B, T, top_k]
        
        return expert_weights, expert_indices


class Mamba3Block(nn.Module):
    """Mamba 3 block with selective state space mechanism"""
    
    def __init__(self, d_model: int, d_state: int, d_conv: int, dropout: float, 
                 use_quantization: bool = False, bits: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Input projection with optional quantization
        if use_quantization:
            self.in_proj = QuantizedLinear(d_model, 2 * d_model, bits)
        else:
            self.in_proj = nn.Linear(d_model, 2 * d_model)
        
        # Convolutional layer for local dependencies
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=d_conv, 
                               groups=d_model, padding=d_conv - 1)
        
        # State space parameters
        self.delta_proj = nn.Linear(d_model, d_state)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Output projection
        if use_quantization:
            self.out_proj = QuantizedLinear(d_model, d_model, bits)
        else:
            self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def selective_scan(self, u: torch.Tensor, delta: torch.Tensor, 
                      B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Selective state space scan mechanism"""
        batch_size, seq_len, d_model = u.shape
        
        # Initialize state
        A = torch.ones(self.d_state, device=u.device) * -1.0  # Decay parameter
        h = torch.zeros(batch_size, self.d_state, device=u.device)
        
        outputs = []
        
        for t in range(seq_len):
            # Update state
            h = h * torch.exp(delta[:, t, :] * A) + u[:, t, :] * B[:, t, :]
            
            # Compute output
            y = torch.sum(h * C[:, t, :], dim=-1, keepdim=True)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)  # [B, T, 1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer norm
        h = self.norm(x)
        
        # Input projection
        u, v = self.in_proj(h).chunk(2, dim=-1)
        
        # Convolutional processing
        u_conv = self.conv1d(u.transpose(1, 2))[:, :, :u.size(1)]
        u_conv = u_conv.transpose(1, 2)
        
        # State space parameters
        delta = F.softplus(self.delta_proj(u_conv))
        B = self.B_proj(u_conv)
        C = self.C_proj(u_conv)
        
        # Selective scan
        y = self.selective_scan(u_conv, delta, B, C)
        y = y.squeeze(-1)  # [B, T]
        
        # Gating
        y = y * torch.sigmoid(v)
        
        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return x + y  # Residual connection


class ModalityEncoder(nn.Module):
    """Base class for modality-specific encoders"""
    
    def __init__(self, input_dim: int, d_model: int, modality: str):
        super().__init__()
        self.modality = modality
        self.input_dim = input_dim
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AudioEncoder(ModalityEncoder):
    """Audio/Speech encoder with priority processing"""
    
    def __init__(self, input_dim: int, d_model: int):
        super().__init__(input_dim, d_model, "audio")
        
        # Convolutional layers for audio features
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.Conv1d(128, d_model, kernel_size=3, padding=1)
        ])
        
        # Attention mechanism for temporal modeling
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # Projection to model dimension
        self.proj = nn.Linear(input_dim, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim] - audio features
        batch_size, seq_len, _ = x.shape
        
        # Convolutional processing
        x_conv = x.transpose(1, 2)  # [B, input_dim, T]
        for conv in self.conv_layers:
            x_conv = F.relu(conv(x_conv))
        x_conv = x_conv.transpose(1, 2)  # [B, T, d_model]
        
        # Self-attention for temporal dependencies
        attn_out, _ = self.attention(x_conv, x_conv, x_conv)
        
        # Residual connection
        output = x_conv + attn_out
        
        return output


class TextEncoder(ModalityEncoder):
    """Text encoder with dynamic vocabulary support"""
    
    def __init__(self, vocab_size: int, d_model: int, dynamic_vocab: DynamicVocabulary):
        super().__init__(vocab_size, d_model, "text")
        self.dynamic_vocab = dynamic_vocab
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1024, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] - token indices
        seq_len = x.size(1)
        
        # Embedding lookup
        embedded = self.embedding(x)  # [B, T, d_model]
        
        # Add positional encoding
        if seq_len <= self.positional_encoding.size(0):
            pos_enc = self.positional_encoding[:seq_len].unsqueeze(0)
            embedded = embedded + pos_enc
        
        return embedded


class VisionEncoder(ModalityEncoder):
    """Vision encoder with patch-based processing"""
    
    def __init__(self, input_dim: int, d_model: int, patch_size: int = 16):
        super().__init__(input_dim, d_model, "vision")
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 196, d_model))  # 14x14 patches
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
            for _ in range(4)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] - image
        batch_size = x.size(0)
        
        # Patch embedding
        patches = self.patch_embed(x)  # [B, d_model, H//patch_size, W//patch_size]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, d_model]
        
        # Add positional encoding
        patches = patches + self.pos_embed
        
        # Transformer processing
        for layer in self.transformer_layers:
            patches = layer(patches)
        
        return patches


class Mamba3MultiModal(nn.Module):
    """Main Mamba 3 Multi-Modal model with MoE"""
    
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        self.dynamic_vocab = DynamicVocabulary(
            config.initial_vocab_size, 
            config.max_vocab_size
        )
        
        # Modality encoders
        self.audio_encoder = AudioEncoder(config.audio_dim, config.d_model)
        self.text_encoder = TextEncoder(config.initial_vocab_size, config.d_model, self.dynamic_vocab)
        self.vision_encoder = VisionEncoder(3, config.d_model)
        
        # Modality fusion
        self.modality_fusion = nn.MultiheadAttention(
            config.d_model, num_heads=8, batch_first=True
        )
        
        # Expert router
        self.expert_router = ExpertRouter(
            config.d_model, 
            config.num_experts, 
            config.expert_categories
        )
        
        # Mamba 3 blocks
        self.mamba_blocks = nn.ModuleList([
            Mamba3Block(
                config.d_model, 
                config.d_state, 
                config.d_conv, 
                config.dropout,
                use_quantization=config.quantization_bits < 16,
                bits=config.quantization_bits
            )
            for _ in range(config.n_layers)
        ])
        
        # Output heads for each modality
        self.audio_head = nn.Linear(config.d_model, config.audio_dim)
        self.text_head = nn.Linear(config.d_model, config.initial_vocab_size)
        self.vision_head = nn.Linear(config.d_model, config.vision_dim)
        
        # Compression components
        self.quantization_enabled = config.quantization_bits < 16
        self.pruning_mask = None
        
    def forward(self, inputs: Dict[str, torch.Tensor], 
                modality_priority: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with modality priority: audio -> text -> vision
        """
        if modality_priority is None:
            modality_priority = self.config.modality_priority
        
        # Encode each modality
        encoded_modalities = {}
        
        if "audio" in inputs and "audio" in modality_priority:
            encoded_modalities["audio"] = self.audio_encoder(inputs["audio"])
        
        if "text" in inputs and "text" in modality_priority:
            encoded_modalities["text"] = self.text_encoder(inputs["text"])
        
        if "vision" in inputs and "vision" in modality_priority:
            encoded_modalities["vision"] = self.vision_encoder(inputs["vision"])
        
        # Fuse modalities in priority order
        fused_features = None
        for modality in modality_priority:
            if modality in encoded_modalities:
                if fused_features is None:
                    fused_features = encoded_modalities[modality]
                else:
                    # Cross-modal attention
                    attn_out, _ = self.modality_fusion(
                        fused_features, 
                        encoded_modalities[modality], 
                        encoded_modalities[modality]
                    )
                    fused_features = fused_features + attn_out
        
        # Process through Mamba 3 blocks with MoE
        for i, block in enumerate(self.mamba_blocks):
            # Get expert routing
            expert_weights, expert_indices = self.expert_router(
                fused_features, 
                modality_priority[0] if modality_priority else "text"
            )
            
            # Process through Mamba block
            fused_features = block(fused_features)
            
            # Apply expert weighting (simplified)
            if i % 2 == 0:  # Apply MoE every other layer
                expert_weights_expanded = expert_weights.unsqueeze(-1).expand_as(fused_features)
                fused_features = fused_features * expert_weights_expanded.mean(dim=-2, keepdim=True)
        
        # Generate outputs for each modality
        outputs = {}
        if "audio" in inputs:
            outputs["audio"] = self.audio_head(fused_features)
        if "text" in inputs:
            outputs["text"] = self.text_head(fused_features)
        if "vision" in inputs:
            outputs["vision"] = self.vision_head(fused_features)
        
        return outputs
    
    def apply_pruning(self, pruning_ratio: float = 0.1):
        """Apply structured pruning to reduce model size"""
        if self.pruning_mask is not None:
            return
        
        # Create pruning mask based on weight magnitudes
        all_weights = []
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                all_weights.append(param.data.abs().flatten())
        
        if not all_weights:
            return
        
        all_weights = torch.cat(all_weights)
        threshold = torch.quantile(all_weights, pruning_ratio)
        
        # Create masks for each layer
        self.pruning_mask = {}
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                mask = (param.data.abs() > threshold).float()
                self.pruning_mask[name] = mask
                param.data *= mask
    
    def compress_model(self):
        """Apply compression techniques"""
        if self.quantization_enabled:
            # Quantize weights
            for module in self.modules():
                if isinstance(module, QuantizedLinear):
                    module.quantize_weights()
        
        # Apply pruning
        self.apply_pruning(self.config.pruning_ratio)
    
    def update_vocabulary(self, new_tokens: List[int]):
        """Update dynamic vocabulary during training"""
        self.dynamic_vocab.update_frequencies(new_tokens)
        
        if self.dynamic_vocab.should_grow_vocab():
            growth_size = self.dynamic_vocab.grow_vocabulary()
            logger.info(f"Vocabulary grown by {growth_size} tokens")
            
            # Expand embedding layer
            old_embedding = self.text_encoder.embedding
            new_vocab_size = self.dynamic_vocab.current_vocab_size
            
            new_embedding = nn.Embedding(new_vocab_size, self.config.d_model)
            new_embedding.weight.data[:old_embedding.weight.size(0)] = old_embedding.weight.data
            
            self.text_encoder.embedding = new_embedding
            self.text_head = nn.Linear(self.config.d_model, new_vocab_size)


def create_mamba3_multimodal(config: Mamba3Config) -> Mamba3MultiModal:
    """Create Mamba 3 Multi-Modal model from config"""
    return Mamba3MultiModal(config)


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = Mamba3Config(
        d_model=768,
        n_layers=12,
        num_experts=8,
        quantization_bits=8,
        pruning_ratio=0.1
    )
    
    # Create model
    model = create_mamba3_multimodal(config)
    
    # Test with sample inputs
    batch_size = 2
    seq_len = 128
    
    inputs = {
        "audio": torch.randn(batch_size, seq_len, config.audio_dim),
        "text": torch.randint(0, config.initial_vocab_size, (batch_size, seq_len)),
        "vision": torch.randn(batch_size, 3, 224, 224)
    }
    
    # Forward pass
    outputs = model(inputs)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Audio output shape: {outputs['audio'].shape}")
    print(f"Text output shape: {outputs['text'].shape}")
    print(f"Vision output shape: {outputs['vision'].shape}")