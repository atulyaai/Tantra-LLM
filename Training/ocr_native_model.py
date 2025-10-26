"""
OCR-Native Neural Architecture
A novel model designed specifically for OCR-based memory and weight storage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from PIL import Image
import math
from dataclasses import dataclass
import logging
from .ocr_memory import OCRMemoryBank, OCRContextualMemory, OCRParameterEfficientMemory, OCRMemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class OCRNativeConfig:
    """Configuration for OCR-native model"""
    # Core architecture
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    
    # OCR-specific settings
    image_size: int = 224
    patch_size: int = 16
    num_patches: int = 196  # (224/16)^2
    
    # Memory settings
    memory_capacity: int = 1000
    memory_dim: int = 256
    ocr_precision: int = 6
    
    # Visual processing
    visual_layers: int = 4
    text_layers: int = 4
    fusion_layers: int = 2
    
    # Multi-modal
    audio_dim: int = 128
    text_dim: int = 512
    vision_dim: int = 512


class OCRPatchEmbedding(nn.Module):
    """Patch embedding specifically designed for OCR images"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # OCR-optimized patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # OCR-specific positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # OCR character recognition enhancement
        self.ocr_enhance = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B = x.shape[0]
        
        # Patch embedding
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, n_patches, embed_dim]
        
        # Add OCR enhancement
        x = self.ocr_enhance(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        return x


class OCRAttention(nn.Module):
    """Attention mechanism optimized for OCR pattern recognition"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # OCR-specific attention bias
        self.ocr_bias = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.w_q(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add OCR bias for character recognition
        scores = scores + self.ocr_bias
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, seq_len, d_model)
        
        return self.w_o(context)


class OCRMemoryLayer(nn.Module):
    """Layer that processes OCR-based memory"""
    
    def __init__(self, d_model: int, memory_dim: int, ocr_config: OCRMemoryConfig):
        super().__init__()
        self.d_model = d_model
        self.memory_dim = memory_dim
        
        # OCR memory components
        self.ocr_memory_bank = OCRMemoryBank(ocr_config)
        self.contextual_memory = OCRContextualMemory(ocr_config, d_model)
        self.param_efficient_memory = OCRParameterEfficientMemory(ocr_config)
        
        # Memory processing layers
        self.memory_encoder = nn.Linear(d_model, memory_dim)
        self.memory_decoder = nn.Linear(memory_dim, d_model)
        self.memory_attention = OCRAttention(memory_dim, n_heads=8)
        
        # Memory fusion
        self.memory_fusion = nn.Sequential(
            nn.Linear(d_model + memory_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, x: torch.Tensor, memory_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encode input for memory processing
        memory_encoded = self.memory_encoder(x)
        
        # Retrieve relevant memories
        if memory_context is not None:
            # Contextual memory retrieval
            retrieved_memories = self.contextual_memory.retrieve_contextual_memory(memory_context)
            
            if retrieved_memories:
                # Process retrieved memories
                memory_tensors = [mem[0] for mem in retrieved_memories]
                if memory_tensors:
                    # Stack and process memories
                    memory_stack = torch.stack(memory_tensors, dim=1)  # [B, num_memories, memory_dim]
                    
                    # Apply memory attention
                    memory_processed = self.memory_attention(memory_stack)
                    
                    # Average over memories
                    memory_avg = memory_processed.mean(dim=1)  # [B, memory_dim]
                    
                    # Decode back to model dimension
                    memory_decoded = self.memory_decoder(memory_avg)
                    
                    # Fuse with input
                    x = self.memory_fusion(torch.cat([x, memory_decoded], dim=-1))
        
        return x
    
    def store_memory(self, x: torch.Tensor, memory_context: torch.Tensor, 
                    memory_type: str = "general") -> str:
        """Store current state as OCR memory"""
        # Encode for storage
        memory_encoded = self.memory_encoder(x)
        
        # Store in OCR format
        memory_id = self.contextual_memory.store_contextual_memory(
            memory_context, memory_encoded, memory_type
        )
        
        return memory_id


class OCRTransformerBlock(nn.Module):
    """Transformer block optimized for OCR processing"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, 
                 ocr_config: OCRMemoryConfig = None):
        super().__init__()
        self.d_model = d_model
        
        # OCR-optimized attention
        self.attention = OCRAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # OCR memory layer
        if ocr_config:
            self.ocr_memory = OCRMemoryLayer(d_model, d_model // 2, ocr_config)
        else:
            self.ocr_memory = None
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, memory_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # OCR memory processing
        if self.ocr_memory:
            x = self.ocr_memory(x, memory_context)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class OCRNativeModel(nn.Module):
    """Main OCR-native model architecture"""
    
    def __init__(self, config: OCRNativeConfig):
        super().__init__()
        self.config = config
        
        # OCR memory configuration
        ocr_config = OCRMemoryConfig(
            image_width=config.image_size,
            image_height=config.image_size,
            precision_digits=config.ocr_precision
        )
        
        # Patch embedding for OCR images
        self.patch_embed = OCRPatchEmbedding(
            config.image_size, config.patch_size, 3, config.d_model
        )
        
        # Multi-modal encoders
        self.audio_encoder = nn.Sequential(
            nn.Linear(config.audio_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Embedding(32000, config.d_model),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # OCR transformer blocks
        self.ocr_blocks = nn.ModuleList([
            OCRTransformerBlock(
                config.d_model, config.n_heads, config.d_ff, 
                config.dropout, ocr_config
            )
            for _ in range(config.n_layers)
        ])
        
        # Output heads
        self.audio_head = nn.Linear(config.d_model, config.audio_dim)
        self.text_head = nn.Linear(config.d_model, 32000)
        self.vision_head = nn.Linear(config.d_model, config.vision_dim)
        
        # Memory management
        self.memory_context = None
        self.stored_memories = []
    
    def forward(self, inputs: Dict[str, torch.Tensor], 
                use_memory: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass with OCR memory processing"""
        
        # Encode different modalities
        encoded_modalities = []
        
        if "audio" in inputs:
            audio_encoded = self.audio_encoder(inputs["audio"])
            encoded_modalities.append(audio_encoded)
        
        if "text" in inputs:
            text_encoded = self.text_encoder(inputs["text"])
            encoded_modalities.append(text_encoded)
        
        if "vision" in inputs:
            # Process vision as OCR images
            vision_patches = self.patch_embed(inputs["vision"])
            encoded_modalities.append(vision_patches)
        
        # Fuse modalities
        if len(encoded_modalities) > 1:
            # Simple concatenation for now - could be improved
            x = torch.cat(encoded_modalities, dim=1)
        else:
            x = encoded_modalities[0]
        
        # Process through OCR transformer blocks
        for block in self.ocr_blocks:
            x = block(x, self.memory_context if use_memory else None)
        
        # Generate outputs
        outputs = {}
        if "audio" in inputs:
            outputs["audio"] = self.audio_head(x)
        if "text" in inputs:
            outputs["text"] = self.text_head(x)
        if "vision" in inputs:
            outputs["vision"] = self.vision_head(x)
        
        return outputs
    
    def store_memory(self, inputs: Dict[str, torch.Tensor], memory_type: str = "general"):
        """Store current state as OCR memory"""
        # Create memory context from inputs
        context_parts = []
        if "audio" in inputs:
            context_parts.append(inputs["audio"].mean(dim=1))  # Average over sequence
        if "text" in inputs:
            context_parts.append(inputs["text"].float().mean(dim=1))  # Average embeddings
        if "vision" in inputs:
            context_parts.append(inputs["vision"].mean(dim=(2, 3)).flatten(1))  # Average spatial
        
        if context_parts:
            memory_context = torch.cat(context_parts, dim=1)
            
            # Store in OCR memory
            for block in self.ocr_blocks:
                if hasattr(block, 'ocr_memory') and block.ocr_memory:
                    memory_id = block.ocr_memory.store_memory(
                        block.attention.w_o.weight, memory_context, memory_type
                    )
                    self.stored_memories.append(memory_id)
    
    def retrieve_memory(self, query_inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Retrieve relevant memories based on query"""
        # Create query context
        context_parts = []
        if "audio" in query_inputs:
            context_parts.append(query_inputs["audio"].mean(dim=1))
        if "text" in query_inputs:
            context_parts.append(query_inputs["text"].float().mean(dim=1))
        if "vision" in query_inputs:
            context_parts.append(query_inputs["vision"].mean(dim=(2, 3)).flatten(1))
        
        if context_parts:
            query_context = torch.cat(context_parts, dim=1)
            
            # Retrieve memories from all blocks
            retrieved_memories = []
            for block in self.ocr_blocks:
                if hasattr(block, 'ocr_memory') and block.ocr_memory:
                    memories = block.ocr_memory.contextual_memory.retrieve_contextual_memory(
                        query_context
                    )
                    retrieved_memories.extend([mem[0] for mem in memories])
            
            return retrieved_memories
        
        return []
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        total_memories = 0
        memory_types = {}
        
        for block in self.ocr_blocks:
            if hasattr(block, 'ocr_memory') and block.ocr_memory:
                bank_stats = block.ocr_memory.memory_bank.list_memories()
                total_memories += len(bank_stats)
                
                for memory in bank_stats:
                    mem_type = memory["metadata"].get("context_type", "unknown")
                    memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
        
        return {
            "total_memories": total_memories,
            "memory_types": memory_types,
            "stored_memory_ids": len(self.stored_memories)
        }


def create_ocr_native_model(config: OCRNativeConfig) -> OCRNativeModel:
    """Create OCR-native model from configuration"""
    return OCRNativeModel(config)


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = OCRNativeConfig(
        d_model=512,
        n_layers=6,
        image_size=224,
        memory_capacity=100
    )
    
    # Create model
    model = create_ocr_native_model(config)
    
    # Test with sample inputs
    batch_size = 2
    seq_len = 128
    
    inputs = {
        "audio": torch.randn(batch_size, seq_len, config.audio_dim),
        "text": torch.randint(0, 32000, (batch_size, seq_len)),
        "vision": torch.randn(batch_size, 3, config.image_size, config.image_size)
    }
    
    # Forward pass
    outputs = model(inputs)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Audio output shape: {outputs['audio'].shape}")
    print(f"Text output shape: {outputs['text'].shape}")
    print(f"Vision output shape: {outputs['vision'].shape}")
    
    # Test memory storage
    model.store_memory(inputs, "test_memory")
    
    # Test memory retrieval
    retrieved = model.retrieve_memory(inputs)
    print(f"Retrieved {len(retrieved)} memories")
    
    # Get memory statistics
    stats = model.get_memory_statistics()
    print(f"Memory statistics: {stats}")