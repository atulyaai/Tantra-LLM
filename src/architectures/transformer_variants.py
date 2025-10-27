"""
OCR-Native Transformer Variants
Different transformer architectures optimized for OCR processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image

from src.core.ocr_native_llm import OCRNativeConfig, OCRWeightEncoder, OCRMemoryBank
from src.utils.error_handler import logger


@dataclass
class TransformerVariantConfig(OCRNativeConfig):
    """Configuration for transformer variants"""
    # Architecture variants
    variant: str = "standard"  # standard, mamba, hybrid, memory_enhanced
    use_rotary_embeddings: bool = True
    use_relative_attention: bool = False
    use_gated_attention: bool = False
    use_ocr_pattern_attention: bool = True
    
    # Advanced features
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    layer_dropout: float = 0.0
    use_gradient_checkpointing: bool = True
    
    # OCR-specific enhancements
    ocr_attention_heads: int = 4
    ocr_pattern_window: int = 64
    ocr_memory_compression: float = 0.8
    
    # Additional fields for compatibility
    use_cuda: bool = False
    mixed_precision: bool = False
    gradient_checkpointing: bool = True


class OCRRotaryEmbedding(nn.Module):
    """Rotary position embeddings for OCR sequences"""
    
    def __init__(self, dim: int, max_seq_len: int = 16384):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class OCRPatternAttention(nn.Module):
    """OCR-optimized attention mechanism with pattern recognition"""
    
    def __init__(self, config: TransformerVariantConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.ocr_heads = config.ocr_attention_heads
        
        # Standard attention projections
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        # OCR-specific projections
        self.ocr_q_proj = nn.Linear(config.d_model, config.ocr_attention_heads * self.head_dim)
        self.ocr_k_proj = nn.Linear(config.d_model, config.ocr_attention_heads * self.head_dim)
        self.ocr_v_proj = nn.Linear(config.d_model, config.ocr_attention_heads * self.head_dim)
        self.ocr_out_proj = nn.Linear(config.ocr_attention_heads * self.head_dim, config.d_model)
        
        # Pattern recognition
        self.pattern_conv = nn.Conv1d(config.d_model, config.d_model, 
                                    kernel_size=config.ocr_pattern_window, 
                                    padding=config.ocr_pattern_window//2)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        
    def forward(self, x: torch.Tensor, ocr_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Standard attention
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        standard_out = self.out_proj(out)
        
        # OCR pattern attention
        if ocr_context is not None:
            ocr_q = self.ocr_q_proj(ocr_context).view(batch_size, seq_len, self.ocr_attention_heads, self.head_dim).transpose(1, 2)
            ocr_k = self.ocr_k_proj(ocr_context).view(batch_size, seq_len, self.ocr_attention_heads, self.head_dim).transpose(1, 2)
            ocr_v = self.ocr_v_proj(ocr_context).view(batch_size, seq_len, self.ocr_attention_heads, self.head_dim).transpose(1, 2)
            
            # Pattern-based attention
            ocr_scores = torch.matmul(ocr_q, ocr_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            ocr_attn_weights = F.softmax(ocr_scores, dim=-1)
            ocr_attn_weights = self.attn_dropout(ocr_attn_weights)
            
            ocr_out = torch.matmul(ocr_attn_weights, ocr_v)
            ocr_out = ocr_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.ocr_attention_heads * self.head_dim)
            ocr_out = self.ocr_out_proj(ocr_out)
            
            # Combine standard and OCR attention
            return standard_out + ocr_out
        
        return standard_out


class OCRNativeTransformer(nn.Module):
    """Enhanced OCR-Native Transformer with multiple variants"""
    
    def __init__(self, config: TransformerVariantConfig):
        super().__init__()
        self.config = config
        
        # OCR components
        self.weight_encoder = OCRWeightEncoder(config)
        self.memory_bank = OCRMemoryBank(config)
        
        # Embeddings
        self.ocr_embedding = nn.Linear(config.ocr_image_width * config.ocr_image_height, config.d_model)
        
        # Position embeddings
        if config.use_rotary_embeddings:
            self.rotary_emb = OCRRotaryEmbedding(config.d_model, config.max_seq_length)
        else:
            self.pos_embedding = nn.Embedding(config.max_seq_length, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            self._create_transformer_block() for _ in range(config.n_layers)
        ])
        
        # Output heads
        self.text_head = nn.Linear(config.d_model, config.vocab_size)
        self.ocr_head = nn.Linear(config.d_model, config.ocr_image_width * config.ocr_image_height)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized OCR-Native Transformer variant: {config.variant}")
    
    def _create_transformer_block(self):
        """Create transformer block based on variant"""
        if self.config.variant == "mamba":
            return OCRMambaBlock(self.config)
        elif self.config.variant == "hybrid":
            return OCRHybridBlock(self.config)
        elif self.config.variant == "memory_enhanced":
            return OCRMemoryEnhancedBlock(self.config)
        else:
            return OCRStandardBlock(self.config)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass with variant-specific processing"""
        # Process inputs to OCR format
        ocr_inputs = self._process_inputs_to_ocr(inputs)
        
        # Convert to embeddings
        embeddings = self._ocr_to_embeddings(ocr_inputs)
        
        # Add position embeddings
        seq_len = embeddings.size(1)
        if self.config.use_rotary_embeddings:
            cos, sin = self.rotary_emb(embeddings, seq_len)
            # Apply rotary embeddings (simplified)
            embeddings = embeddings * cos + embeddings * sin
        else:
            pos_ids = torch.arange(seq_len, device=embeddings.device)
            pos_emb = self.pos_embedding(pos_ids)
            embeddings = embeddings + pos_emb
        
        # Process through transformer blocks
        x = embeddings
        for block in self.blocks:
            x = block(x)
        
        # Generate outputs
        text_logits = self.text_head(x)
        ocr_output = self.ocr_head(x)
        
        return {
            'text_logits': text_logits,
            'ocr_output': ocr_output,
            'embeddings': x
        }
    
    def _process_inputs_to_ocr(self, inputs: Dict[str, Any]) -> List[Image.Image]:
        """Convert inputs to OCR format"""
        from src.core.ocr_native_llm import OCRInputProcessor
        processor = OCRInputProcessor(self.config)
        
        ocr_inputs = []
        if 'text' in inputs:
            ocr_text = processor.process_text_to_ocr(inputs['text'])
            ocr_inputs.append(ocr_text)
        if 'speech' in inputs:
            ocr_speech = processor.process_speech_to_ocr(inputs['speech'])
            ocr_inputs.append(ocr_speech)
        if 'image' in inputs and inputs['image'] is not None:
            ocr_image = processor.process_image_to_ocr(inputs['image'])
            ocr_inputs.append(ocr_image)
        
        return ocr_inputs
    
    def _ocr_to_embeddings(self, ocr_images: List[Image.Image]) -> torch.Tensor:
        """Convert OCR images to embeddings"""
        if not ocr_images:
            return torch.zeros(1, 1, self.config.d_model)
        
        img = ocr_images[0]
        img = img.convert('L').resize((self.config.ocr_image_width, self.config.ocr_image_height))
        img_array = np.array(img).flatten()
        
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        expected_size = self.config.ocr_image_width * self.config.ocr_image_height
        
        if len(img_tensor) != expected_size:
            if len(img_tensor) < expected_size:
                img_tensor = F.pad(img_tensor, (0, expected_size - len(img_tensor)))
            else:
                img_tensor = img_tensor[:expected_size]
        
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        return self.ocr_embedding(img_tensor)


class OCRStandardBlock(nn.Module):
    """Standard transformer block with OCR optimizations"""
    
    def __init__(self, config: TransformerVariantConfig):
        super().__init__()
        self.attention = OCRPatternAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.ffn_dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.attention_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out = self.attention(x)
        x = self.ln1(x + self.dropout(attn_out))
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        return x


class OCRMambaBlock(nn.Module):
    """Mamba-inspired block for OCR processing"""
    
    def __init__(self, config: TransformerVariantConfig):
        super().__init__()
        self.config = config
        
        # Mamba-style selective mechanism
        self.selective_proj = nn.Linear(config.d_model, config.d_model * 2)
        self.conv1d = nn.Conv1d(config.d_model, config.d_model, kernel_size=3, padding=1)
        
        # Standard components
        self.attention = OCRPatternAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Selective mechanism (simplified Mamba)
        selective = self.selective_proj(x)
        gate, value = selective.chunk(2, dim=-1)
        selective_out = gate * value
        
        # Conv1D processing
        conv_out = self.conv1d(selective_out.transpose(1, 2)).transpose(1, 2)
        
        # Attention
        attn_out = self.attention(conv_out)
        x = self.ln1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        return x


class OCRHybridBlock(nn.Module):
    """Hybrid block combining multiple mechanisms"""
    
    def __init__(self, config: TransformerVariantConfig):
        super().__init__()
        self.attention = OCRPatternAttention(config)
        self.mamba_component = OCRMambaBlock(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mix_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mix attention and mamba
        attn_out = self.attention(x)
        mamba_out = self.mamba_component(x)
        
        mixed = self.mix_weights[0] * attn_out + self.mix_weights[1] * mamba_out
        x = self.ln1(x + mixed)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        return x


class OCRMemoryEnhancedBlock(nn.Module):
    """Memory-enhanced block with OCR pattern memory"""
    
    def __init__(self, config: TransformerVariantConfig):
        super().__init__()
        self.attention = OCRPatternAttention(config)
        self.memory_attention = nn.MultiheadAttention(config.d_model, config.n_heads, 
                                                    dropout=config.attention_dropout)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ln3 = nn.LayerNorm(config.d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out = self.attention(x)
        x = self.ln1(x + attn_out)
        
        # Memory attention (simplified - would use actual memory in practice)
        memory_out, _ = self.memory_attention(x, x, x)
        x = self.ln2(x + memory_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ln3(x + ffn_out)
        
        return x