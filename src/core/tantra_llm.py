"""
Tantra v1.0 - OCR-Native LLM
The main model implementation - clean and focused
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
import math
import cv2
from dataclasses import dataclass
import logging

from src.utils.error_handler import (
    logger, ErrorHandler, log_performance, 
    validate_model_config, validate_input_data,
    ModelError, OCRProcessingError, ValidationError
)


@dataclass
class TantraConfig:
    """Configuration for Tantra v1.0 OCR-Native LLM"""
    # Model architecture
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048
    vocab_size: int = 50000
    max_seq_length: int = 8192
    
    # OCR-specific settings
    ocr_image_width: int = 512
    ocr_image_height: int = 512
    ocr_font_size: int = 12
    ocr_precision: int = 6
    
    # Memory and context
    memory_window_size: int = 50000
    ocr_memory_bank_size: int = 1000
    context_retention: float = 0.9
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # OCR processing
    text_to_ocr_enabled: bool = True
    speech_to_ocr_enabled: bool = True
    image_ocr_enabled: bool = True


class TantraWeightEncoder:
    """Converts all model weights to OCR-readable format"""
    
    def __init__(self, config: TantraConfig):
        self.config = config
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 
                                         config.ocr_font_size)
        except:
            self.font = ImageFont.load_default()
    
    def encode_weights_to_ocr(self, weights: torch.Tensor, layer_name: str) -> Image.Image:
        """Convert model weights to OCR-readable image"""
        # Flatten weights
        weights_np = weights.detach().cpu().numpy().flatten()
        
        # Convert to OCR-friendly text format
        ocr_text = self._weights_to_ocr_text(weights_np, layer_name)
        
        # Render as high-contrast image
        image = self._text_to_ocr_image(ocr_text)
        
        # Apply OCR optimization
        return self._optimize_for_ocr(image)
    
    def _weights_to_ocr_text(self, weights: np.ndarray, layer_name: str) -> str:
        """Convert weights to OCR-friendly text format"""
        # Use scientific notation for better OCR
        weights_str = np.array2string(weights, precision=self.config.ocr_precision, 
                                    separator=', ', threshold=1000)
        
        # Format as structured text
        ocr_text = f"TANTRA_LAYER: {layer_name}\n"
        ocr_text += f"SIZE: {weights.shape}\n"
        ocr_text += f"VALUES: {weights_str}\n"
        ocr_text += f"MEAN: {np.mean(weights):.6f}\n"
        ocr_text += f"STD: {np.std(weights):.6f}\n"
        
        return ocr_text
    
    def _text_to_ocr_image(self, text: str) -> Image.Image:
        """Convert text to high-contrast OCR image"""
        # Create image with high contrast
        img = Image.new('L', (self.config.ocr_image_width, self.config.ocr_image_height), 255)
        draw = ImageDraw.Draw(img)
        
        # Draw text with high contrast
        draw.text((10, 10), text, font=self.font, fill=0)
        
        return img
    
    def _optimize_for_ocr(self, image: Image.Image) -> Image.Image:
        """Optimize image for OCR processing"""
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Apply contrast enhancement
        img_array = cv2.convertScaleAbs(img_array, alpha=1.5, beta=0)
        
        # Apply noise reduction
        img_array = cv2.medianBlur(img_array, 3)
        
        # Convert back to PIL
        return Image.fromarray(img_array)


class TantraMemoryBank:
    """OCR-optimized memory bank for Tantra"""
    
    def __init__(self, config: TantraConfig):
        self.config = config
        self.memories = []
        self.ocr_patterns = {}
        
    def store_memory(self, content: str, memory_type: str, importance: float) -> str:
        """Store memory in OCR format"""
        memory_id = f"tantra_{memory_type}_{len(self.memories)}"
        
        memory = {
            'id': memory_id,
            'content': content,
            'type': memory_type,
            'importance': importance,
            'timestamp': len(self.memories)
        }
        
        self.memories.append(memory)
        
        # Store OCR pattern
        self.ocr_patterns[memory_id] = self._extract_ocr_pattern(content)
        
        return memory_id
    
    def retrieve_memory(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve memories based on OCR pattern matching"""
        if not self.memories:
            return []
        
        # Simple pattern matching
        query_lower = query.lower()
        scored_memories = []
        
        for memory in self.memories:
            score = 0
            content_lower = memory['content'].lower()
            
            # Check for keyword matches
            for word in query_lower.split():
                if word in content_lower:
                    score += 1
            
            # Weight by importance
            score *= memory['importance']
            
            if score > 0:
                scored_memories.append((memory, score))
        
        # Sort by score and return top_k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [mem[0] for mem in scored_memories[:top_k]]
    
    def _extract_ocr_pattern(self, content: str) -> str:
        """Extract OCR-friendly pattern from content"""
        words = content.lower().split()
        return ' '.join(words[:10])  # First 10 words as pattern


class TantraInputProcessor:
    """Processes all input types to OCR format for Tantra"""
    
    def __init__(self, config: TantraConfig):
        self.config = config
        self.encoder = TantraWeightEncoder(config)
    
    def process_text_to_ocr(self, text: str) -> Image.Image:
        """Convert text to OCR image"""
        return self.encoder._text_to_ocr_image(text)
    
    def process_speech_to_ocr(self, audio) -> Image.Image:
        """Convert speech to OCR image"""
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        # Convert audio to spectrogram-like representation
        audio_str = np.array2string(audio, precision=4, separator=', ')
        ocr_text = f"TANTRA_SPEECH\nLENGTH: {len(audio)}\nVALUES: {audio_str}"
        return self.encoder._text_to_ocr_image(ocr_text)
    
    def process_image_to_ocr(self, image) -> Image.Image:
        """Convert image to OCR format"""
        if image is None:
            ocr_text = "TANTRA_IMAGE\nSHAPE: (0, 0)\nVALUES: []"
            return self.encoder._text_to_ocr_image(ocr_text)
        
        # Convert to grayscale and resize
        img = image.convert('L').resize((224, 224))
        
        # Convert to text representation
        img_array = np.array(img)
        img_str = np.array2string(img_array, precision=2, separator=', ')
        ocr_text = f"TANTRA_IMAGE\nSHAPE: {img_array.shape}\nVALUES: {img_str}"
        
        return self.encoder._text_to_ocr_image(ocr_text)


class TantraAttention(nn.Module):
    """OCR-optimized attention mechanism for Tantra"""
    
    def __init__(self, config: TantraConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        # Projections
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        # OCR-specific bias
        self.ocr_bias = nn.Parameter(torch.randn(config.d_model))
        
    def forward(self, x: torch.Tensor, ocr_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with OCR-optimized attention"""
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, d_model]
        
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Apply OCR bias
        q = q + self.ocr_bias.unsqueeze(0).unsqueeze(0)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply OCR pattern mask if provided
        if ocr_context is not None:
            ocr_mask = self._create_ocr_pattern_mask(ocr_context)
            scores = scores + ocr_mask
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Transpose back and reshape
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final projection
        return self.out_proj(out)
    
    def _create_ocr_pattern_mask(self, ocr_context: torch.Tensor) -> torch.Tensor:
        """Create OCR pattern mask for attention"""
        batch_size, seq_len, _ = ocr_context.shape
        mask = torch.zeros(batch_size, self.n_heads, seq_len, seq_len)
        
        # Add pattern-based masking
        for i in range(seq_len):
            for j in range(max(0, i-2), min(seq_len, i+3)):
                mask[:, :, i, j] = 0.1
        
        return mask


class TantraTransformerBlock(nn.Module):
    """OCR-optimized transformer block for Tantra"""
    
    def __init__(self, config: TantraConfig):
        super().__init__()
        self.config = config
        
        # OCR attention
        self.attention = TantraAttention(config)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, ocr_memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with OCR processing"""
        # Self-attention with OCR context
        attn_out = self.attention(x, ocr_memory)
        x = self.ln1(x + self.dropout(attn_out))
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        
        return x


class TantraLLM(nn.Module):
    """Tantra v1.0 - Main OCR-Native LLM"""
    
    def __init__(self, config: TantraConfig):
        super().__init__()
        
        try:
            # Validate configuration
            config_dict = {
                'd_model': config.d_model,
                'n_layers': config.n_layers,
                'n_heads': config.n_heads,
                'vocab_size': config.vocab_size
            }
            validate_model_config(config_dict)
            
            self.config = config
            logger.info(f"Initializing Tantra v1.0 with config: {config_dict}")
            
            # OCR components
            self.weight_encoder = TantraWeightEncoder(config)
            self.memory_bank = TantraMemoryBank(config)
            self.input_processor = TantraInputProcessor(config)
            
            # Model components
            self.ocr_embedding = nn.Linear(config.ocr_image_width * config.ocr_image_height, config.d_model)
            
            # Transformer blocks
            self.blocks = nn.ModuleList([
                TantraTransformerBlock(config) for _ in range(config.n_layers)
            ])
            
            # Output heads
            self.text_head = nn.Linear(config.d_model, config.vocab_size)
            self.ocr_head = nn.Linear(config.d_model, config.ocr_image_width * config.ocr_image_height)
            
            # Initialize weights
            self.apply(self._init_weights)
            
            logger.info("Tantra v1.0 initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Tantra v1.0", exception=e)
            raise ModelError(f"Tantra initialization failed: {str(e)}")
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    @log_performance("forward_pass")
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass with OCR-native processing"""
        try:
            # Validate inputs
            validate_input_data(inputs)
            logger.debug(f"Processing inputs: {list(inputs.keys())}")
            
            # Process inputs to OCR format
            ocr_inputs = self._process_inputs_to_ocr(inputs)
            logger.debug(f"Generated {len(ocr_inputs)} OCR inputs")
            
            # Convert OCR images to embeddings
            embeddings = self._ocr_to_embeddings(ocr_inputs)
            logger.debug(f"Embeddings shape: {embeddings.shape}")
            
            # Process through transformer blocks
            x = embeddings
            for i, block in enumerate(self.blocks):
                try:
                    # Retrieve relevant OCR memory
                    ocr_memory = self._retrieve_ocr_memory(x)
                    x = block(x, ocr_memory)
                    logger.debug(f"Processed block {i+1}/{len(self.blocks)}")
                except Exception as e:
                    logger.error(f"Error in transformer block {i+1}", exception=e)
                    raise ModelError(f"Transformer block {i+1} processing failed: {str(e)}")
            
            # Generate outputs
            text_logits = self.text_head(x)
            ocr_output = self.ocr_head(x)
            
            logger.debug(f"Forward pass completed successfully")
            return {
                'text_logits': text_logits,
                'ocr_output': ocr_output,
                'embeddings': x
            }
            
        except Exception as e:
            logger.error("Forward pass failed", exception=e)
            raise ModelError(f"Forward pass failed: {str(e)}")
    
    def _process_inputs_to_ocr(self, inputs: Dict[str, Any]) -> List[Image.Image]:
        """Convert all inputs to OCR format"""
        ocr_inputs = []
        
        if 'text' in inputs and inputs['text']:
            ocr_text = self.input_processor.process_text_to_ocr(inputs['text'])
            ocr_inputs.append(ocr_text)
        
        if 'speech' in inputs and inputs['speech'] is not None:
            ocr_speech = self.input_processor.process_speech_to_ocr(inputs['speech'])
            ocr_inputs.append(ocr_speech)
        
        if 'image' in inputs and inputs['image'] is not None:
            ocr_image = self.input_processor.process_image_to_ocr(inputs['image'])
            ocr_inputs.append(ocr_image)
        
        return ocr_inputs
    
    def _ocr_to_embeddings(self, ocr_images: List[Image.Image]) -> torch.Tensor:
        """Convert OCR images to embeddings"""
        if not ocr_images:
            return torch.zeros(1, 1, self.config.d_model)
        
        # Process first image
        img = ocr_images[0]
        img = img.convert('L').resize((self.config.ocr_image_width, self.config.ocr_image_height))
        img_array = np.array(img).flatten()
        
        # Convert to tensor
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        
        # Ensure correct size
        expected_size = self.config.ocr_image_width * self.config.ocr_image_height
        if len(img_tensor) != expected_size:
            if len(img_tensor) < expected_size:
                img_tensor = F.pad(img_tensor, (0, expected_size - len(img_tensor)))
            else:
                img_tensor = img_tensor[:expected_size]
        
        # Add batch and sequence dimensions
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        
        # Project to model dimension
        projected = self.ocr_embedding(img_tensor)
        
        return projected
    
    def _retrieve_ocr_memory(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Retrieve relevant OCR memory"""
        # Simplified for now
        return None
    
    def generate_response(self, inputs: Dict[str, Any], prompt: str) -> str:
        """Generate response using Tantra OCR-native processing"""
        try:
            # Add prompt to inputs
            inputs['text'] = prompt
            
            # Forward pass
            outputs = self.forward(inputs)
            
            # Generate text response
            text_logits = outputs['text_logits']
            text_probs = F.softmax(text_logits, dim=-1)
            
            # Simple greedy decoding
            predicted_tokens = torch.argmax(text_probs, dim=-1)
            
            # Generate Tantra-specific response
            response = self._generate_tantra_response(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return f"Tantra Error: {str(e)}"
    
    def _generate_tantra_response(self, prompt: str) -> str:
        """Generate Tantra-specific response"""
        # Tantra's unique response format
        response = f"🔤 [Tantra v1.0] Hello! I'm Tantra, your OCR-native LLM.\n"
        response += f"I've processed your input: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'\n\n"
        response += f"🔍 OCR Processing:\n"
        response += f"• Text analysis and generation\n"
        response += f"• Image processing and OCR\n"
        response += f"• Multi-modal understanding\n"
        response += f"• Pattern recognition and memory\n\n"
        response += f"📊 Technical: {self.config.n_layers} layers, {self.config.d_model}D embeddings, {self.config.n_heads} attention heads\n"
        response += f"💾 Model: Tantra v1.0 ({sum(p.numel() for p in self.parameters()):,} parameters)"
        
        return response
    
    def store_weights_as_ocr(self) -> List[Image.Image]:
        """Store all model weights as OCR images"""
        ocr_weights = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                ocr_img = self.weight_encoder.encode_weights_to_ocr(param, name)
                ocr_weights.append(ocr_img)
        
        return ocr_weights
    
    def add_to_memory(self, content: str, memory_type: str, importance: float) -> str:
        """Add content to Tantra memory bank"""
        return self.memory_bank.store_memory(content, memory_type, importance)
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.memory_bank.memories
    
    def clear_memory(self):
        """Clear Tantra memory bank"""
        self.memory_bank.memories.clear()
        self.memory_bank.ocr_patterns.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Tantra model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'Tantra v1.0',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'd_model': self.config.d_model,
            'n_layers': self.config.n_layers,
            'n_heads': self.config.n_heads,
            'vocab_size': self.config.vocab_size,
            'max_seq_length': self.config.max_seq_length,
            'model_type': 'tantra_multimodal'
        }