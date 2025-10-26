"""
OCR-Based Memory System for Mamba3 Multi-Modal Architecture
Stores model weights and memory as OCR-readable text/images for enhanced pattern recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
import math
from dataclasses import dataclass
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import logging

logger = logging.getLogger(__name__)


@dataclass
class OCRMemoryConfig:
    """Configuration for OCR-based memory system"""
    # OCR settings
    image_width: int = 512
    image_height: int = 512
    font_size: int = 12
    text_color: str = "black"
    background_color: str = "white"
    
    # Memory encoding
    precision_digits: int = 6
    max_values_per_image: int = 1000
    encoding_scheme: str = "scientific"  # scientific, decimal, hex, binary
    
    # Compression
    compression_ratio: float = 0.8
    use_visual_compression: bool = True
    text_compression: bool = True
    
    # Retrieval
    similarity_threshold: float = 0.85
    max_memory_images: int = 100
    cache_size: int = 50


class OCREncoder:
    """Encodes numerical weights/memory into OCR-readable text/images"""
    
    def __init__(self, config: OCRMemoryConfig):
        self.config = config
        self.processor = None
        self.ocr_model = None
        self._load_ocr_model()
    
    def _load_ocr_model(self):
        """Load TrOCR model for OCR processing"""
        try:
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            logger.info("TrOCR model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load TrOCR model: {e}. Using fallback OCR.")
            self.processor = None
            self.ocr_model = None
    
    def encode_weights_to_text(self, weights: torch.Tensor, layer_name: str = "") -> str:
        """Convert weight tensor to OCR-readable text format"""
        weights_np = weights.detach().cpu().numpy()
        
        # Flatten and format weights
        flat_weights = weights_np.flatten()
        
        # Apply encoding scheme
        if self.config.encoding_scheme == "scientific":
            text_values = [f"{w:.{self.config.precision_digits}e}" for w in flat_weights]
        elif self.config.encoding_scheme == "decimal":
            text_values = [f"{w:.{self.config.precision_digits}f}" for w in flat_weights]
        elif self.config.encoding_scheme == "hex":
            text_values = [hex(int(w * 1000000)) for w in flat_weights]  # Scale for hex
        else:  # binary
            text_values = [bin(int(w * 1000000)) for w in flat_weights]
        
        # Create structured text
        header = f"LAYER:{layer_name}\nSHAPE:{list(weights.shape)}\nVALUES:\n"
        values_text = " ".join(text_values)
        
        return header + values_text
    
    def create_memory_image(self, text: str, image_id: str = "") -> Image.Image:
        """Create OCR-readable image from text"""
        img = Image.new('RGB', (self.config.image_width, self.config.image_height), 
                       self.config.background_color)
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to load a font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 
                                    self.config.font_size)
        except:
            font = ImageFont.load_default()
        
        # Split text into lines that fit the image
        lines = self._wrap_text(text, font, self.config.image_width - 20)
        
        # Draw text
        y_offset = 10
        for line in lines:
            if y_offset + self.config.font_size > self.config.image_height:
                break
            draw.text((10, y_offset), line, fill=self.config.text_color, font=font)
            y_offset += self.config.font_size + 2
        
        # Add image ID watermark
        if image_id:
            draw.text((self.config.image_width - 100, self.config.image_height - 20), 
                     f"ID:{image_id}", fill="gray", font=font)
        
        return img
    
    def _wrap_text(self, text: str, font, max_width: int) -> List[str]:
        """Wrap text to fit within image width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = font.getbbox(test_line)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines
    
    def encode_weights_to_image(self, weights: torch.Tensor, layer_name: str = "", 
                              image_id: str = "") -> Image.Image:
        """Convert weight tensor to OCR-readable image"""
        text = self.encode_weights_to_text(weights, layer_name)
        return self.create_memory_image(text, image_id)
    
    def compress_memory_image(self, image: Image.Image) -> Image.Image:
        """Apply visual compression to memory image"""
        if not self.config.use_visual_compression:
            return image
        
        # Resize with compression
        new_width = int(image.width * self.config.compression_ratio)
        new_height = int(image.height * self.config.compression_ratio)
        
        compressed = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to grayscale for better OCR
        if compressed.mode != 'L':
            compressed = compressed.convert('L')
        
        return compressed


class OCRDecoder:
    """Decodes OCR text/images back to numerical weights/memory"""
    
    def __init__(self, config: OCRMemoryConfig):
        self.config = config
        self.processor = None
        self.ocr_model = None
        self._load_ocr_model()
    
    def _load_ocr_model(self):
        """Load TrOCR model for OCR processing"""
        try:
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            logger.info("TrOCR model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load TrOCR model: {e}. Using fallback OCR.")
            self.processor = None
            self.ocr_model = None
    
    def decode_image_to_text(self, image: Image.Image) -> str:
        """Extract text from OCR image using TrOCR or fallback"""
        if self.processor and self.ocr_model:
            return self._decode_with_trocr(image)
        else:
            return self._decode_with_fallback(image)
    
    def _decode_with_trocr(self, image: Image.Image) -> str:
        """Decode using TrOCR model"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process with TrOCR
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            generated_ids = self.ocr_model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text
        except Exception as e:
            logger.warning(f"TrOCR decoding failed: {e}. Using fallback.")
            return self._decode_with_fallback(image)
    
    def _decode_with_fallback(self, image: Image.Image) -> str:
        """Fallback OCR using OpenCV and Tesseract"""
        try:
            import pytesseract
            
            # Convert to OpenCV format
            img_array = np.array(image)
            
            # Preprocess for better OCR
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text
            text = pytesseract.image_to_string(thresh)
            return text
        except Exception as e:
            logger.error(f"Fallback OCR failed: {e}")
            return ""
    
    def decode_text_to_weights(self, text: str) -> Tuple[torch.Tensor, str]:
        """Convert OCR text back to weight tensor"""
        lines = text.strip().split('\n')
        
        # Parse header
        layer_name = ""
        shape = None
        
        for line in lines:
            if line.startswith("LAYER:"):
                layer_name = line.split(":", 1)[1].strip()
            elif line.startswith("SHAPE:"):
                shape_str = line.split(":", 1)[1].strip()
                shape = eval(shape_str)  # Convert string representation to tuple
        
        # Find values section
        values_start = -1
        for i, line in enumerate(lines):
            if line.strip() == "VALUES:":
                values_start = i + 1
                break
        
        if values_start == -1:
            raise ValueError("Could not find VALUES section in OCR text")
        
        # Extract values
        values_text = " ".join(lines[values_start:])
        values = values_text.split()
        
        # Convert values based on encoding scheme
        if self.config.encoding_scheme == "scientific":
            float_values = [float(v) for v in values if v]
        elif self.config.encoding_scheme == "decimal":
            float_values = [float(v) for v in values if v]
        elif self.config.encoding_scheme == "hex":
            float_values = [int(v, 16) / 1000000.0 for v in values if v]  # Scale back
        else:  # binary
            float_values = [int(v, 2) / 1000000.0 for v in values if v]  # Scale back
        
        # Reshape to original shape
        if shape:
            tensor = torch.tensor(float_values).reshape(shape)
        else:
            tensor = torch.tensor(float_values)
        
        return tensor, layer_name


class OCRMemoryBank:
    """Memory bank that stores and retrieves information using OCR format"""
    
    def __init__(self, config: OCRMemoryConfig):
        self.config = config
        self.encoder = OCREncoder(config)
        self.decoder = OCRDecoder(config)
        self.memory_images: Dict[str, Image.Image] = {}
        self.memory_metadata: Dict[str, Dict[str, Any]] = {}
        self.similarity_cache: Dict[str, torch.Tensor] = {}
    
    def store_memory(self, key: str, weights: torch.Tensor, metadata: Dict[str, Any] = None) -> str:
        """Store weights as OCR image in memory bank"""
        # Create OCR image
        image_id = f"{key}_{len(self.memory_images)}"
        memory_image = self.encoder.encode_weights_to_image(weights, key, image_id)
        
        # Compress if enabled
        if self.config.use_visual_compression:
            memory_image = self.encoder.compress_memory_image(memory_image)
        
        # Store in memory bank
        self.memory_images[image_id] = memory_image
        self.memory_metadata[image_id] = {
            "key": key,
            "shape": list(weights.shape),
            "dtype": str(weights.dtype),
            "timestamp": torch.tensor([torch.tensor(0).item()]),  # Placeholder
            **(metadata or {})
        }
        
        # Update cache
        if len(self.similarity_cache) < self.config.cache_size:
            self.similarity_cache[image_id] = self._compute_image_features(memory_image)
        
        return image_id
    
    def retrieve_memory(self, query_key: str, similarity_threshold: float = None) -> List[Tuple[str, torch.Tensor, Dict[str, Any]]]:
        """Retrieve memories similar to query key"""
        if similarity_threshold is None:
            similarity_threshold = self.config.similarity_threshold
        
        results = []
        
        for image_id, metadata in self.memory_metadata.items():
            if metadata["key"] == query_key:
                # Direct key match
                image = self.memory_images[image_id]
                weights, _ = self.decoder.decode_text_to_weights(
                    self.decoder.decode_image_to_text(image)
                )
                results.append((image_id, weights, metadata))
            else:
                # Similarity-based retrieval
                if image_id in self.similarity_cache:
                    similarity = self._compute_similarity(query_key, image_id)
                    if similarity > similarity_threshold:
                        image = self.memory_images[image_id]
                        weights, _ = self.decoder.decode_text_to_weights(
                            self.decoder.decode_image_to_text(image)
                        )
                        results.append((image_id, weights, metadata))
        
        return results
    
    def _compute_image_features(self, image: Image.Image) -> torch.Tensor:
        """Compute visual features for similarity comparison"""
        # Convert to tensor and compute basic features
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Compute histogram features
        hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        return torch.tensor(hist, dtype=torch.float32)
    
    def _compute_similarity(self, query_key: str, image_id: str) -> float:
        """Compute similarity between query and stored memory"""
        # Simple key-based similarity for now
        # In practice, this could use more sophisticated similarity measures
        if query_key in image_id or image_id in query_key:
            return 0.9
        return 0.1
    
    def get_memory_image(self, image_id: str) -> Optional[Image.Image]:
        """Get memory image by ID"""
        return self.memory_images.get(image_id)
    
    def list_memories(self) -> List[Dict[str, Any]]:
        """List all stored memories"""
        return [
            {
                "image_id": image_id,
                "metadata": metadata
            }
            for image_id, metadata in self.memory_metadata.items()
        ]
    
    def clear_memory(self, image_id: str = None):
        """Clear specific memory or all memories"""
        if image_id:
            if image_id in self.memory_images:
                del self.memory_images[image_id]
                del self.memory_metadata[image_id]
                if image_id in self.similarity_cache:
                    del self.similarity_cache[image_id]
        else:
            self.memory_images.clear()
            self.memory_metadata.clear()
            self.similarity_cache.clear()


class OCRContextualMemory:
    """Contextual memory system using OCR format for enhanced pattern recognition"""
    
    def __init__(self, config: OCRMemoryConfig, d_model: int = 768):
        self.config = config
        self.d_model = d_model
        self.memory_bank = OCRMemoryBank(config)
        
        # Contextual processing layers
        self.context_encoder = nn.Linear(d_model, d_model)
        self.pattern_matcher = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.memory_fusion = nn.Linear(d_model * 2, d_model)
        
    def store_contextual_memory(self, context: torch.Tensor, weights: torch.Tensor, 
                              context_type: str = "general") -> str:
        """Store memory with contextual information"""
        # Encode context
        context_encoded = self.context_encoder(context)
        
        # Create contextual metadata
        metadata = {
            "context_type": context_type,
            "context_encoded": context_encoded.detach().cpu().numpy().tolist(),
            "context_shape": list(context.shape)
        }
        
        # Store in OCR format
        key = f"{context_type}_{hash(str(context_encoded.flatten().tolist()))}"
        return self.memory_bank.store_memory(key, weights, metadata)
    
    def retrieve_contextual_memory(self, query_context: torch.Tensor, 
                                 context_type: str = None) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Retrieve memories based on contextual similarity"""
        # Encode query context
        query_encoded = self.context_encoder(query_context)
        
        # Find similar contexts
        similar_memories = []
        
        for image_id, metadata in self.memory_bank.memory_metadata.items():
            if context_type and metadata.get("context_type") != context_type:
                continue
            
            # Compute context similarity
            stored_context = torch.tensor(metadata["context_encoded"])
            similarity = F.cosine_similarity(
                query_encoded.flatten(), 
                stored_context.flatten(), 
                dim=0
            ).item()
            
            if similarity > 0.7:  # Threshold for contextual similarity
                # Retrieve the memory
                memory_results = self.memory_bank.retrieve_memory(image_id)
                for _, weights, meta in memory_results:
                    similar_memories.append((weights, meta))
        
        return similar_memories
    
    def update_contextual_memory(self, context: torch.Tensor, new_weights: torch.Tensor,
                               context_type: str = "general"):
        """Update existing contextual memory"""
        # Find existing memory for this context
        existing_memories = self.retrieve_contextual_memory(context, context_type)
        
        if existing_memories:
            # Update existing memory
            for weights, metadata in existing_memories:
                # Find the image_id for this memory
                for image_id, meta in self.memory_bank.memory_metadata.items():
                    if meta == metadata:
                        # Update the memory
                        self.memory_bank.store_memory(
                            metadata["key"], 
                            new_weights, 
                            metadata
                        )
                        break
        else:
            # Create new memory
            self.store_contextual_memory(context, new_weights, context_type)


class OCRParameterEfficientMemory:
    """Parameter-efficient memory storage using OCR-based representations"""
    
    def __init__(self, config: OCRMemoryConfig, compression_ratio: float = 0.1):
        self.config = config
        self.compression_ratio = compression_ratio
        self.memory_bank = OCRMemoryBank(config)
        
        # Parameter sharing and compression
        self.shared_embeddings = nn.Embedding(1000, 64)  # Shared parameter space
        self.compression_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decompression_net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
    
    def compress_and_store(self, weights: torch.Tensor, layer_name: str) -> str:
        """Compress weights and store in OCR format"""
        # Flatten weights
        flat_weights = weights.flatten()
        
        # Create compressed representation
        compressed_indices = []
        for i in range(0, len(flat_weights), 64):
            chunk = flat_weights[i:i+64]
            if len(chunk) < 64:
                chunk = F.pad(chunk, (0, 64 - len(chunk)))
            
            # Find closest shared embedding
            chunk_emb = self.shared_embeddings.weight
            similarities = F.cosine_similarity(chunk.unsqueeze(0), chunk_emb)
            closest_idx = similarities.argmax().item()
            compressed_indices.append(closest_idx)
        
        # Further compress using neural compression
        compressed_repr = self.compression_net(
            self.shared_embeddings(torch.tensor(compressed_indices))
        )
        
        # Store compressed representation as OCR
        metadata = {
            "layer_name": layer_name,
            "original_shape": list(weights.shape),
            "compression_ratio": self.compression_ratio,
            "compressed_indices": compressed_indices
        }
        
        key = f"compressed_{layer_name}_{hash(str(compressed_indices))}"
        return self.memory_bank.store_memory(key, compressed_repr, metadata)
    
    def decompress_and_retrieve(self, layer_name: str) -> Optional[torch.Tensor]:
        """Decompress and retrieve weights"""
        # Find compressed memory
        memories = self.memory_bank.retrieve_memory(f"compressed_{layer_name}")
        
        if not memories:
            return None
        
        # Get the most recent memory
        _, compressed_repr, metadata = memories[0]
        
        # Decompress
        decompressed = self.decompression_net(compressed_repr)
        
        # Reconstruct using shared embeddings
        compressed_indices = metadata["compressed_indices"]
        reconstructed = []
        
        for idx in compressed_indices:
            chunk = self.shared_embeddings(torch.tensor(idx))
            reconstructed.append(chunk)
        
        # Reshape to original shape
        original_shape = metadata["original_shape"]
        weights = torch.cat(reconstructed).reshape(original_shape)
        
        return weights


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = OCRMemoryConfig(
        image_width=256,
        image_height=256,
        precision_digits=4,
        encoding_scheme="scientific"
    )
    
    # Test OCR memory system
    memory_bank = OCRMemoryBank(config)
    
    # Create sample weights
    sample_weights = torch.randn(4, 4)
    
    # Store memory
    image_id = memory_bank.store_memory("test_layer", sample_weights)
    print(f"Stored memory with ID: {image_id}")
    
    # Retrieve memory
    memories = memory_bank.retrieve_memory("test_layer")
    if memories:
        retrieved_id, retrieved_weights, metadata = memories[0]
        print(f"Retrieved weights shape: {retrieved_weights.shape}")
        print(f"Original shape: {sample_weights.shape}")
        print(f"Reconstruction error: {F.mse_loss(sample_weights, retrieved_weights).item():.6f}")
    
    # Test contextual memory
    contextual_memory = OCRContextualMemory(config)
    context = torch.randn(1, 10, 768)
    
    # Store contextual memory
    context_id = contextual_memory.store_contextual_memory(context, sample_weights, "test_context")
    print(f"Stored contextual memory with ID: {context_id}")
    
    # Retrieve contextual memory
    retrieved_contexts = contextual_memory.retrieve_contextual_memory(context, "test_context")
    print(f"Retrieved {len(retrieved_contexts)} contextual memories")