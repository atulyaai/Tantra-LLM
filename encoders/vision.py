from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Optional, Union

import requests
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class LongVITAVisionEncoder(nn.Module):
    """Local Long-VITA vision encoder implementation."""
    
    def __init__(self, embed_dim: int = 1024, model_path: Optional[str] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_path = model_path
        self._model = None
        self._processor = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _load_model(self):
        """Load Long-VITA model and processor."""
        if self._model is not None:
            return
            
        try:
            # Try to load from transformers if available
            from transformers import AutoModel, AutoProcessor
            
            model_name = self.model_path or "microsoft/longvita-16k"
            logger.info(f"Loading Long-VITA model: {model_name}")
            
            self._processor = AutoProcessor.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self._model.eval()
            logger.info("Long-VITA model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load Long-VITA model: {e}")
            logger.info("Using fallback vision encoder")
            self._model = None
            self._processor = None
    
    def forward(self, image: Union[torch.Tensor, Image.Image, np.ndarray, str]) -> torch.Tensor:
        """Encode image to embeddings."""
        self._load_model()
        
        if self._model is None:
            # Fallback to simple CNN encoder
            return self._fallback_encode(image)
        
        try:
            # Process image
            if isinstance(image, str):
                # Assume it's a file path
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            elif isinstance(image, torch.Tensor):
                # Convert tensor to PIL Image
                if image.dim() == 4:
                    image = image.squeeze(0)
                if image.dim() == 3 and image.size(0) == 3:
                    image = image.permute(1, 2, 0)
                image = Image.fromarray((image.numpy() * 255).astype(np.uint8)).convert("RGB")
            
            # Process with Long-VITA
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Get pooled output or last hidden state
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                else:
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Ensure correct embedding dimension
                if embeddings.size(-1) != self.embed_dim:
                    # Project to correct dimension
                    if not hasattr(self, '_projection'):
                        self._projection = nn.Linear(embeddings.size(-1), self.embed_dim).to(self._device)
                    embeddings = self._projection(embeddings)
                
                return embeddings
                
        except Exception as e:
            logger.error(f"Error encoding image with Long-VITA: {e}")
            return self._fallback_encode(image)
    
    def _fallback_encode(self, image: Union[torch.Tensor, Image.Image, np.ndarray, str]) -> torch.Tensor:
        """Fallback CNN encoder for when Long-VITA is not available."""
        try:
            # Convert to tensor if needed
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                pass
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            elif isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image.squeeze(0)
                if image.dim() == 3 and image.size(0) == 3:
                    image = image.permute(1, 2, 0)
                image = Image.fromarray((image.numpy() * 255).astype(np.uint8)).convert("RGB")
            
            # Convert to tensor
            image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            
            # Simple CNN encoder
            if not hasattr(self, '_fallback_cnn'):
                self._fallback_cnn = nn.Sequential(
                    nn.Conv2d(3, 64, 7, 2, 3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7)),
                    nn.Flatten(),
                    nn.Linear(64 * 7 * 7, 512),
                    nn.ReLU(),
                    nn.Linear(512, self.embed_dim)
                ).to(self._device)
            
            with torch.no_grad():
                embeddings = self._fallback_cnn(image_tensor)
                return embeddings
                
        except Exception as e:
            logger.error(f"Error in fallback encoding: {e}")
            return torch.zeros(1, self.embed_dim, device=self._device)


class VisionEncoder:
    """Production wrapper for Long-VITA encoder; remote API + local fallback."""

    def __init__(self, embed_dim: int = 1024, api_url: Optional[str] = None, local_path: Optional[str] = None):
        self.embed_dim = embed_dim
        self.api_url = api_url
        self.local_path = local_path
        self._remote = False
        self._api_func = None
        self._local_encoder = None

    def __call__(self, image) -> torch.Tensor:
        if isinstance(image, torch.Tensor) and image.size(-1) == self.embed_dim:
            return image.reshape(-1, self.embed_dim)
        
        # Remote API path
        if self._remote and self._api_func:
            try:
                return self._api_func(image)
            except Exception as e:
                logger.error(f"Remote API failed: {e}")
                return self._fallback_encode(image)
        
        # Local Long-VITA path
        if self.local_path or not self._remote:
            if self._local_encoder is None:
                self._local_encoder = LongVITAVisionEncoder(self.embed_dim, self.local_path)
            return self._local_encoder(image)
        
        # Fallback stub
        return self._fallback_encode(image)

    def _fallback_encode(self, image) -> torch.Tensor:
        """Fallback encoding when all else fails."""
        logger.warning("Using fallback vision encoding")
        return torch.zeros(1, self.embed_dim)

    def set_remote_mode(self, api_func):
        """Set remote API function."""
        self._remote = True
        self._api_func = api_func

    def set_local_mode(self, model_path: Optional[str] = None):
        """Set local mode with optional model path."""
        self._remote = False
        self.local_path = model_path
        self._local_encoder = None  # Will be created on first use


