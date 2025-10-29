from __future__ import annotations

import base64
import logging
import os
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
        # Check for forced CPU mode
        force_cpu = bool(os.environ.get("TANTRA_FORCE_CPU", "0") in {"1", "true", "True"})
        if force_cpu:
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _load_model(self):
        """Load Long-VITA model and processor."""
        if self._model is not None:
            return
            
        from utils.low_resource import is_low_resource_mode
        if is_low_resource_mode():
            logger.info("Low-resource mode: skipping Long-VITA load, using fallback encoder")
            self._model = None
            self._processor = None
            return
        try:
            # Try to load Long-VITA model
            from transformers import AutoModel, AutoProcessor
            
            model_name = self.model_path or "microsoft/longvita-16k"
            logger.info(f"Attempting to load Long-VITA: {model_name}")
            
            self._processor = AutoProcessor.from_pretrained(model_name)
            use_cuda = self._device.type == "cuda" and torch.cuda.is_available()
            self._model = AutoModel.from_pretrained(
                model_name,
                dtype=torch.float16 if use_cuda else torch.float32,
                device_map="auto" if use_cuda else None
            )
            # If not using device_map, manually move to device
            if not use_cuda:
                self._model = self._model.to(self._device)
            self._model.eval()
            logger.info(f"Long-VITA model loaded successfully on {self._device}")
            
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
            
            # Convert to tensor and move to device
            image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            image_tensor = image_tensor.to(self._device)
            
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
        # Determine device for fallback
        device = getattr(self._local_encoder, '_device', torch.device("cpu")) if self._local_encoder else torch.device("cpu")
        return torch.zeros(1, self.embed_dim, device=device)

    def set_remote_mode(self, api_func):
        """Set remote API function."""
        self._remote = True
        self._api_func = api_func

    def set_local_mode(self, model_path: Optional[str] = None):
        """Set local mode with optional model path."""
        self._remote = False
        self.local_path = model_path
        self._local_encoder = None  # Will be created on first use


