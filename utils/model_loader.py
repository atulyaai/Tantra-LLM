from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelLoader:
    """Mixed model loading system supporting local/API models.
    
    Handles:
    - Local inference: SpikingBrain-7B (~14GB)
    - API/Remote: Long-VITA (initially, migrate to local later)
    - Local: Whisper Large-v3 (~3GB)
    - Local: Coqui TTS
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Check for forced CPU mode via environment variable
        force_cpu = bool(os.environ.get("TANTRA_FORCE_CPU", "0") in {"1", "true", "True"})
        if force_cpu:
            self.device = torch.device("cpu")
            logger.info("Forced CPU mode via TANTRA_FORCE_CPU environment variable")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, Any] = {}
        logger.info(f"ModelLoader initialized on device: {self.device}")
    
    def load_spikingbrain(self, path: Optional[str] = None) -> Any:
        """Load SpikingBrain model and tokenizer.
        
        Args:
            path: Local model path (optional), defaults to custom SpikingBrain
            
        Returns:
            Loaded model and tokenizer
        """
        if "spikingbrain" in self.models:
            return self.models["spikingbrain"]
        
        # Try to load custom SpikingBrain first
        try:
            from utils.low_resource import is_low_resource_mode
            if is_low_resource_mode():
                logger.info("Low-resource mode: using DistilGPT-2 instead of SpikingBrain")
                raise ImportError("Low-resource mode")
                
            from core.models.spikingbrain_model import SpikingBrainForCausalLM, SpikingBrainConfig
            from transformers import AutoTokenizer
            
            logger.info("Loading custom SpikingBrain-7B model")
            
            # Create SpikingBrain-7B configuration (original)
            sb_config = self.config.get("spikingbrain", {})
            config = SpikingBrainConfig(
                vocab_size=sb_config.get("vocab_size", 50257),
                n_embd=sb_config.get("hidden_size", 4096),
                n_head=sb_config.get("num_attention_heads", 32),
                n_layer=sb_config.get("num_hidden_layers", 24),
                n_inner=sb_config.get("intermediate_size", 16384),
                n_positions=sb_config.get("max_seq", 32768),
                resid_pdrop=0.1,
                attn_pdrop=0.1,
                initializer_range=0.02,
                use_cache=True,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            
            # Create model
            model = SpikingBrainForCausalLM(config)
            
            # Move model to device (CPU or GPU)
            model = model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Resize token embeddings if needed
            if len(tokenizer) != config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))
                config.vocab_size = len(tokenizer)
            
            self.models["spikingbrain"] = {"model": model, "tokenizer": tokenizer}
            logger.info(f"✅ SpikingBrain-7B loaded successfully on {self.device}!")
            return self.models["spikingbrain"]
            
        except Exception as e:
            logger.error(f"SpikingBrain failed to load: {e}")
            logger.error("Please check your SpikingBrain model configuration")
            raise RuntimeError("SpikingBrain is required and failed to load") from e
    
    def load_whisper(self, model_size: str = "base") -> Any:
        """Load Whisper locally.
        
        Args:
            model_size: Whisper model size (tiny/base/small/medium/large-v3)
            
        Returns:
            Loaded Whisper pipeline
        """
        if "whisper" in self.models:
            return self.models["whisper"]
        
        try:
            import whisper
            logger.info(f"Loading Whisper {model_size} on {self.device}")
            # Whisper expects string device name: "cpu" or "cuda"
            device_str = "cpu" if self.device.type == "cpu" else "cuda"
            model = whisper.load_model(model_size, device=device_str)
            
            self.models["whisper"] = model
            logger.info(f"✅ Whisper loaded successfully on {device_str}")
            return self.models["whisper"]
            
        except ImportError:
            logger.error("Whisper not installed. Install with: pip install openai-whisper")
            raise RuntimeError("Whisper is required but not installed")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise RuntimeError(f"Whisper failed to load: {e}") from e
    
    def load_coqui_tts(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC") -> Any:
        """Load Coqui TTS locally.
        
        Args:
            model_name: Coqui TTS model name
            
        Returns:
            Loaded TTS pipeline
        """
        if "coqui_tts" in self.models:
            return self.models["coqui_tts"]
        
        try:
            from TTS.api import TTS
            logger.info(f"Loading Coqui TTS: {model_name}")
            
            # Coqui TTS: gpu=False forces CPU mode, gpu=True only works if CUDA is available
            use_gpu = self.device.type == "cuda" and torch.cuda.is_available()
            tts = TTS(model_name=model_name, gpu=use_gpu)
            
            self.models["coqui_tts"] = tts
            logger.info(f"✅ Coqui TTS loaded successfully on {'GPU' if use_gpu else 'CPU'}")
            return self.models["coqui_tts"]
            
        except ImportError:
            logger.error("Coqui TTS not installed. Install with: pip install TTS")
            raise RuntimeError("Coqui TTS is required but not installed")
        except Exception as e:
            logger.error(f"Failed to load Coqui TTS: {e}")
            raise RuntimeError(f"Coqui TTS failed to load: {e}") from e
    
    def get_long_vit_embeddings_remote(self, image_data: bytes, api_key: Optional[str] = None) -> torch.Tensor:
        """Get vision embeddings from remote Long-VITA API.
        
        Args:
            image_data: Image bytes
            api_key: API key for remote service
            
        Returns:
            Vision embeddings tensor
        """
        try:
            # For now, use local encoder as fallback
            # In production, this would call a remote API
            logger.info("Using local Long-VITA encoder as remote API fallback")
            from encoders.vision import LongVITAVisionEncoder
            
            encoder = LongVITAVisionEncoder(embed_dim=self.config["vision"]["embed_dim"])
            # Ensure encoder is on the same device
            encoder._device = self.device
            
            # Convert bytes to PIL Image
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return encoder(image)
            
        except Exception as e:
            logger.error(f"Remote Long-VITA API failed: {e}")
            return torch.zeros(1, self.config["vision"]["embed_dim"])
    
    def get_vision_encoder(self, mode: str = "local") -> Any:
        """Get vision encoder (remote or local).
        
        Args:
            mode: 'remote' or 'local'
            
        Returns:
            Vision encoder wrapper
        """
        from encoders.vision import VisionEncoder
        
        embed_dim = self.config["vision"]["embed_dim"]
        
        if mode == "remote":
            encoder = VisionEncoder(embed_dim=embed_dim)
            encoder.set_remote_mode(self.get_long_vit_embeddings_remote)
            return encoder
        else:
            # Use local mode by default
            encoder = VisionEncoder(embed_dim=embed_dim)
            encoder.set_local_mode()  # Will use default model path
            return encoder


