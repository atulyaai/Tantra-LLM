from __future__ import annotations

import logging
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, Any] = {}
        logger.info(f"ModelLoader initialized on device: {self.device}")
    
    def load_spikingbrain(self, path: Optional[str] = None) -> Any:
        """Load SpikingBrain-7B locally.
        
        Args:
            path: Local model path, defaults to config value
            
        Returns:
            Loaded model and tokenizer
        """
        if "spikingbrain" in self.models:
            return self.models["spikingbrain"]
        
        model_path = path or self.config["spikingbrain"]["path"]
        
        if not Path(model_path).exists():
            logger.warning(f"Model not found at {model_path}, using stub")
            self.models["spikingbrain"] = {"model": None, "tokenizer": None}
            return self.models["spikingbrain"]
        
        try:
            logger.info(f"Loading SpikingBrain from {model_path}")
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            self.models["spikingbrain"] = {"model": model, "tokenizer": tokenizer}
            logger.info("SpikingBrain loaded successfully")
            return self.models["spikingbrain"]
            
        except Exception as e:
            logger.error(f"Failed to load SpikingBrain: {e}")
            self.models["spikingbrain"] = {"model": None, "tokenizer": None}
            return self.models["spikingbrain"]
    
    def load_whisper(self, model_size: str = "large-v3") -> Any:
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
            logger.info(f"Loading Whisper {model_size}")
            
            model = whisper.load_model(model_size, device=self.device)
            
            self.models["whisper"] = model
            logger.info("Whisper loaded successfully")
            return self.models["whisper"]
            
        except ImportError:
            logger.warning("Whisper not installed, using stub")
            self.models["whisper"] = None
            return self.models["whisper"]
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            self.models["whisper"] = None
            return self.models["whisper"]
    
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
            
            tts = TTS(model_name=model_name, gpu=torch.cuda.is_available())
            
            self.models["coqui_tts"] = tts
            logger.info("Coqui TTS loaded successfully")
            return self.models["coqui_tts"]
            
        except ImportError:
            logger.warning("Coqui TTS not installed, using stub")
            self.models["coqui_tts"] = None
            return self.models["coqui_tts"]
        except Exception as e:
            logger.error(f"Failed to load Coqui TTS: {e}")
            self.models["coqui_tts"] = None
            return self.models["coqui_tts"]
    
    def get_long_vit_embeddings_remote(self, image_data: bytes, api_key: Optional[str] = None) -> torch.Tensor:
        """Get vision embeddings from remote Long-VITA API (stub for now).
        
        Args:
            image_data: Image bytes
            api_key: API key for remote service
            
        Returns:
            Vision embeddings tensor (stub zeros for now)
        """
        logger.info("Long-VITA remote API not implemented, returning stub")
        return torch.zeros(1, self.config["vision"]["embed_dim"])
    
    def get_vision_encoder(self, mode: str = "remote") -> Any:
        """Get vision encoder (remote or local).
        
        Args:
            mode: 'remote' or 'local'
            
        Returns:
            Vision encoder wrapper
        """
        from encoders.vision import VisionEncoder
        
        if mode == "remote":
            encoder = VisionEncoder(embed_dim=self.config["vision"]["embed_dim"])
            encoder._remote = True
            encoder._api_func = self.get_long_vit_embeddings_remote
            return encoder
        else:
            logger.warning("Local vision encoder not implemented, using remote")
            return self.get_vision_encoder(mode="remote")


