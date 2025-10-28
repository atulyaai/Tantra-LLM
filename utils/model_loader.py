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
        """Load SpikingBrain model and tokenizer.
        
        Args:
            path: Local model path (optional), defaults to custom SpikingBrain
            
        Returns:
            Loaded model and tokenizer
        """
        if "spikingbrain" in self.models:
            return self.models["spikingbrain"]
        
        try:
            from core.models.spikingbrain_model import SpikingBrainForCausalLM, SpikingBrainConfig
            from transformers import AutoTokenizer
            
            logger.info("Loading custom SpikingBrain model")
            
            # Create SpikingBrain configuration
            config = SpikingBrainConfig(
                vocab_size=50257,
                hidden_size=4096,
                num_attention_heads=32,
                num_hidden_layers=24,
                intermediate_size=16384,
                max_position_embeddings=32768,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                initializer_range=0.02,
                use_cache=True,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            
            # Create model
            model = SpikingBrainForCausalLM(config)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Resize token embeddings if needed
            if len(tokenizer) != config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))
                config.vocab_size = len(tokenizer)
            
            self.models["spikingbrain"] = {"model": model, "tokenizer": tokenizer}
            logger.info("SpikingBrain loaded successfully")
            return self.models["spikingbrain"]
            
        except Exception as e:
            logger.error(f"Failed to load custom SpikingBrain: {e}")
            logger.warning("Falling back to GPT-2")
            
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model_name = path or "gpt2"
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                self.models["spikingbrain"] = {"model": model, "tokenizer": tokenizer}
                logger.info("GPT-2 fallback loaded successfully")
                return self.models["spikingbrain"]
            except Exception as e2:
                logger.error(f"Failed to load GPT-2 fallback: {e2}")
                logger.warning("Using basic tokenizer only")
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    self.models["spikingbrain"] = {"model": None, "tokenizer": tokenizer}
                except:
                    from encoders.text import TextTokenizer
                    tokenizer = TextTokenizer()
                    self.models["spikingbrain"] = {"model": None, "tokenizer": tokenizer}
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
            model = whisper.load_model(model_size, device=self.device.name if hasattr(self.device, 'name') else str(self.device))
            
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


