from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional
import os
import json

class MemorySettings(BaseModel):
    working_tokens: int = 8192
    episodic_threshold: float = 0.7
    embedding_dim: int = 4096
    max_episodic: int = 10000
    consolidation_frequency: int = 100

class SpikingBrainSettings(BaseModel):
    max_seq: int = 32768
    path: str = "Model/spikingbrain-7b"
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_hidden_layers: int = 24
    intermediate_size: int = 16384
    vocab_size: int = 50257
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

class VisionSettings(BaseModel):
    embed_dim: int = 4096
    remote: bool = True
    api_url: Optional[str] = None
    local_path: Optional[str] = None

class AudioSettings(BaseModel):
    embed_dim: int = 4096
    remote: bool = False
    model_name: str = "openai/whisper-large-v3"
    local_path: Optional[str] = None

class ComputeSettings(BaseModel):
    max_tokens: int = 200
    context_window: int = 4096
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.1

class TantraSettings(BaseModel):
    model_dim: int = 4096
    memory: MemorySettings = Field(default_factory=MemorySettings)
    spikingbrain: SpikingBrainSettings = Field(default_factory=SpikingBrainSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    compute: ComputeSettings = Field(default_factory=ComputeSettings)

    @model_validator(mode="after")
    def validate_dimensions(self) -> 'TantraSettings':
        dim = self.model_dim
        if self.spikingbrain.hidden_size != dim:
            raise ValueError(f"SpikingBrain hidden_size ({self.spikingbrain.hidden_size}) must match model_dim ({dim})")
        if self.vision.embed_dim != dim:
            raise ValueError(f"Vision embed_dim ({self.vision.embed_dim}) must match model_dim ({dim})")
        if self.audio.embed_dim != dim:
            raise ValueError(f"Audio embed_dim ({self.audio.embed_dim}) must match model_dim ({dim})")
        if self.memory.embedding_dim != dim:
            raise ValueError(f"Memory embedding_dim ({self.memory.embedding_dim}) must match model_dim ({dim})")
        return self


# Global singleton settings loaded and validated at startup
_settings = None

def get_settings() -> TantraSettings:
    global _settings
    if _settings is None:
        # Load raw dict settings
        try:
            from config.model_config import MODEL_CONFIG
            _settings = TantraSettings(**MODEL_CONFIG)
        except Exception as e:
            # Fall back to default schema config if error loading
            print(f"[Config] Failed to load config from model_config.py: {e}. Falling back to default schema.")
            _settings = TantraSettings()
    return _settings
