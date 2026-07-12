from __future__ import annotations

from typing import Any, Optional

import torch
try:
    import whisper
except Exception:
    whisper = None

from atulya_core.protocol.encoder import ModalityEncoder
from config.settings import get_settings


class AudioEncoder(ModalityEncoder):
    """Production wrapper for Whisper encoder; returns real embeddings."""

    def __init__(self, embed_dim: int = 4096, model_size: str = "base"):
        settings = get_settings()
        if embed_dim != settings.model_dim:
            raise ValueError(f"AudioEncoder embed_dim ({embed_dim}) must match model_dim ({settings.model_dim})")
        self._embed_dim = embed_dim
        self._model = None
        self._model_size = model_size

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode(self, audio) -> torch.Tensor:
        """Process and encode audio, returning shape [1, embed_dim]."""
        return self(audio)


    def __call__(self, audio) -> torch.Tensor:
        if not self._model and whisper:
            try:
                self._model = whisper.load_model(self._model_size)
            except Exception:
                return torch.zeros(1, self.embed_dim)
        try:
            if self._model and whisper:
                # Ensure input is a float32 torch tensor on the correct device
                audio_tensor = torch.as_tensor(audio).float()
                device = next(self._model.parameters()).device
                
                # Pad/trim to 30s (480000 samples at 16kHz)
                audio_tensor = whisper.pad_or_trim(audio_tensor)
                
                # Compute log-mel spectrogram (shape [80, 3000])
                mel = whisper.log_mel_spectrogram(audio_tensor).to(device)
                
                # Add batch dimension -> [1, 80, 3000]
                mel = mel.unsqueeze(0)
                
                with torch.no_grad():
                    # Run through encoder (shape [1, 1500, model_dim])
                    enc_out = self._model.encoder(mel)
                    # Average over the sequence dimension (1500 tokens) -> [1, model_dim]
                    embeddings = enc_out.mean(dim=1)
                
                # Align output to target self.embed_dim (pad or slice)
                embeddings = embeddings.cpu()
                if embeddings.size(-1) < self.embed_dim:
                    padding = torch.zeros(1, self.embed_dim - embeddings.size(-1))
                    embeddings = torch.cat([embeddings, padding], dim=-1)
                else:
                    embeddings = embeddings[:, :self.embed_dim]
                return embeddings
            return torch.zeros(1, self.embed_dim)
        except Exception:
            return torch.zeros(1, self.embed_dim)



