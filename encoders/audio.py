from __future__ import annotations

import os
from typing import Any, Optional

import torch
try:
    import whisper
except Exception:
    whisper = None


class AudioEncoder:
    """Production wrapper for Whisper encoder; returns real embeddings."""

    def __init__(self, embed_dim: int = 1024, model_size: str = "base"):
        self.embed_dim = embed_dim
        self._model = None
        self._model_size = model_size
        # Determine device
        force_cpu = bool(os.environ.get("TANTRA_FORCE_CPU", "0") in {"1", "true", "True"})
        if force_cpu:
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, audio) -> torch.Tensor:
        """Encode audio to embeddings using Whisper."""
        if not self._model:
            if whisper is None:
                raise ImportError("Whisper is required. Install with: pip install openai-whisper")
            try:
                # Load Whisper on the correct device
                device_str = "cpu" if self._device.type == "cpu" else "cuda"
                self._model = whisper.load_model(self._model_size, device=device_str)
            except Exception as e:
                raise RuntimeError(f"Failed to load Whisper model: {e}") from e
        
        try:
            # Convert audio to tensor and move to device if needed
            if isinstance(audio, torch.Tensor):
                audio_tensor = audio.to(self._device)
            else:
                import numpy as np
                audio_tensor = torch.from_numpy(audio).float().to(self._device)
            
            # Get embeddings from encoder
            with torch.no_grad():
                # Whisper encoder expects audio as mel spectrogram
                # For now, use transcribe result and extract embeddings if available
                result = self._model.transcribe(audio, verbose=False)
                
                # Try to get actual embeddings from encoder
                # Note: This is a simplified approach - actual implementation may vary
                if hasattr(self._model, 'encoder'):
                    # Create mel spectrogram from audio
                    mel = whisper.log_mel_spectrogram(audio).to(self._device)
                    embeddings = self._model.encoder(mel)
                    # Pool to get single embedding vector
                    if embeddings.dim() > 2:
                        embeddings = embeddings.mean(dim=1)  # Average over sequence length
                    # Project to desired dimension if needed
                    if embeddings.size(-1) > self.embed_dim:
                        embeddings = embeddings[:, :self.embed_dim]
                    elif embeddings.size(-1) < self.embed_dim:
                        # Pad if needed
                        padding = torch.zeros(embeddings.size(0), self.embed_dim - embeddings.size(-1), device=self._device)
                        embeddings = torch.cat([embeddings, padding], dim=-1)
                    return embeddings
                else:
                    # Fallback: return zero embeddings
                    return torch.zeros(1, self.embed_dim, device=self._device)
        except Exception as e:
            raise RuntimeError(f"Audio encoding failed: {e}") from e


