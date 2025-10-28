from __future__ import annotations

from typing import Optional, Dict, Any


class Perception:
    """Stub: routes raw inputs to encoders and returns normalized embeddings."""

    def __init__(self, vision_encoder=None, audio_encoder=None, tokenizer=None):
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        self.tokenizer = tokenizer

    def perceive(self, text: Optional[str] = None, image=None, audio=None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"text_tokens": [], "vision_embeds": None, "audio_embeds": None}
        if text and self.tokenizer:
            out["text_tokens"] = self.tokenizer.encode(text, add_special_tokens=True)
        if image and self.vision_encoder:
            out["vision_embeds"] = self.vision_encoder(image)
        if audio and self.audio_encoder:
            out["audio_embeds"] = self.audio_encoder(audio)
        return out


