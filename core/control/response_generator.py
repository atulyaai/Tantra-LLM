from __future__ import annotations

from typing import Dict, Any


class ResponseGenerator:
    """Stub: builds prompts, runs SpikingBrain, post-process via personality."""

    def __init__(self, spiking_model, fusion_orchestrator, tokenizer, personality_layer):
        self.model = spiking_model
        self.fusion = fusion_orchestrator
        self.tokenizer = tokenizer
        self.personality = personality_layer

    def generate(self, perception_out: Dict[str, Any], decision: Dict[str, Any]) -> str:
        params = self.personality.parameterize(decision["mode"])  # decoding params, prefixes
        prefix = params.get("prompt_prefix", "")
        text_tokens = self.tokenizer.encode(prefix, add_special_tokens=False) + perception_out.get("text_tokens", [])
        _stream = self.fusion.build_stream(
            text_tokens=text_tokens,
            vision_embeds=perception_out.get("vision_embeds"),
            audio_embeds=perception_out.get("audio_embeds"),
        )
        return prefix.strip()


