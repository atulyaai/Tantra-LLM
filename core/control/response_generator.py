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
        if self.model is None:
            # Fallback if model not loaded
            return "Model not available. Please check model loading."
        
        params = self.personality.parameterize(decision["mode"])  # decoding params, prefixes
        prefix = params.get("prompt_prefix", "")
        text_tokens = self.tokenizer.encode(prefix, add_special_tokens=False) + perception_out.get("text_tokens", [])
        
        # Build fusion stream
        stream = self.fusion.build_stream(
            text_tokens=text_tokens,
            vision_embeds=perception_out.get("vision_embeds"),
            audio_embeds=perception_out.get("audio_embeds"),
        )
        
        # Merge embeddings for forward pass
        merged_embeds = self.fusion.merge_embeddings(
            input_ids=stream["input_ids"],
            modality_embeds=stream["modality_embeds"],
            model=self.model,
        )
        
        # Generate with merged embeddings
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs_embeds=merged_embeds.unsqueeze(0),
                    max_new_tokens=params.get("max_tokens", 512),
                    temperature=params.get("temperature", 0.7),
                    top_p=params.get("top_p", 0.9),
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
        except Exception as e:
            return f"Generation error: {e}"


