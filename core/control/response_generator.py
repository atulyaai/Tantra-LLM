from __future__ import annotations

from typing import Dict, Any
import torch
from core.models.compute_routing import ComputeRouter


class ResponseGenerator:
    """Stub: builds prompts, runs SpikingBrain, post-process via personality."""

    def __init__(self, spiking_model, fusion_orchestrator, tokenizer, personality_layer):
        self.model = spiking_model
        self.fusion = fusion_orchestrator
        self.tokenizer = tokenizer
        self.personality = personality_layer
        self.compute_router = ComputeRouter()

    def generate(self, perception_out: Dict[str, Any], decision: Dict[str, Any]) -> str:
        if self.model is None:
            # Fallback if model not loaded - generate a simple response based on input
            text_tokens = perception_out.get("text_tokens", [])
            if text_tokens and self.tokenizer:
                try:
                    # Decode the input text to understand what was asked
                    input_text = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
                    # Generate a simple response based on the input
                    if "hello" in input_text.lower():
                        return "Hello! I'm here to help, though I'm running in fallback mode."
                    elif "what" in input_text.lower() or "?" in input_text:
                        return "I understand you have a question. I'm currently running in fallback mode, so I can't provide detailed responses right now."
                    elif "mode" in input_text.lower():
                        return f"I'm currently in {decision.get('mode', 'default')} mode, running in fallback mode."
                    else:
                        return f"I received your message: '{input_text[:100]}...' I'm running in fallback mode."
                except:
                    return "I received your input but I'm running in fallback mode and can't process it fully."
            return "I'm running in fallback mode. Please check model loading for full functionality."
        
        # Get input text for complexity analysis
        input_text = ""
        if perception_out.get("text_tokens"):
            try:
                input_text = self.tokenizer.decode(perception_out["text_tokens"], skip_special_tokens=True)
            except:
                input_text = str(perception_out.get("text_tokens", ""))
        
        # Determine compute path based on complexity
        context_len = len(perception_out.get("text_tokens", []))
        compute_path = self.compute_router.select_path(input_text, context_len)
        max_tokens = self.compute_router.get_max_tokens(compute_path)
        
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
        
        # Generate with merged embeddings using compute path settings
        try:
            # Create attention mask
            attention_mask = torch.ones(merged_embeds.shape[0], dtype=torch.long)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs_embeds=merged_embeds.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    max_new_tokens=min(max_tokens, params.get("max_tokens", max_tokens)),
                    temperature=params.get("temperature", 0.8),
                    top_p=params.get("top_p", 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Strip prefix to return only new generation
            if prefix and prefix in generated_text:
                result = generated_text.split(prefix, 1)[-1].strip()
                return result
            
            # If no prefix or prefix not found, return the full text
            return generated_text
        except Exception as e:
            return f"Generation error: {e}"


