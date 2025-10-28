from __future__ import annotations

from typing import Dict, Any
import torch
from core.models.compute_routing import ComputeRouter
from core.memory.advanced_memory import AdvancedMemoryManager
from core.fusion.multimodal_fusion import MultimodalFusion


class ResponseGenerator:
    """Stub: builds prompts, runs SpikingBrain, post-process via personality."""

    def __init__(self, spiking_model, fusion_orchestrator, tokenizer, personality_layer):
        self.model = spiking_model
        self.fusion = fusion_orchestrator
        self.tokenizer = tokenizer
        self.personality = personality_layer
        self.compute_router = ComputeRouter()
        self.memory_manager = AdvancedMemoryManager()
        self.multimodal_fusion = MultimodalFusion(
            text_dim=4096,
            vision_dim=1024,
            audio_dim=1024,
            hidden_dim=4096
        )

    def generate(self, perception_out: Dict[str, Any], decision: Dict[str, Any]) -> str:
        # Get input text for analysis
        input_text = ""
        if perception_out.get("text_tokens"):
            try:
                input_text = self.tokenizer.decode(perception_out["text_tokens"], skip_special_tokens=True)
            except:
                input_text = str(perception_out.get("text_tokens", ""))
        
        # Store in memory
        if input_text:
            self.memory_manager.store(
                content=input_text,
                importance=0.6,
                modality='text',
                metadata={'mode': decision.get('mode', 'default')}
            )
        
        # Recall relevant memories
        relevant_memories = self.memory_manager.recall(input_text, top_k=3)
        memory_context = " ".join(relevant_memories) if relevant_memories else ""
        
        # Build enhanced context
        if memory_context:
            enhanced_input = f"[Context: {memory_context}] {input_text}"
        else:
            enhanced_input = input_text
        
        if self.model is None:
            # Fallback if model not loaded
            if "hello" in input_text.lower():
                return "Hello! I'm here to help, though I'm running in fallback mode."
            elif "what" in input_text.lower() or "?" in input_text:
                return "I understand you have a question. I'm currently running in fallback mode, so I can't provide detailed responses right now."
            elif "mode" in input_text.lower():
                return f"I'm currently in {decision.get('mode', 'default')} mode, running in fallback mode."
            else:
                return f"I received your message: '{input_text[:100]}...' I'm running in fallback mode."
        
        # Determine compute path based on complexity
        context_len = len(perception_out.get("text_tokens", []))
        compute_path = self.compute_router.select_path(enhanced_input, context_len)
        max_tokens = self.compute_router.get_max_tokens(compute_path)
        
        # Get personality parameters
        params = self.personality.parameterize(decision["mode"])
        prefix = params.get("prompt_prefix", "")
        
        # Build input sequence with enhanced context
        enhanced_tokens = self.tokenizer.encode(enhanced_input, add_special_tokens=True)
        text_tokens = self.tokenizer.encode(prefix, add_special_tokens=False) + enhanced_tokens
        
        # Handle multimodal inputs
        vision_embeds = perception_out.get("vision_embeds")
        audio_embeds = perception_out.get("audio_embeds")
        
        if vision_embeds is not None or audio_embeds is not None:
            # Use multimodal fusion
            try:
                # Convert text tokens to embeddings
                text_embeds = self.model.transformer.wte(torch.tensor(text_tokens, dtype=torch.long))
                
                # Fuse modalities
                fused_embeds = self.multimodal_fusion(
                    text_embeds=text_embeds,
                    vision_embeds=vision_embeds,
                    audio_embeds=audio_embeds
                )
                
                # Generate with fused embeddings
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs_embeds=fused_embeds.unsqueeze(0),
                        max_new_tokens=min(max_tokens, params.get("max_tokens", max_tokens)),
                        temperature=params.get("temperature", 0.8),
                        top_p=params.get("top_p", 0.9),
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove prefix if present
                if prefix and prefix in generated_text:
                    result = generated_text.split(prefix, 1)[-1].strip()
                    return result
                
                return generated_text
                
            except Exception as e:
                # Fallback to text-only generation
                pass
        
        # Text-only generation
        try:
            # Create attention mask
            attention_mask = torch.ones(len(text_tokens), dtype=torch.long)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=torch.tensor(text_tokens, dtype=torch.long).unsqueeze(0),
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
            
            # Remove prefix if present
            if prefix and prefix in generated_text:
                result = generated_text.split(prefix, 1)[-1].strip()
                return result
            
            return generated_text
            
        except Exception as e:
            return f"Generation error: {e}"


