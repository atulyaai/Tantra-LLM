from __future__ import annotations

from typing import Dict, Any
import torch
import logging
from core.models.compute_routing import ComputeRouter
from core.memory.advanced_memory import AdvancedMemoryManager
from core.fusion.unified_fusion import FusionOrchestrator
from utils.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity, SafeOperation
from utils.performance_optimizer import performance_optimized, global_optimizer

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Stub: builds prompts, runs SpikingBrain, post-process via personality."""

    def __init__(self, spiking_model, fusion_orchestrator, tokenizer, personality_layer):
        self.model = spiking_model
        self.fusion = fusion_orchestrator
        self.tokenizer = tokenizer
        self.personality = personality_layer
        self.compute_router = ComputeRouter()
        self.memory_manager = AdvancedMemoryManager()
        self.multimodal_fusion = FusionOrchestrator(
            text_dim=4096,
            vision_dim=1024,
            audio_dim=1024,
            model_dim=4096,
            fusion_type="unified"
        )
        self.error_handler = ErrorHandler()

    @performance_optimized("response_generation", enable_caching=True, enable_monitoring=True)
    def generate(self, perception_out: Dict[str, Any], decision: Dict[str, Any]) -> str:
        """Generate response with comprehensive error handling."""
        try:
            # Get input text for analysis
            input_text = self._extract_input_text(perception_out)
            
            # Store in memory with error handling
            self._store_memory_safely(input_text, decision)
            
            # Recall relevant memories with error handling
            memory_context = self._recall_memories_safely(input_text, decision)
            
            # Build enhanced context
            enhanced_input = self._build_enhanced_context(input_text, memory_context)
            
            # Generate response with error handling
            return self._generate_response_safely(enhanced_input, perception_out, decision)
            
        except Exception as e:
            logger.error(f"Critical error in response generation: {e}")
            return self._generate_fallback_response(perception_out, decision)
    
    @performance_optimized("text_extraction", enable_caching=False, enable_monitoring=True)
    def _extract_input_text(self, perception_out: Dict[str, Any]) -> str:
        """Extract input text with error handling."""
        with SafeOperation("text_extraction", ErrorCategory.GENERATION, ErrorSeverity.MEDIUM, self.error_handler) as safe_op:
            if perception_out.get("text_tokens"):
                return safe_op.execute(
                    self.tokenizer.decode, 
                    perception_out["text_tokens"], 
                    skip_special_tokens=True
                ) or str(perception_out.get("text_tokens", ""))
            return ""
    
    @performance_optimized("memory_storage", enable_caching=False, enable_monitoring=True)
    def _store_memory_safely(self, input_text: str, decision: Dict[str, Any]) -> None:
        """Store memory with error handling."""
        if not input_text:
            return
            
        with SafeOperation("memory_storage", ErrorCategory.MEMORY_OPERATION, ErrorSeverity.LOW, self.error_handler) as safe_op:
            safe_op.execute(
                self.memory_manager.store,
                content=input_text,
                importance=decision.get('storage_importance', 0.6),
                modality='text',
                metadata={'mode': decision.get('mode', 'default')}
            )
    
    @performance_optimized("memory_recall", enable_caching=True, enable_monitoring=True)
    def _recall_memories_safely(self, input_text: str, decision: Dict[str, Any]) -> str:
        """Recall memories with error handling."""
        if not input_text:
            return ""
            
        with SafeOperation("memory_recall", ErrorCategory.MEMORY_OPERATION, ErrorSeverity.LOW, self.error_handler) as safe_op:
            recall_depth = decision.get('recall_depth', 3)
            relevant_memories = safe_op.execute(
                self.memory_manager.recall,
                input_text,
                top_k=recall_depth
            ) or []
            
            return " ".join(relevant_memories) if relevant_memories else ""
    
    def _build_enhanced_context(self, input_text: str, memory_context: str) -> str:
        """Build enhanced context with error handling."""
        if memory_context:
            return f"[Context: {memory_context}] {input_text}"
        return input_text
    
    def _generate_response_safely(self, enhanced_input: str, perception_out: Dict[str, Any], decision: Dict[str, Any]) -> str:
        """Generate response with comprehensive error handling."""
        # Check for safety concerns
        safety_check = decision.get('safety_check', {'safe': True})
        if not safety_check.get('safe', True):
            return self._generate_safety_response(safety_check)
        
        # Get personality parameters
        params = self.personality.parameterize(decision["mode"])
        prefix = params.get("prompt_prefix", "")
        
        # Build input sequence
        text_tokens = self._build_input_sequence(enhanced_input, prefix)
        
        # Handle multimodal inputs
        vision_embeds = perception_out.get("vision_embeds")
        audio_embeds = perception_out.get("audio_embeds")
        
        if vision_embeds is not None or audio_embeds is not None:
            return self._generate_multimodal_response(text_tokens, vision_embeds, audio_embeds, params)
        else:
            return self._generate_text_response(text_tokens, params)
    
    def _build_input_sequence(self, enhanced_input: str, prefix: str) -> list:
        """Build input token sequence with error handling."""
        with SafeOperation("tokenization", ErrorCategory.GENERATION, ErrorSeverity.MEDIUM, self.error_handler) as safe_op:
            enhanced_tokens = safe_op.execute(
                self.tokenizer.encode,
                enhanced_input,
                add_special_tokens=True
            ) or []
            
            prefix_tokens = safe_op.execute(
                self.tokenizer.encode,
                prefix,
                add_special_tokens=False
            ) or []
            
            return prefix_tokens + enhanced_tokens
    
    def _generate_multimodal_response(self, text_tokens: list, vision_embeds, audio_embeds, params: Dict) -> str:
        """Generate multimodal response with error handling."""
        with SafeOperation("multimodal_generation", ErrorCategory.FUSION_PROCESSING, ErrorSeverity.HIGH, self.error_handler) as safe_op:
            if self.model is None:
                return self._generate_fallback_response({"text_tokens": text_tokens}, {})
            
            # Convert text tokens to embeddings
            text_embeds = safe_op.execute(
                self.model.get_input_embeddings,
                torch.tensor(text_tokens, dtype=torch.long)
            )
            
            if text_embeds is None:
                return self._generate_fallback_response({"text_tokens": text_tokens}, {})
            
            # Fuse modalities
            fused_embeds = safe_op.execute(
                self.multimodal_fusion.fuse,
                text_embeds=text_embeds,
                vision_embeds=vision_embeds,
                audio_embeds=audio_embeds
            )
            
            if fused_embeds is None:
                return self._generate_fallback_response({"text_tokens": text_tokens}, {})
            
            # Generate with fused embeddings
            return self._generate_with_embeddings(fused_embeds, params)
    
    def _generate_text_response(self, text_tokens: list, params: Dict) -> str:
        """Generate text-only response with error handling."""
        with SafeOperation("text_generation", ErrorCategory.GENERATION, ErrorSeverity.MEDIUM, self.error_handler) as safe_op:
            if self.model is None:
                return self._generate_fallback_response({"text_tokens": text_tokens}, {})
            
            # Create attention mask
            attention_mask = torch.ones(len(text_tokens), dtype=torch.long)
            
            # Generate response
            outputs = safe_op.execute(
                self.model.generate,
                input_ids=torch.tensor(text_tokens, dtype=torch.long).unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                max_new_tokens=min(200, params.get("max_tokens", 200)),
                temperature=params.get("temperature", 0.8),
                top_p=params.get("top_p", 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            
            if outputs is None:
                return self._generate_fallback_response({"text_tokens": text_tokens}, {})
            
            # Decode response
            generated_text = safe_op.execute(
                self.tokenizer.decode,
                outputs[0],
                skip_special_tokens=True
            ) or "I apologize, but I encountered an error generating a response."
            
            # Remove prefix if present
            prefix = params.get("prompt_prefix", "")
            if prefix and prefix in generated_text:
                return generated_text.split(prefix, 1)[-1].strip()
            
            return generated_text
    
    def _generate_with_embeddings(self, embeddings: torch.Tensor, params: Dict) -> str:
        """Generate response using embeddings with error handling."""
        with SafeOperation("embedding_generation", ErrorCategory.GENERATION, ErrorSeverity.MEDIUM, self.error_handler) as safe_op:
            outputs = safe_op.execute(
                self.model.generate,
                inputs_embeds=embeddings.unsqueeze(0),
                max_new_tokens=min(200, params.get("max_tokens", 200)),
                temperature=params.get("temperature", 0.8),
                top_p=params.get("top_p", 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            
            if outputs is None:
                return "I apologize, but I encountered an error generating a response."
            
            generated_text = safe_op.execute(
                self.tokenizer.decode,
                outputs[0],
                skip_special_tokens=True
            ) or "I apologize, but I encountered an error generating a response."
            
            # Remove prefix if present
            prefix = params.get("prompt_prefix", "")
            if prefix and prefix in generated_text:
                return generated_text.split(prefix, 1)[-1].strip()
            
            return generated_text
    
    def _generate_safety_response(self, safety_check: Dict) -> str:
        """Generate response for safety concerns."""
        concerns = safety_check.get('concerns', [])
        if concerns:
            return f"I cannot assist with this request as it contains potentially harmful content: {', '.join(concerns[:2])}. Please rephrase your question in a safe and appropriate manner."
        return "I cannot assist with this request as it may contain harmful content. Please rephrase your question in a safe and appropriate manner."
    
    def _generate_fallback_response(self, perception_out: Dict[str, Any], decision: Dict[str, Any]) -> str:
        """Generate fallback response when all else fails."""
        input_text = ""
        if perception_out.get("text_tokens"):
            try:
                input_text = str(perception_out["text_tokens"])
            except:
                input_text = "your message"
        
        if "hello" in input_text.lower():
            return "Hello! I'm here to help, though I'm running in fallback mode due to a technical issue."
        elif "what" in input_text.lower() or "?" in input_text:
            return "I understand you have a question. I'm currently experiencing technical difficulties, so I can't provide a detailed response right now."
        elif "mode" in input_text.lower():
            return f"I'm currently in {decision.get('mode', 'default')} mode, but I'm running in fallback mode due to technical issues."
        else:
            return f"I received your message: '{input_text[:100]}...' I'm currently experiencing technical difficulties and running in fallback mode."


