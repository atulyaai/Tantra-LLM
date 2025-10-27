"""
Conversational Interface for OCR-Native LLM
Interactive chat system with memory and context
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
import io
import base64

from src.architectures.transformer_variants import OCRNativeTransformer, TransformerVariantConfig
from src.utils.error_handler import logger


@dataclass
class ConversationMessage:
    """Single message in a conversation"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float
    message_id: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationContext:
    """Context for a conversation session"""
    session_id: str
    user_id: str
    start_time: float
    messages: List[ConversationMessage]
    model_config: Dict[str, Any]
    memory_context: Dict[str, Any]
    
    def add_message(self, message: ConversationMessage):
        """Add a message to the conversation"""
        self.messages.append(message)
    
    def get_recent_messages(self, n: int = 10) -> List[ConversationMessage]:
        """Get the most recent n messages"""
        return self.messages[-n:] if self.messages else []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'start_time': self.start_time,
            'messages': [msg.to_dict() for msg in self.messages],
            'model_config': self.model_config,
            'memory_context': self.memory_context
        }


class OCRNativeConversational:
    """Conversational interface for OCR-Native LLM"""
    
    def __init__(self, config: TransformerVariantConfig, model_variant: str = "standard"):
        self.config = config
        self.model_variant = model_variant
        self.model = self._initialize_model()
        
        # Conversation management
        self.active_sessions: Dict[str, ConversationContext] = {}
        self.conversation_history: List[ConversationContext] = []
        
        # Response generation settings
        self.max_response_length = 500
        self.temperature = 0.7
        self.top_p = 0.9
        
        logger.info(f"Initialized conversational interface with {model_variant} variant")
    
    def _initialize_model(self) -> OCRNativeTransformer:
        """Initialize the model with specified variant"""
        config = self.config
        config.variant = self.model_variant
        return OCRNativeTransformer(config)
    
    def start_conversation(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Start a new conversation session"""
        if session_id is None:
            session_id = f"session_{int(time.time())}_{user_id}"
        
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            start_time=time.time(),
            messages=[],
            model_config=self.config.__dict__,
            memory_context={}
        )
        
        self.active_sessions[session_id] = context
        
        # Add welcome message
        welcome_msg = ConversationMessage(
            role="assistant",
            content="Hello! I'm an OCR-Native LLM. I can process text, images, and audio through OCR-optimized neural networks. How can I help you today?",
            timestamp=time.time(),
            message_id=f"msg_{int(time.time())}_welcome"
        )
        context.add_message(welcome_msg)
        
        logger.info(f"Started conversation session: {session_id} for user: {user_id}")
        return session_id
    
    def send_message(self, session_id: str, content: str, 
                    message_type: str = "text", 
                    media_data: Optional[Any] = None) -> Dict[str, Any]:
        """Send a message and get response"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        context = self.active_sessions[session_id]
        
        # Create user message
        user_msg = ConversationMessage(
            role="user",
            content=content,
            timestamp=time.time(),
            message_id=f"msg_{int(time.time())}_user",
            metadata={"type": message_type, "media_data": media_data}
        )
        context.add_message(user_msg)
        
        # Process message and generate response
        response = self._generate_response(context, user_msg)
        
        # Create assistant message
        assistant_msg = ConversationMessage(
            role="assistant",
            content=response["content"],
            timestamp=time.time(),
            message_id=f"msg_{int(time.time())}_assistant",
            metadata=response.get("metadata", {})
        )
        context.add_message(assistant_msg)
        
        # Update memory context
        self._update_memory_context(context, user_msg, assistant_msg)
        
        return {
            "response": response["content"],
            "message_id": assistant_msg.message_id,
            "timestamp": assistant_msg.timestamp,
            "metadata": response.get("metadata", {}),
            "session_info": {
                "session_id": session_id,
                "message_count": len(context.messages),
                "model_variant": self.model_variant
            }
        }
    
    def _generate_response(self, context: ConversationContext, user_msg: ConversationMessage) -> Dict[str, Any]:
        """Generate response using the OCR-Native model"""
        try:
            # Prepare inputs based on message type
            inputs = self._prepare_model_inputs(user_msg, context)
            
            # Generate response using the model
            with torch.no_grad():
                outputs = self.model(inputs)
            
            # Process outputs
            response_content = self._process_model_outputs(outputs, context)
            
            # Add OCR-specific metadata
            metadata = {
                "model_variant": self.model_variant,
                "processing_time": time.time() - user_msg.timestamp,
                "ocr_processed": True,
                "memory_retrieved": len(context.memory_context) > 0
            }
            
            return {
                "content": response_content,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "content": f"I encountered an error processing your message: {str(e)}. Please try again.",
                "metadata": {"error": True, "error_message": str(e)}
            }
    
    def _prepare_model_inputs(self, user_msg: ConversationMessage, context: ConversationContext) -> Dict[str, Any]:
        """Prepare inputs for the model based on message type"""
        inputs = {}
        
        # Add text content
        if user_msg.content:
            inputs['text'] = user_msg.content
        
        # Add media data if present
        if user_msg.metadata and user_msg.metadata.get("media_data"):
            media_data = user_msg.metadata["media_data"]
            if user_msg.metadata.get("type") == "image":
                inputs['image'] = media_data
            elif user_msg.metadata.get("type") == "audio":
                inputs['speech'] = media_data
        
        # Add conversation context
        recent_messages = context.get_recent_messages(5)
        context_text = " ".join([msg.content for msg in recent_messages if msg.role == "user"])
        if context_text:
            inputs['context'] = context_text
        
        return inputs
    
    def _process_model_outputs(self, outputs: Dict[str, torch.Tensor], context: ConversationContext) -> str:
        """Process model outputs into response text"""
        # Get text logits
        text_logits = outputs.get('text_logits', torch.zeros(1, 1, self.config.vocab_size))
        
        # Simple greedy decoding (can be enhanced with sampling)
        predicted_tokens = torch.argmax(text_logits, dim=-1)
        
        # Convert to text (simplified - would use proper tokenizer in practice)
        response_parts = []
        
        # Generate response based on model variant
        if self.model_variant == "mamba":
            response_parts.append("🔤 [Mamba-OCR] I've processed your input through a Mamba-inspired OCR-native architecture.")
        elif self.model_variant == "hybrid":
            response_parts.append("🔀 [Hybrid-OCR] I've analyzed your input using a hybrid OCR-native approach.")
        elif self.model_variant == "memory_enhanced":
            response_parts.append("🧠 [Memory-OCR] I've processed your input with enhanced OCR memory capabilities.")
        else:
            response_parts.append("🔤 [OCR-Native] I've processed your input through OCR-optimized neural networks.")
        
        # Add context-aware response
        if context.messages:
            recent_user_msg = next((msg for msg in reversed(context.messages) if msg.role == "user"), None)
            if recent_user_msg:
                response_parts.append(f"Regarding '{recent_user_msg.content[:50]}...', I can help you with:")
                response_parts.append("• Text analysis and generation")
                response_parts.append("• Image processing and OCR")
                response_parts.append("• Multi-modal understanding")
                response_parts.append("• Pattern recognition and memory")
        
        # Add technical details
        response_parts.append(f"\n[Technical: {self.config.n_layers} layers, {self.config.d_model}D embeddings, {self.config.n_heads} attention heads]")
        
        return "\n".join(response_parts)
    
    def _update_memory_context(self, context: ConversationContext, user_msg: ConversationMessage, assistant_msg: ConversationMessage):
        """Update memory context with conversation information"""
        # Add to OCR memory bank
        self.model.memory_bank.store_ocr_memory(
            content=user_msg.content,
            memory_type="conversation",
            importance=0.8
        )
        
        # Update context metadata
        context.memory_context.update({
            "last_user_message": user_msg.content,
            "last_response": assistant_msg.content,
            "conversation_length": len(context.messages),
            "last_update": time.time()
        })
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        if session_id not in self.active_sessions:
            return []
        
        context = self.active_sessions[session_id]
        return [msg.to_dict() for msg in context.messages]
    
    def end_conversation(self, session_id: str) -> Dict[str, Any]:
        """End a conversation session"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        context = self.active_sessions[session_id]
        
        # Move to history
        self.conversation_history.append(context)
        del self.active_sessions[session_id]
        
        # Generate summary
        summary = {
            "session_id": session_id,
            "duration": time.time() - context.start_time,
            "message_count": len(context.messages),
            "model_variant": self.model_variant,
            "end_time": time.time()
        }
        
        logger.info(f"Ended conversation session: {session_id}")
        return summary
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        context = self.active_sessions[session_id]
        return context.to_dict()
    
    def list_active_sessions(self) -> List[str]:
        """List all active session IDs"""
        return list(self.active_sessions.keys())
    
    def switch_model_variant(self, new_variant: str) -> bool:
        """Switch to a different model variant"""
        try:
            self.model_variant = new_variant
            self.model = self._initialize_model()
            logger.info(f"Switched to model variant: {new_variant}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch model variant: {e}")
            return False
    
    def get_available_variants(self) -> List[str]:
        """Get list of available model variants"""
        return ["standard", "mamba", "hybrid", "memory_enhanced"]
    
    def export_conversation(self, session_id: str, format: str = "json") -> str:
        """Export conversation in specified format"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        context = self.active_sessions[session_id]
        
        if format == "json":
            return json.dumps(context.to_dict(), indent=2)
        elif format == "text":
            lines = [f"Conversation Session: {session_id}"]
            lines.append(f"User: {context.user_id}")
            lines.append(f"Started: {datetime.fromtimestamp(context.start_time)}")
            lines.append("-" * 50)
            
            for msg in context.messages:
                timestamp = datetime.fromtimestamp(msg.timestamp).strftime("%H:%M:%S")
                lines.append(f"[{timestamp}] {msg.role.upper()}: {msg.content}")
            
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience functions for easy usage
def create_conversational_interface(model_size: str = "small", variant: str = "standard") -> OCRNativeConversational:
    """Create a conversational interface with specified settings"""
    from src.configs.ocr_config import ConfigManager
    
    if model_size == "small":
        config = ConfigManager.get_small_config()
    elif model_size == "large":
        config = ConfigManager.get_large_config()
    else:
        config = ConfigManager.get_default_config()
    
    # Convert to transformer variant config
    variant_config = TransformerVariantConfig(**config.__dict__)
    
    return OCRNativeConversational(variant_config, variant)


def quick_chat(message: str, variant: str = "standard") -> str:
    """Quick chat function for testing"""
    interface = create_conversational_interface("small", variant)
    session_id = interface.start_conversation("test_user")
    
    response = interface.send_message(session_id, message)
    interface.end_conversation(session_id)
    
    return response["response"]