"""
RAG Generator Module
Text generation with retrieved context
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .knowledge_base import Document

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    
    # Generation parameters
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Context handling
    max_context_length: int = 2048
    context_separator: str = "\n\n"
    
    # Model settings
    model_name: str = "tantra_generator"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RAGGenerator:
    """Text generator for RAG system"""
    
    def __init__(self, config, base_model=None):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize base model
        if base_model is not None:
            self.model = base_model
        else:
            # Create simple generator model
            self.model = self._create_simple_generator()
        
        self.model.to(self.device)
        
        logger.info("RAGGenerator initialized")
    
    def _create_simple_generator(self):
        """Create a simple generator model"""
        # Simple LSTM-based generator for demonstration
        class SimpleGenerator(nn.Module):
            def __init__(self, vocab_size=10000, embedding_dim=256, hidden_dim=512):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
                self.output = nn.Linear(hidden_dim, vocab_size)
                
            def forward(self, x):
                embedded = self.embedding(x)
                lstm_out, _ = self.lstm(embedded)
                output = self.output(lstm_out)
                return output
        
        return SimpleGenerator()
    
    def generate(self, query: str, context: str = "", 
                retrieved_documents: List[Document] = None) -> str:
        """Generate response using query and retrieved context"""
        try:
            # Prepare context from retrieved documents
            context_text = self._prepare_context(query, context, retrieved_documents)
            
            # Generate response
            response = self._generate_text(query, context_text)
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def _prepare_context(self, query: str, context: str, 
                        retrieved_documents: List[Document]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        
        # Add base context
        if context:
            context_parts.append(f"Context: {context}")
        
        # Add retrieved documents
        if retrieved_documents:
            for i, doc in enumerate(retrieved_documents):
                doc_context = f"Document {i+1}: {doc.content[:500]}..."  # Truncate long documents
                context_parts.append(doc_context)
        
        # Combine context parts
        full_context = self.config.context_separator.join(context_parts)
        
        # Truncate if too long
        if len(full_context) > self.config.max_context_length:
            full_context = full_context[:self.config.max_context_length] + "..."
        
        return full_context
    
    def _generate_text(self, query: str, context: str) -> str:
        """Generate text using the model"""
        try:
            # Simple text generation for demonstration
            # In practice, you'd use a proper language model
            
            # Create prompt
            prompt = f"Query: {query}\n\nContext: {context}\n\nResponse:"
            
            # Generate response based on query and context
            response = self._generate_simple_response(query, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return "I'm sorry, I couldn't generate a proper response."
    
    def _generate_simple_response(self, query: str, context: str) -> str:
        """Simple response generation for demonstration"""
        # This is a very basic implementation
        # In practice, you'd use a proper language model
        
        # Analyze query type
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where']):
            response_type = "explanatory"
        elif any(word in query_lower for word in ['tell me', 'describe', 'explain']):
            response_type = "descriptive"
        elif any(word in query_lower for word in ['help', 'assist', 'support']):
            response_type = "helpful"
        else:
            response_type = "general"
        
        # Generate response based on type
        if response_type == "explanatory":
            response = f"Based on the available information, {query.lower()} can be answered as follows: "
            if context:
                response += f"The context suggests that {context[:200]}..."
            else:
                response += "I don't have specific information about this topic, but I'd be happy to help you find more details."
        
        elif response_type == "descriptive":
            response = f"I'll help you understand {query.lower()}. "
            if context:
                response += f"From the information available: {context[:300]}..."
            else:
                response += "Let me provide you with a comprehensive explanation based on general knowledge."
        
        elif response_type == "helpful":
            response = f"I'm here to help with {query.lower()}. "
            if context:
                response += f"Based on the context, here's what I can suggest: {context[:250]}..."
            else:
                response += "I'd be happy to assist you further. Could you provide more specific details about what you need help with?"
        
        else:
            response = f"Regarding {query.lower()}, "
            if context:
                response += f"the information indicates that {context[:200]}..."
            else:
                response += "I'd be happy to discuss this topic with you. What specific aspect would you like to know more about?"
        
        return response
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Train the generator on provided data"""
        try:
            logger.info(f"Training generator on {len(training_data)} examples")
            
            # Simple training for demonstration
            # In practice, you'd implement proper training logic
            
            # Extract query-response pairs
            query_response_pairs = []
            for item in training_data:
                query = item.get('query', '')
                response = item.get('response', '')
                context = item.get('context', '')
                
                if query and response:
                    query_response_pairs.append({
                        'query': query,
                        'response': response,
                        'context': context
                    })
            
            # Train on query-response pairs
            # This is a simplified implementation
            logger.info(f"Generator training completed with {len(query_response_pairs)} pairs")
            
        except Exception as e:
            logger.error(f"Generator training failed: {e}")
            raise
    
    def save(self, path: str):
        """Save generator to disk"""
        try:
            import os
            os.makedirs(path, exist_ok=True)
            
            # Save model state
            torch.save(self.model.state_dict(), os.path.join(path, "generator_model.pt"))
            
            # Save configuration
            import json
            with open(os.path.join(path, "config.json"), 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
            
            logger.info(f"Generator saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save generator: {e}")
            raise
    
    def load(self, path: str):
        """Load generator from disk"""
        try:
            import os
            import json
            
            # Load model state
            model_path = os.path.join(path, "generator_model.pt")
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # Load configuration
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update configuration
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            
            logger.info(f"Generator loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load generator: {e}")
            raise