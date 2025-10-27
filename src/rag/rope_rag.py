"""
ROPE RAG (Retrieval-Augmented Generation) System
Advanced knowledge retrieval and generation with ROPE (Retrieval-Optimized Prompt Engineering)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import time
from pathlib import Path

from .knowledge_base import KnowledgeBase, Document
from .retriever import RAGRetriever
from .generator import RAGGenerator
from .evaluator import RAGEvaluator

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for ROPE RAG system"""
    
    # Model settings
    model_name: str = "tantra_rope_rag"
    embedding_dim: int = 768
    max_context_length: int = 4096
    retrieval_top_k: int = 5
    generation_max_length: int = 512
    
    # ROPE settings
    rope_alpha: float = 0.1  # ROPE scaling factor
    rope_theta: float = 10000.0  # ROPE base frequency
    rope_scaling: str = "linear"  # linear, dynamic
    
    # Retrieval settings
    retrieval_method: str = "dense"  # dense, sparse, hybrid
    similarity_threshold: float = 0.7
    rerank_top_k: int = 10
    
    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Knowledge base settings
    knowledge_base_path: str = "knowledge_base"
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Training settings
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100


class ROPERAG:
    """ROPE RAG system with advanced knowledge retrieval"""
    
    def __init__(self, config: RAGConfig, base_model=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.knowledge_base = KnowledgeBase(config)
        self.retriever = RAGRetriever(config)
        self.generator = RAGGenerator(config, base_model)
        self.evaluator = RAGEvaluator(config)
        
        # ROPE parameters
        self.rope_alpha = config.rope_alpha
        self.rope_theta = config.rope_theta
        
        logger.info(f"ROPERAG initialized with device: {self.device}")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to knowledge base"""
        self.knowledge_base.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to knowledge base")
    
    def add_web_content(self, web_content: List[Dict[str, Any]]):
        """Add web scraped content to knowledge base"""
        documents = []
        
        for content in web_content:
            if content.get('success', False) and content.get('content'):
                doc = Document(
                    id=content.get('url', ''),
                    content=content['content'],
                    metadata={
                        'url': content.get('url', ''),
                        'title': content.get('metadata', {}).get('title', ''),
                        'source': 'web_scraping',
                        'scraped_at': content.get('scraped_at', time.time())
                    }
                )
                documents.append(doc)
        
        self.add_documents(documents)
    
    def retrieve_and_generate(self, query: str, context: str = "") -> Dict[str, Any]:
        """Retrieve relevant documents and generate response"""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retrieve(query, top_k=self.config.retrieval_top_k)
            
            if not retrieved_docs:
                return {
                    'response': "I couldn't find relevant information to answer your question.",
                    'retrieved_documents': [],
                    'confidence': 0.0,
                    'generation_time': 0.0
                }
            
            # Apply ROPE to retrieved documents
            rope_docs = self._apply_rope(retrieved_docs, query)
            
            # Generate response
            start_time = time.time()
            response = self.generator.generate(
                query=query,
                context=context,
                retrieved_documents=rope_docs
            )
            generation_time = time.time() - start_time
            
            # Calculate confidence
            confidence = self._calculate_confidence(query, retrieved_docs, response)
            
            return {
                'response': response,
                'retrieved_documents': [
                    {
                        'content': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        'metadata': doc.metadata,
                        'similarity_score': doc.similarity_score
                    }
                    for doc in retrieved_docs
                ],
                'confidence': confidence,
                'generation_time': generation_time,
                'rope_applied': True
            }
            
        except Exception as e:
            logger.error(f"Retrieve and generate failed: {e}")
            return {
                'response': f"Error generating response: {str(e)}",
                'retrieved_documents': [],
                'confidence': 0.0,
                'generation_time': 0.0
            }
    
    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """Retrieve relevant documents for query"""
        top_k = top_k or self.config.retrieval_top_k
        
        try:
            # Get query embedding
            query_embedding = self.retriever.encode_query(query)
            
            # Retrieve documents
            retrieved_docs = self.knowledge_base.search(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=self.config.similarity_threshold
            )
            
            # Rerank if needed
            if len(retrieved_docs) > self.config.rerank_top_k:
                retrieved_docs = self.retriever.rerank(query, retrieved_docs, self.config.rerank_top_k)
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def _apply_rope(self, documents: List[Document], query: str) -> List[Document]:
        """Apply ROPE (Retrieval-Optimized Prompt Engineering) to documents"""
        rope_docs = []
        
        for doc in documents:
            # Create ROPE-enhanced content
            rope_content = self._create_rope_prompt(query, doc.content, doc.metadata)
            
            # Create new document with ROPE content
            rope_doc = Document(
                id=doc.id,
                content=rope_content,
                metadata=doc.metadata.copy(),
                similarity_score=doc.similarity_score
            )
            rope_doc.metadata['rope_applied'] = True
            rope_doc.metadata['rope_alpha'] = self.rope_alpha
            
            rope_docs.append(rope_doc)
        
        return rope_docs
    
    def _create_rope_prompt(self, query: str, content: str, metadata: Dict[str, Any]) -> str:
        """Create ROPE-enhanced prompt"""
        
        # Extract key information from metadata
        source = metadata.get('source', 'unknown')
        title = metadata.get('title', '')
        url = metadata.get('url', '')
        
        # Create ROPE prompt structure
        rope_prompt = f"""Context Information:
Source: {source}
Title: {title}
URL: {url}

Relevant Content:
{content}

Query: {query}

Instructions:
Based on the context information above, provide a comprehensive answer to the query. 
Focus on the most relevant parts of the content and maintain accuracy.
"""
        
        return rope_prompt
    
    def _calculate_confidence(self, query: str, documents: List[Document], response: str) -> float:
        """Calculate confidence score for the response"""
        try:
            # Base confidence on document similarity scores
            if not documents:
                return 0.0
            
            # Average similarity score
            avg_similarity = np.mean([doc.similarity_score for doc in documents])
            
            # Adjust based on number of documents
            doc_count_factor = min(len(documents) / self.config.retrieval_top_k, 1.0)
            
            # Calculate final confidence
            confidence = avg_similarity * doc_count_factor
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Train the RAG system on provided data"""
        try:
            logger.info(f"Starting ROPE RAG training with {len(training_data)} examples")
            
            # Prepare training data
            train_docs = []
            for item in training_data:
                doc = Document(
                    id=item.get('id', ''),
                    content=item.get('content', ''),
                    metadata=item.get('metadata', {})
                )
                train_docs.append(doc)
            
            # Add to knowledge base
            self.add_documents(train_docs)
            
            # Train retriever
            self.retriever.train(training_data)
            
            # Train generator
            self.generator.train(training_data)
            
            logger.info("ROPE RAG training completed")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate the RAG system performance"""
        try:
            logger.info(f"Evaluating ROPE RAG with {len(test_data)} test examples")
            
            metrics = self.evaluator.evaluate(self, test_data)
            
            logger.info(f"Evaluation completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}
    
    def save(self, path: str):
        """Save RAG system to disk"""
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save knowledge base
            self.knowledge_base.save(save_path / "knowledge_base")
            
            # Save retriever
            self.retriever.save(save_path / "retriever")
            
            # Save generator
            self.generator.save(save_path / "generator")
            
            # Save config
            with open(save_path / "config.json", 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
            
            logger.info(f"ROPE RAG system saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save RAG system: {e}")
            raise
    
    def load(self, path: str):
        """Load RAG system from disk"""
        try:
            load_path = Path(path)
            
            # Load config
            with open(load_path / "config.json", 'r') as f:
                config_data = json.load(f)
            
            # Load knowledge base
            self.knowledge_base.load(load_path / "knowledge_base")
            
            # Load retriever
            self.retriever.load(load_path / "retriever")
            
            # Load generator
            self.generator.load(load_path / "generator")
            
            logger.info(f"ROPE RAG system loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load RAG system: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'knowledge_base_size': len(self.knowledge_base.documents),
            'embedding_dim': self.config.embedding_dim,
            'retrieval_top_k': self.config.retrieval_top_k,
            'rope_alpha': self.rope_alpha,
            'rope_theta': self.rope_theta,
            'device': str(self.device)
        }