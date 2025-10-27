"""
Knowledge Base for ROPE RAG
Document storage and retrieval system
"""

import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pickle
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document representation for knowledge base"""
    
    id: str
    content: str
    metadata: Dict[str, Any] = None
    embedding: Optional[np.ndarray] = None
    similarity_score: float = 0.0
    
    def __post_init__(self):
        """Initialize default values"""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'similarity_score': self.similarity_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create from dictionary"""
        doc = cls(
            id=data['id'],
            content=data['content'],
            metadata=data.get('metadata', {}),
            similarity_score=data.get('similarity_score', 0.0)
        )
        
        if data.get('embedding'):
            doc.embedding = np.array(data['embedding'])
        
        return doc


class KnowledgeBase:
    """Knowledge base for document storage and retrieval"""
    
    def __init__(self, config):
        self.config = config
        self.documents = []
        self.embeddings = None
        self.embedding_model = None
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        logger.info("KnowledgeBase initialized")
    
    def _initialize_embedding_model(self):
        """Initialize embedding model"""
        try:
            # Use a simple embedding model for demonstration
            # In practice, you'd use sentence-transformers or similar
            self.embedding_dim = self.config.embedding_dim
            logger.info(f"Embedding model initialized with dimension {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_dim = 768  # Default dimension
    
    def add_documents(self, documents: List[Document]):
        """Add documents to knowledge base"""
        for doc in documents:
            # Generate embedding if not present
            if doc.embedding is None:
                doc.embedding = self._generate_embedding(doc.content)
            
            self.documents.append(doc)
        
        # Update embeddings matrix
        self._update_embeddings_matrix()
        
        logger.info(f"Added {len(documents)} documents to knowledge base")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               threshold: float = 0.0) -> List[Document]:
        """Search for similar documents"""
        if not self.documents or self.embeddings is None:
            return []
        
        try:
            # Calculate similarities
            similarities = np.dot(self.embeddings, query_embedding)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Filter by threshold and create results
            results = []
            for idx in top_indices:
                if similarities[idx] >= threshold:
                    doc = self.documents[idx]
                    doc.similarity_score = float(similarities[idx])
                    results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        # Simple hash-based embedding for demonstration
        # In practice, you'd use a proper embedding model
        
        # Create a simple embedding based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to embedding
        embedding = np.zeros(self.embedding_dim)
        for i, char in enumerate(text_hash[:self.embedding_dim]):
            embedding[i] = ord(char) / 255.0  # Normalize to 0-1
        
        # Fill remaining dimensions with text statistics
        if len(text_hash) < self.embedding_dim:
            text_stats = [
                len(text) / 1000.0,  # Normalized length
                text.count(' ') / max(len(text), 1),  # Word density
                text.count('.') / max(len(text), 1),  # Sentence density
                text.count(',') / max(len(text), 1),  # Comma density
            ]
            
            for i, stat in enumerate(text_stats):
                if len(text_hash) + i < self.embedding_dim:
                    embedding[len(text_hash) + i] = stat
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _update_embeddings_matrix(self):
        """Update embeddings matrix"""
        if not self.documents:
            self.embeddings = None
            return
        
        # Create embeddings matrix
        embeddings_list = []
        for doc in self.documents:
            if doc.embedding is not None:
                embeddings_list.append(doc.embedding)
            else:
                # Generate embedding if missing
                doc.embedding = self._generate_embedding(doc.content)
                embeddings_list.append(doc.embedding)
        
        self.embeddings = np.array(embeddings_list)
        logger.info(f"Updated embeddings matrix: {self.embeddings.shape}")
    
    def save(self, path: str):
        """Save knowledge base to disk"""
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save documents
            documents_data = [doc.to_dict() for doc in self.documents]
            with open(save_path / "documents.json", 'w') as f:
                json.dump(documents_data, f, indent=2)
            
            # Save embeddings
            if self.embeddings is not None:
                np.save(save_path / "embeddings.npy", self.embeddings)
            
            # Save metadata
            metadata = {
                'num_documents': len(self.documents),
                'embedding_dim': self.embedding_dim,
                'config': self.config.__dict__
            }
            with open(save_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Knowledge base saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
            raise
    
    def load(self, path: str):
        """Load knowledge base from disk"""
        try:
            load_path = Path(path)
            
            # Load documents
            with open(load_path / "documents.json", 'r') as f:
                documents_data = json.load(f)
            
            self.documents = [Document.from_dict(doc_data) for doc_data in documents_data]
            
            # Load embeddings
            embeddings_file = load_path / "embeddings.npy"
            if embeddings_file.exists():
                self.embeddings = np.load(embeddings_file)
            else:
                # Regenerate embeddings if missing
                self._update_embeddings_matrix()
            
            logger.info(f"Knowledge base loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            'num_documents': len(self.documents),
            'embedding_dim': self.embedding_dim,
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None,
            'average_content_length': np.mean([len(doc.content) for doc in self.documents]) if self.documents else 0
        }