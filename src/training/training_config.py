"""
Training Configuration for Tantra Conversational Speech Model
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import torch
import os


@dataclass
class TrainingConfig:
    """Configuration for training Tantra conversational speech model"""
    
    # Model Architecture
    model_name: str = "tantra_conversational_v1.0"
    base_model_path: str = "Model/weights/Tantra_v1.0.pt"
    
    # Training Parameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 10
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Conversational Training
    conversation_max_length: int = 2048
    conversation_context_window: int = 512
    conversation_temperature: float = 0.7
    conversation_top_p: float = 0.9
    conversation_top_k: int = 50
    
    # Speech Training
    speech_sample_rate: int = 16000
    speech_max_duration: float = 30.0  # seconds
    speech_hop_length: int = 512
    speech_n_fft: int = 2048
    speech_n_mels: int = 80
    speech_f_min: float = 0.0
    speech_f_max: float = 8000.0
    
    # Data Paths
    conversation_data_path: str = "data/raw/conversations"
    speech_data_path: str = "data/raw/speech"
    processed_data_path: str = "data/processed"
    
    # Output Paths
    model_output_path: str = "Model/weights"
    checkpoint_path: str = "Model/checkpoints"
    logs_path: str = "logs"
    
    # GitHub Integration
    github_repo: str = "tantra-ai/tantra-models"
    github_token: Optional[str] = None
    auto_commit: bool = True
    commit_message: str = "Update Tantra conversational speech model"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Conversational Specific
    conversation_types: List[str] = None
    personality_traits: List[str] = None
    response_style: str = "helpful"
    
    def __post_init__(self):
        """Initialize default values after dataclass creation"""
        if self.conversation_types is None:
            self.conversation_types = [
                "general_chat",
                "technical_support", 
                "creative_writing",
                "problem_solving",
                "emotional_support"
            ]
        
        if self.personality_traits is None:
            self.personality_traits = [
                "helpful",
                "knowledgeable", 
                "empathetic",
                "creative",
                "analytical"
            ]
        
        # Create directories if they don't exist
        os.makedirs(self.conversation_data_path, exist_ok=True)
        os.makedirs(self.speech_data_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)
        os.makedirs(self.model_output_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'model_name': self.model_name,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'conversation_max_length': self.conversation_max_length,
            'speech_sample_rate': self.speech_sample_rate,
            'device': self.device,
            'conversation_types': self.conversation_types,
            'personality_traits': self.personality_traits
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def save_config(self, path: str):
        """Save configuration to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_config(cls, path: str) -> 'TrainingConfig':
        """Load configuration from file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)