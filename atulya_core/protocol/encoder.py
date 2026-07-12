import torch
from typing import Protocol, Any

class ModalityEncoder(Protocol):
    """Protocol enforcing shape and dimension constraints on all sensory modality encoders."""
    
    @property
    def embed_dim(self) -> int:
        """The target embedding dimension required by the base model projection."""
        ...

    def encode(self, data: Any) -> torch.Tensor:
        """
        Processes sensory input data and returns a normalized embedding tensor.
        
        Returns:
            A torch.Tensor of shape [1, embed_dim].
        """
        ...
