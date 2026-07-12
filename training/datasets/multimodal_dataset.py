from __future__ import annotations

"""Multimodal dataset for pre-computed embeddings.

Supports two modes:
1. In-memory: pass pre-computed embedding tensors directly
2. Cached-to-disk: load .pt files containing pre-computed embeddings (for large datasets)
"""

import os
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional


class MultimodalDataset(Dataset):
    """
    Dataset of pre-computed vision/audio embeddings paired with target token IDs.
    
    Each sample is a dict with keys:
        - vision_embeds: torch.Tensor [vision_dim]
        - audio_embeds: torch.Tensor [audio_dim]  (optional)
        - target_ids: torch.Tensor [seq_len]
    """

    def __init__(
        self, 
        samples: Optional[List[Dict[str, torch.Tensor]]] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            samples: List of dicts with pre-computed tensors (in-memory mode)
            cache_dir: Path to directory of .pt files (cached-to-disk mode)
        """
        self.samples = samples or []
        self.cache_dir = cache_dir
        self._cached_files: List[str] = []
        
        if cache_dir and os.path.isdir(cache_dir):
            self._cached_files = sorted([
                os.path.join(cache_dir, f) 
                for f in os.listdir(cache_dir) 
                if f.endswith(".pt")
            ])

    def __len__(self) -> int:
        if self._cached_files:
            return len(self._cached_files)
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._cached_files:
            return torch.load(self._cached_files[idx], weights_only=True)
        return self.samples[idx]

    @staticmethod
    def generate_synthetic(
        num_samples: int = 500,
        vision_dim: int = 768,
        audio_dim: int = 512,
        target_dim: int = 4096,
        seq_len: int = 32
    ) -> 'MultimodalDataset':
        """
        Generate a synthetic dataset of random embeddings for pipeline testing.
        Simulates pre-computed encoder outputs.
        """
        samples = []
        for _ in range(num_samples):
            sample = {
                "vision_embeds": torch.randn(vision_dim),
                "audio_embeds": torch.randn(audio_dim),
                "target_ids": torch.randint(0, target_dim, (seq_len,)),
            }
            samples.append(sample)
        return MultimodalDataset(samples=samples)

    def save_to_cache(self, cache_dir: str):
        """Save all in-memory samples to disk as .pt files for future fast loading."""
        os.makedirs(cache_dir, exist_ok=True)
        for i, sample in enumerate(self.samples):
            torch.save(sample, os.path.join(cache_dir, f"sample_{i:06d}.pt"))
        print(f"[Dataset] Saved {len(self.samples)} samples to {cache_dir}")
