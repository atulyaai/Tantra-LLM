"""
tantra/tokenjuice.py — TokenJuice Data Density Squeezing & Enrichment Engine for Tantra-LLM.
Squeezes low-entropy noise, injects synthetic tokens, and applies dynamic loss weighting.
"""
from __future__ import annotations

import math
import random
import torch
from typing import List, Tuple, Dict, Any

from Tantra.utils import get_logger

log = get_logger("tantra.tokenjuice")


class TokenJuiceEngine:
    """
    High-signal dataset squeezing, synthetic token enrichment,
    and dynamic loss weighting engine for Tantra-LLM.
    """

    def __init__(self, entropy_threshold: float = 0.5, enrichment_rate: float = 0.1):
        self.entropy_threshold = entropy_threshold
        self.enrichment_rate = enrichment_rate
        self.synthetic_pool: List[Tuple[List[int], List[int]]] = []
        log.info(f"Initialized TokenJuice Engine (threshold={entropy_threshold}, enrichment_rate={enrichment_rate})")

    def register_synthetic_pair(self, input_ids: List[int], target_ids: List[int]) -> None:
        """Register high-signal synthetic QA / logic pairs for dynamic batch enrichment."""
        self.synthetic_pool.append((input_ids, target_ids))

    def compute_token_entropy(self, token_ids: List[int], vocab_size: int = 32000) -> float:
        """Compute Shannon entropy of a sequence of token IDs."""
        if not token_ids:
            return 0.0
        counts: Dict[int, int] = {}
        for tok in token_ids:
            counts[tok] = counts.get(tok, 0) + 1
        n = len(token_ids)
        entropy = 0.0
        for count in counts.values():
            p = count / n
            entropy -= p * math.log2(p)
        max_entropy = math.log2(vocab_size) if vocab_size > 1 else 1.0
        return entropy / max_entropy  # Normalized entropy in [0, 1]

    def squeeze_tokens(self, token_ids: List[int], vocab_size: int = 32000) -> List[int]:
        """Filter low-entropy token sequences to squeeze data density."""
        if len(token_ids) < 4:
            return token_ids
        
        # Calculate sliding entropy
        chunk_size = 8
        squeezed = []
        for i in range(0, len(token_ids), chunk_size):
            chunk = token_ids[i : i + chunk_size]
            entropy = self.compute_token_entropy(chunk, vocab_size)
            if entropy >= self.entropy_threshold or i == 0:
                squeezed.extend(chunk)
        return squeezed if squeezed else token_ids

    def enrich_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inject high-signal synthetic token pairs into training batch at enrichment_rate."""
        if not self.synthetic_pool or random.random() > self.enrichment_rate:
            return x, y
        
        # Sample synthetic pair
        synth_input, synth_target = random.choice(self.synthetic_pool)
        seq_len = x.shape[1] if x.dim() >= 2 else x.shape[0]
        
        # Truncate or pad synthetic pair
        if len(synth_input) > seq_len:
            synth_input = synth_input[:seq_len]
            synth_target = synth_target[:seq_len]
        else:
            pad_len = seq_len - len(synth_input)
            synth_input = synth_input + [0] * pad_len
            synth_target = synth_target + [0] * pad_len
            
        synth_x_tensor = torch.tensor([synth_input], device=x.device, dtype=x.dtype)
        synth_y_tensor = torch.tensor([synth_target], device=y.device, dtype=y.dtype)
        
        # Overwrite last sample in batch with high-signal synthetic pair
        x[-1:] = synth_x_tensor
        y[-1:] = synth_y_tensor
        return x, y

    def compute_dynamic_loss_weights(self, targets: torch.Tensor, high_priority_ids: List[int]) -> torch.Tensor:
        """Compute loss weight multiplier tensor giving higher weight to high-priority identity/logic tokens."""
        weights = torch.ones_like(targets, dtype=torch.float32)
        if not high_priority_ids:
            return weights
        mask = torch.isin(targets, torch.tensor(high_priority_ids, device=targets.device))
        weights[mask] = 2.5  # 2.5x gradient weight multiplier for high-signal tokens
        return weights
