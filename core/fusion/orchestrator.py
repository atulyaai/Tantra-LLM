from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


IMG_START = "<IMG_START>"
IMG_END = "<IMG_END>"
AUD_START = "<AUD_START>"
AUD_END = "<AUD_END>"


class FusionOrchestrator:
    """Stub: construct unified stream with modality gate tokens."""

    def __init__(self, tokenizer, vision_projector, audio_projector, model_dim: int):
        self.tokenizer = tokenizer
        self.vision_projector = vision_projector
        self.audio_projector = audio_projector
        self.model_dim = model_dim

        for tok in (IMG_START, IMG_END, AUD_START, AUD_END):
            if tok not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens([tok])

    def build_stream(
        self,
        text_tokens: List[int],
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        input_ids = list(text_tokens)
        modality_embeds: List[torch.Tensor] = []

        if vision_embeds is not None:
            projected = self.vision_projector(vision_embeds)
            input_ids += [self.tokenizer.convert_tokens_to_ids(IMG_START)]
            modality_embeds.append(projected)
            input_ids += [self.tokenizer.convert_tokens_to_ids(IMG_END)]

        if audio_embeds is not None:
            projected = self.audio_projector(audio_embeds)
            input_ids += [self.tokenizer.convert_tokens_to_ids(AUD_START)]
            modality_embeds.append(projected)
            input_ids += [self.tokenizer.convert_tokens_to_ids(AUD_END)]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "modality_embeds": modality_embeds,
        }
    
    def merge_embeddings(
        self,
        input_ids: torch.Tensor,
        modality_embeds: List[torch.Tensor],
        model: Any,
    ) -> torch.Tensor:
        """Merge text and modality embeddings for model forward pass.
        
        Args:
            input_ids: Token IDs [seq_len]
            modality_embeds: List of projected modality embeddings
            model: Transformer model with get_input_embeddings()
            
        Returns:
            Merged embeddings tensor [seq_len, model_dim]
        """
        emb_layer = model.get_input_embeddings()
        text_embeds = emb_layer(input_ids)  # [seq_len, model_dim]
        
        # Find gate token positions
        img_start_id = self.tokenizer.convert_tokens_to_ids(IMG_START)
        aud_start_id = self.tokenizer.convert_tokens_to_ids(AUD_START)
        
        merged = text_embeds.clone()
        emb_idx = 0
        
        for i, token_id in enumerate(input_ids):
            if token_id.item() == img_start_id and emb_idx < len(modality_embeds):
                # Replace IMG_START embedding with projected vision
                merged[i] = modality_embeds[emb_idx]
                emb_idx += 1
            elif token_id.item() == aud_start_id and emb_idx < len(modality_embeds):
                # Replace AUD_START embedding with projected audio
                merged[i] = modality_embeds[emb_idx]
                emb_idx += 1
        
        return merged


