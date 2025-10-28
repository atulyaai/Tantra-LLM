from __future__ import annotations

from typing import Dict, List, Optional

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
    ) -> Dict[str, torch.Tensor]:
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


