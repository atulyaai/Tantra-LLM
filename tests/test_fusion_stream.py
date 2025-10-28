"""Tests for v0.6-fusion_wiring: gates placement and embed shapes."""

import torch
from encoders.text import TextTokenizer
from core.fusion.sensory_projectors import VisionProjector, AudioProjector
from core.fusion.orchestrator import FusionOrchestrator, IMG_START, IMG_END, AUD_START, AUD_END


def test_fusion_stream_with_vision_audio():
    tok = TextTokenizer()
    model_dim = 4096
    vp = VisionProjector(vision_dim=1024, model_dim=model_dim)
    ap = AudioProjector(audio_dim=1024, model_dim=model_dim)
    fusion = FusionOrchestrator(tok, vp, ap, model_dim)

    text_ids = tok.encode("hello", add_special_tokens=True)
    v = torch.zeros(1, 1024)
    a = torch.zeros(1, 1024)
    out = fusion.build_stream(text_ids, vision_embeds=v, audio_embeds=a)

    input_ids = out["input_ids"].tolist()
    vocab = tok.get_vocab()
    assert vocab.get(IMG_START) in input_ids
    assert vocab.get(IMG_END) in input_ids
    assert vocab.get(AUD_START) in input_ids
    assert vocab.get(AUD_END) in input_ids

    embeds = out["modality_embeds"]
    assert len(embeds) >= 2
    for e in embeds:
        assert e.shape[-1] == model_dim
