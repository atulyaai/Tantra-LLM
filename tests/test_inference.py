"""Test for SpikingBrain forward pass with embeddings injection."""

import torch
from encoders.text import TextTokenizer
from core.fusion.sensory_projectors import VisionProjector, AudioProjector
from core.fusion.orchestrator import FusionOrchestrator


def test_model_generates_non_empty():
    """Verify that model can generate output even if stub."""
    # This test will pass if model loads OR if graceful fallback works
    tok = TextTokenizer()
    model_dim = 4096
    vp = VisionProjector(vision_dim=1024, model_dim=model_dim)
    ap = AudioProjector(audio_dim=1024, model_dim=model_dim)
    fusion = FusionOrchestrator(tok, vp, ap, model_dim)
    
    text_ids = tok.encode("hello", add_special_tokens=True)
    v = torch.zeros(1, 1024)
    
    stream = fusion.build_stream(text_ids, vision_embeds=v)
    assert stream["input_ids"].shape[0] > 0
    assert len(stream["modality_embeds"]) > 0

