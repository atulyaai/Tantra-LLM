import os
import sys
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Tantra.config import VocabConfig, NeuroCoreConfig
from Tantra.tokenizer import AudioTokenizer, ImageTokenizer
from Tantra.model import NeuroCoreModel


def test_omnimodal_assembly():
    vcfg = VocabConfig()
    vcfg.vocab_size = 32768
    audio_tok = AudioTokenizer(vcfg)
    img_tok = ImageTokenizer(vcfg)

    raw_audio = torch.randn(1, 1, 16000)
    with torch.no_grad():
        audio_tokens = audio_tok.encode(raw_audio)

    raw_image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        image_tokens = img_tok.encode(raw_image)

    text_ids = torch.randint(0, 1000, (1, 8))

    audio_ids = (audio_tokens + vcfg.audio_range_start).clamp(0, vcfg.vocab_size - 1)
    image_ids = (image_tokens + vcfg.image_range_start).clamp(0, vcfg.vocab_size - 1)
    omnimodal_input = torch.cat([audio_ids, image_ids, text_ids], dim=1)

    cfg = NeuroCoreConfig.small()
    cfg.vocab.vocab_size = 32768
    model = NeuroCoreModel(cfg, use_mtp=False)
    model.eval()

    with torch.no_grad():
        logits, _ = model.forward(omnimodal_input)

    assert logits.shape == (1, omnimodal_input.shape[1], 32768)
