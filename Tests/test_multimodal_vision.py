import os
import sys
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Tantra.config import VocabConfig, NeuroCoreConfig
from Tantra.tokenizer import ImageTokenizer
from Tantra.model import NeuroCoreModel


def test_multimodal_vision_encoding():
    vcfg = VocabConfig()
    vcfg.vocab_size = 32768
    img_tok = ImageTokenizer(vcfg)

    x = torch.linspace(0, 1, 64).repeat(64, 1)
    y = torch.linspace(0, 1, 64).unsqueeze(1).repeat(1, 64)
    test_img = torch.stack([x, y, (x + y) / 2.0]).unsqueeze(0)

    with torch.no_grad():
        visual_token_ids = img_tok.encode(test_img)

    assert visual_token_ids.ndim == 2
    assert visual_token_ids.shape[0] == 1
    assert visual_token_ids.shape[1] > 0

    cfg = NeuroCoreConfig.small()
    cfg.vocab.vocab_size = 32768
    model = NeuroCoreModel(cfg, use_mtp=False)
    model.eval()

    global_visual_ids = (visual_token_ids + vcfg.image_range_start).clamp(0, cfg.vocab.vocab_size - 1)
    with torch.no_grad():
        logits, _ = model.forward(global_visual_ids)

    assert logits.shape == (1, visual_token_ids.shape[1], cfg.vocab.vocab_size)
