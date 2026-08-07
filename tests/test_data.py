"""Tests for tantra.tokenizer + tantra.codec"""
import pytest
import torch

from tantra.config import VocabConfig
from tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer, AudioTokenizer, ImageTokenizer


def test_megabyte_patcher_roundtrip():
    p = MegabytePatcher()
    data = b"hello world test bytes"
    ids = p.encode_bytes(data)
    assert len(ids) > 0

def test_unified_tokenizer_text():
    cfg = VocabConfig()
    bpe = ByteBPETokenizer(cfg)
    patcher = MegabytePatcher()
    ut = UnifiedTokenizer(cfg, bpe, patcher)
    ids = ut.encode("hello", "text")
    assert len(ids) > 0

def test_audio_tokenizer():
    cfg = VocabConfig()
    tok = AudioTokenizer(cfg)
    waveform = torch.randn(1, 1, 16000)
    ids = tok.encode(waveform)
    assert ids.min() >= 0 and ids.max() < cfg.audio_codebook_size

def test_image_tokenizer():
    cfg = VocabConfig()
    tok = ImageTokenizer(cfg)
    image = torch.rand(1, 3, 64, 64)
    ids = tok.encode(image)
    assert ids.min() >= 0 and ids.max() < cfg.image_codebook_size
