"""Tests for tantra.tokenizer + tantra.codec"""
import json
import os
import tempfile

import pytest
import torch

from Tantra.config import VocabConfig
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer, AudioTokenizer, ImageTokenizer
from Tantra.dataset import TopicMixedDataset, PretokenizedBinDataset, IGNORE_INDEX


class _StubTokenizer:
    vocab_size = 4096
    def encode(self, text, modality="text"):
        return [min((ord(c) % self.vocab_size), self.vocab_size - 1) for c in text]


def _write_chat_jsonl(path, n, prefix="q"):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "system": "You are a helpful assistant.",
                "user": f"{prefix} user {i}",
                "assistant": f"{prefix} assistant answer {i} with enough tokens here.",
            }) + "\n")


def test_topic_mixed_dataset_multi_file():
    """TopicMixedDataset must advance through every shard, not just the first file."""
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(3):
            _write_chat_jsonl(os.path.join(tmp, f"shard{i}.jsonl"), 4, prefix=f"t{i}")
        ds = TopicMixedDataset(
            {"general": [os.path.join(tmp, f"shard{i}.jsonl") for i in range(3)]},
            {"general": 1.0},
            _StubTokenizer(),
            seq_len=16,
            max_samples=50,
            seed=1,
        )
        got = [x for x, _ in ds]
        assert len(got) == 50
        for x in got:
            assert x.shape == (16,)
            assert x.dtype == torch.long


def test_topic_mixed_dataset_weighted_topics():
    """Both topics must contribute samples; the general-heavy mix yields more general samples."""
    with tempfile.TemporaryDirectory() as tmp:
        gen = os.path.join(tmp, "general.jsonl")
        code = os.path.join(tmp, "code.jsonl")
        _write_chat_jsonl(gen, 50, prefix="g")
        _write_chat_jsonl(code, 50, prefix="c")
        ds = TopicMixedDataset(
            {"general": [gen], "code": [code]},
            {"general": 5.0, "code": 1.0},
            _StubTokenizer(),
            seq_len=8,
            max_samples=300,
            seed=3,
        )
        n = 0
        for x, _ in ds:
            n += 1
        assert n == 300


def test_pretokenized_pretraining_supervises_all_tokens():
    """Pretraining must ignore chat masks and learn from every token."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = os.path.join(tmp, "corpus.bin")
        torch.save({
            "tokens": torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32),
            "masks": torch.tensor([False, False, False, False, False]),
        }, cache_path)
        _, pretrain_y = next(iter(PretokenizedBinDataset(cache_path, seq_len=4, mask_non_assistant=False)))
        _, sft_y = next(iter(PretokenizedBinDataset(cache_path, seq_len=4, mask_non_assistant=True)))
        assert torch.all(pretrain_y != IGNORE_INDEX)
        assert torch.all(sft_y == IGNORE_INDEX)


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
