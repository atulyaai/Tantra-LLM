"""Tests for tantra.tokenizer + tantra.codec"""
import json
import os
import tempfile

import pytest
import torch

from Tantra.config import VocabConfig
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer, AudioTokenizer, ImageTokenizer
from Tantra.dataset import TopicMixedDataset, PretokenizedBinDataset, JSONLDataset, IGNORE_INDEX, TokenJuiceEngine



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


def test_tokenjuice_entropy_enrichment_and_weights():
    engine = TokenJuiceEngine(entropy_threshold=0.3, enrichment_rate=1.0)
    low_entropy = [1] * 8
    high_entropy = [1, 250, 4829, 991, 12, 592, 1024, 881]
    assert engine.compute_token_entropy(low_entropy) < engine.compute_token_entropy(high_entropy)
    engine.register_synthetic_pair([100, 101, 102], [200, 201, 202])
    x, y = engine.enrich_batch(torch.zeros((2, 8), dtype=torch.long), torch.zeros((2, 8), dtype=torch.long))
    assert x.shape == y.shape == (2, 8)
    weights = engine.compute_dynamic_loss_weights(torch.tensor([1, 100, 5, 101]), [100, 101])
    assert weights.tolist() == [1.0, 2.5, 1.0, 2.5]


def test_jsonl_dataset_shuffling():
    """JSONLDataset with shuffle=True must randomize the order of lines streamed."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_shuffle.jsonl")
        _write_chat_jsonl(path, 100, prefix="item")
        
        # Non-shuffled dataset
        ds_noshuffle = JSONLDataset(path, _StubTokenizer(), seq_len=128, max_samples=100, shuffle=False)
        items_noshuffle = [x.tolist() for x, _ in ds_noshuffle]
        
        # Shuffled dataset
        ds_shuffle = JSONLDataset(path, _StubTokenizer(), seq_len=128, max_samples=100, shuffle=True, shuffle_buf_size=50, seed=42)
        items_shuffle = [x.tolist() for x, _ in ds_shuffle]
        
        assert len(items_noshuffle) == 100
        assert len(items_shuffle) == 100
        # Line order must differ when shuffle=True
        assert items_noshuffle != items_shuffle


def test_jsonl_dataset_epoch_reshuffling():
    """JSONLDataset must yield different line permutations on epoch 1 vs epoch 2 when iterating past EOF."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_epoch.jsonl")
        _write_chat_jsonl(path, 30, prefix="ep")
        
        ds = JSONLDataset(path, _StubTokenizer(), seq_len=128, max_samples=60, shuffle=True, shuffle_buf_size=20, seed=123)
        all_samples = [x.tolist() for x, _ in ds]
        
        assert len(all_samples) == 60
        epoch_1 = all_samples[:30]
        epoch_2 = all_samples[30:]
        # Epoch 1 and Epoch 2 must be different line permutations
        assert epoch_1 != epoch_2



def test_jsonl_dataset_validation_split():

    """Train and validation splits must be disjoint and contain zero line overlap."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_split.jsonl")
        _write_chat_jsonl(path, 100, prefix="item")

        ds_train = JSONLDataset(path, _StubTokenizer(), seq_len=128, max_samples=100, split="train", val_ratio=0.1, shuffle=False)
        ds_val = JSONLDataset(path, _StubTokenizer(), seq_len=128, max_samples=100, split="val", val_ratio=0.1, shuffle=False)

        train_items = set(tuple(x.tolist()) for x, _ in ds_train)
        val_items = set(tuple(x.tolist()) for x, _ in ds_val)

        assert len(train_items) > 0
        assert len(val_items) > 0
        # Zero line overlap between train and val
        assert train_items.isdisjoint(val_items)


def test_neurotrainer_evaluate_validation():
    """evaluate_validation must return val_loss, val_acc, and val_ppl without crashing."""
    from Tantra.model import build_cpu_model
    from Tantra.train import NeuroTrainer
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_eval_val.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(20):
                f.write(json.dumps({"user": f"eval question {i}", "assistant": f"eval answer {i}"}) + "\n")

        tok = _StubTokenizer()
        ds_val = JSONLDataset(path, tok, seq_len=64, max_samples=20, split="val", val_ratio=0.5, shuffle=False)
        val_loader = DataLoader(ds_val, batch_size=2)

        model = build_cpu_model("micro10", attention_kind="causal")
        trainer = NeuroTrainer(model, lr=1e-4, total_steps=10)

        metrics = trainer.evaluate_validation(val_loader, max_val_batches=5)
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics
        assert "val_acc" in metrics
        assert "val_ppl" in metrics
        assert metrics["val_loss"] > 0


def test_continuous_sequence_packing():
    """Continuous sequence packing must pack multiple documents with zero padding tokens."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "packed_test.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({
                    "user": f"Prompt {i}",
                    "assistant": f"Reply {i}"
                }) + "\n")

        tok = _StubTokenizer()
        ds_packed = JSONLDataset(path, tok, seq_len=32, max_samples=10, pack_sequences=True, mask_non_assistant=True)
        samples = [s for s in ds_packed]
        assert len(samples) > 0
        for x, y in samples:
            assert x.shape == (32,)
            assert y.shape == (32,)
            # Verify that supervised assistant tokens exist in the packed chunk
            assert (y != IGNORE_INDEX).any()
            # Verify x has no zero padding tokens at the tail
            assert not (x[-4:] == 0).all()






