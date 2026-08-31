"""
tests/test_multimodal_weights.py — Tests for Multimodal Weight Space & Unified Encryption Formatter.
"""

import os
import pytest
import torch

from Tantra.config import CompressionConfig, NeuroCoreConfig, VocabConfig
from Tantra.codec import MultimodalWeightFormatter, ZSTDDictTrainer
from Tantra.tokenizer import UnifiedTokenizer, ByteBPETokenizer, MegabytePatcher
from Tantra.model import NeuroCoreModel


@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path)


def test_multimodal_weight_formatter_pack_unpack(temp_dir):
    config = CompressionConfig()
    formatter = MultimodalWeightFormatter(config, secret_key=b"SECRET_KEY_12345678901234567890!")

    weights = {
        "text": torch.randn(100, 64, dtype=torch.float32),
        "audio": torch.randn(50, 64, dtype=torch.float32),
        "image": torch.randn(50, 64, dtype=torch.float32),
        "video": torch.randn(20, 64, dtype=torch.float32),
    }

    output_path = os.path.join(temp_dir, "multimodal_weights.dna")
    stats = formatter.format_weights(weights, output_path)

    assert os.path.exists(output_path)
    assert stats.sha256_match is True
    assert stats.method == "multimodal_dna_encrypted"

    restored_weights = formatter.parse_weights(output_path)

    for key in weights:
        assert key in restored_weights
        assert restored_weights[key].shape == weights[key].shape
        assert restored_weights[key].dtype == weights[key].dtype
        assert torch.allclose(restored_weights[key], weights[key], atol=1e-5)


def test_multimodal_weight_formatter_with_zstd_dict(temp_dir):
    config = CompressionConfig()
    trainer = ZSTDDictTrainer(config)

    sample_tensors = [torch.randn(64, 16) for _ in range(3)]
    dict_path = os.path.join(temp_dir, "test_dict.zstd")
    dict_data = trainer.train_from_tensors(sample_tensors, dict_path)

    formatter = MultimodalWeightFormatter(config)
    weights = {
        "text": torch.randn(32, 16),
        "audio": torch.randn(16, 16),
        "image": torch.randn(16, 16),
        "video": torch.randn(8, 16),
    }

    output_path = os.path.join(temp_dir, "dict_weights.dna")
    stats = formatter.format_weights(weights, output_path, dict_data=dict_data)
    assert stats.sha256_match is True

    restored = formatter.parse_weights(output_path)
    for key in weights:
        assert torch.allclose(restored[key], weights[key], atol=1e-5)


def test_unified_tokenizer_weight_sharing(temp_dir):
    vocab_cfg = VocabConfig()
    bpe = ByteBPETokenizer(vocab_cfg)
    patcher = MegabytePatcher()
    tokenizer = UnifiedTokenizer(vocab_cfg, bpe, patcher)

    modal_weights = {
        "text": torch.randn(10, 32),
        "audio": torch.randn(10, 32),
        "image": torch.randn(10, 32),
        "video": torch.randn(10, 32),
    }

    tokenizer.share_multimodal_weights(modal_weights)
    retrieved = tokenizer.get_multimodal_weights()
    assert len(retrieved) == 4

    comp_cfg = CompressionConfig()
    formatter = MultimodalWeightFormatter(comp_cfg)
    dna_path = os.path.join(temp_dir, "tok_weights.dna")

    tokenizer.export_multimodal_weights(formatter, dna_path)
    assert os.path.exists(dna_path)

    new_tokenizer = UnifiedTokenizer(vocab_cfg, bpe, patcher)
    loaded_weights = new_tokenizer.load_multimodal_weights(formatter, dna_path)
    assert len(loaded_weights) == 4
    for key in modal_weights:
        assert torch.allclose(loaded_weights[key], modal_weights[key], atol=1e-5)


def test_neurocore_model_multimodal_weight_sharing(temp_dir):
    cfg = NeuroCoreConfig.small()
    cfg.block.num_layers = 1
    cfg.block.alra.dim = 32
    cfg.block.alra.num_heads = 4
    cfg.block.alra.head_dim = 8
    model = NeuroCoreModel(cfg)

    weights = model.get_multimodal_weights()
    assert "text" in weights
    assert "audio" in weights
    assert "image" in weights
    assert "video" in weights

    text_len = cfg.vocab.text_range_end - cfg.vocab.text_range_start + 1
    new_text_w = torch.randn(text_len, cfg.block.alra.dim)

    model.bind_multimodal_weights({"text": new_text_w})
    updated_weights = model.get_multimodal_weights()
    assert torch.allclose(updated_weights["text"], new_text_w, atol=1e-5)

    comp_cfg = CompressionConfig()
    formatter = MultimodalWeightFormatter(comp_cfg)
    dna_path = os.path.join(temp_dir, "model_multimodal.dna")

    model.export_multimodal_weights(formatter, dna_path)
    assert os.path.exists(dna_path)

    model2 = NeuroCoreModel(cfg)
    model2.load_multimodal_weights(formatter, dna_path)
    model2_weights = model2.get_multimodal_weights()
    assert torch.allclose(model2_weights["text"], new_text_w, atol=1e-5)
