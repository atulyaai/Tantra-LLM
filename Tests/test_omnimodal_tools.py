"""Consolidated test suite: Tests/test_omnimodal_tools.py"""


# ─────────────────────────────────────────────────────────────────
# Source: test_multimodal_vision.py
# ─────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────
# Source: test_multimodal_weights.py
# ─────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────
# Source: test_omnimodal_pipeline.py
# ─────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────
# Source: test_tool_router.py
# ─────────────────────────────────────────────────────────────────

"""
Tests/test_tool_router.py — Unit tests for safe tool execution router.
"""
import pytest
from Tantra.tool_router import execute_tool_call, parse_and_execute_tool_calls, safe_eval_math

def test_calculator_basic_and_complex():
    assert safe_eval_math("2 + 2") == "4"
    assert safe_eval_math("9482 * 387") == "3669534"
    assert safe_eval_math("(45 * 89) + (1200 / 25)") == "4053.0"
    assert safe_eval_math("2 ** 10") == "1024"

def test_python_executor():
    code = "print(sum(range(10)))"
    res = execute_tool_call("python_executor", {"code": code})
    assert res == "45"

def test_file_reader():
    res = execute_tool_call("file_reader", {"filepath": "pyproject.toml"})
    assert "[project]" in res or "[build-system]" in res

def test_parse_and_execute():
    text = '<tool_call>{"name": "calculator", "arguments": {"expression": "100 * 50"}}</tool_call>'
    updated, did_exec = parse_and_execute_tool_calls(text)
    assert did_exec is True
    assert "<tool_result>" in updated
    assert "5000" in updated


def test_web_search_and_doc_retriever(tmp_path):
    from Tantra.tool_router import search_web, retrieve_local_documents

    # 1. Web search safe execution
    res_web = search_web("Quantum computing")
    assert isinstance(res_web, str)
    assert len(res_web) > 0

    # 2. Document retrieval from temporary directory
    doc_file = tmp_path / "architecture_notes.md"
    doc_file.write_text("Tantra is a local AI model with BitNet 1.58-bit ternary quantization.", encoding="utf-8")

    res_doc = retrieve_local_documents("BitNet quantization", doc_dir=str(tmp_path))
    assert "architecture_notes.md" in res_doc
    assert "BitNet" in res_doc


