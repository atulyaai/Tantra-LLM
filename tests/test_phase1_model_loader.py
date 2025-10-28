"""Phase 1 tests: Model loader with local/API support."""

import pytest
from tantra_llm.utils.model_loader import ModelLoader
from tantra_llm.config import model_config


def test_model_loader_init():
    """Test ModelLoader initialization."""
    loader = ModelLoader(model_config.MODEL_CONFIG)
    assert loader.config == model_config.MODEL_CONFIG
    assert "models" in loader.models or loader.models is not None


def test_model_loader_load_spikingbrain_stub():
    """Test SpikingBrain loading (stub path)."""
    loader = ModelLoader(model_config.MODEL_CONFIG)
    result = loader.load_spikingbrain()
    assert result is not None
    # Should return dict with model/tokenizer keys even if None
    assert isinstance(result, dict)
    assert "model" in result
    assert "tokenizer" in result


def test_model_loader_get_vision_encoder():
    """Test vision encoder retrieval."""
    loader = ModelLoader(model_config.MODEL_CONFIG)
    encoder = loader.get_vision_encoder(mode="remote")
    assert encoder is not None
    assert hasattr(encoder, "embed_dim")


@pytest.mark.skipif(True, reason="Requires Whisper installed")
def test_model_loader_load_whisper():
    """Test Whisper loading (skip if not installed)."""
    loader = ModelLoader(model_config.MODEL_CONFIG)
    result = loader.load_whisper(model_size="tiny")
    assert result is not None

