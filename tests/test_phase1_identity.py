"""Phase 1 tests: System identity configuration."""

import pytest
from tantra_llm.config import identity


def test_identity_structure():
    """Test that identity config has all required keys."""
    assert "name" in identity.IDENTITY
    assert "version" in identity.IDENTITY
    assert "capability" in identity.IDENTITY
    assert identity.IDENTITY["name"] == "Tantra"
    assert identity.IDENTITY["version"] == "0.1-origins"


def test_speaking_style():
    """Test speaking style parameters."""
    ss = identity.IDENTITY["speaking_style"]
    assert "verbosity" in ss
    assert "formality" in ss
    assert "assertiveness" in ss
    assert "humor" in ss
    assert 0.0 <= ss["verbosity"] <= 1.0


def test_reasoning_style():
    """Test reasoning style parameters."""
    rs = identity.IDENTITY["reasoning_style"]
    assert "chain_of_thought" in rs
    assert "show_work" in rs
    assert "confidence_calibration" in rs


def test_memory_personality():
    """Test memory personality parameters."""
    mp = identity.IDENTITY["memory_personality"]
    assert "retention_threshold" in mp
    assert "compression_ratio" in mp
    assert "consolidation_frequency" in mp
    assert 0.0 <= mp["retention_threshold"] <= 1.0


def test_behavioral_boundaries():
    """Test behavioral boundaries and ethical constraints."""
    bb = identity.IDENTITY["behavioral_boundaries"]
    assert "ethical_constraints" in bb
    assert "allowed_risks" in bb
    assert "privacy_level" in bb
    assert isinstance(bb["ethical_constraints"], list)

