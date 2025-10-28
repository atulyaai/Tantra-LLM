"""Smoke test for demo wiring to ensure v0.3 episodic memory integration works."""

import pytest


def test_demo_minimal_imports():
    """Test that demo imports work with flattened structure."""
    from demos.demo_minimal import build_demo
    brain = build_demo()
    assert brain is not None
    assert hasattr(brain, "step")
    assert hasattr(brain, "memory")


def test_demo_step_with_memory():
    """Test that step() works and episodic memory influences responses."""
    from demos.demo_minimal import build_demo
    
    brain = build_demo()
    
    # First interaction - no memory yet
    out1 = brain.step(text="I like coding in Python")
    assert out1 is not None
    
    # Store as memory
    brain.memory.episodic.store("user likes Python coding", importance=0.8)
    
    # Second interaction - should retrieve memory
    out2 = brain.step(text="What languages do I code in?")
    assert out2 is not None


def test_episodic_memory_retrieval():
    """Test episodic memory retrieval influences responses."""
    from core.memory.episodic_memory import EpisodicMemory
    
    mem = EpisodicMemory()
    mem.store("user prefers concise answers", importance=0.9)
    mem.store("user works at night", importance=0.7)
    
    recalls = mem.retrieve("What are my preferences?", top_k=2)
    assert len(recalls) > 0
    assert "concise" in recalls[0] or "night" in recalls[0]

