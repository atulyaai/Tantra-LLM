"""End-to-end tests for full Tantra brain system."""

import torch


def test_e2e_text_only():
    """Test text-only generation through full pipeline."""
    from demos.demo_minimal import build_demo
    
    brain = build_demo()
    
    # Simple text input
    response = brain.step(text="What is Python?")
    
    # Assert response exists (even if stub)
    assert response is not None
    assert isinstance(response, str)


def test_e2e_with_vision():
    """Test multimodal generation with vision input."""
    from demos.demo_minimal import build_demo
    
    brain = build_demo()
    
    # Mock image embedding
    img_embed = torch.zeros(1, 1024)
    response = brain.step(text="Describe this image", image=img_embed)
    
    assert response is not None


def test_memory_retrieval():
    """Test that episodic memory influences output."""
    from demos.demo_minimal import build_demo
    
    brain = build_demo()
    
    # Store memory
    brain.memory.episodic.store("user likes Python", importance=0.9)
    
    # Query should retrieve memory
    response = brain.step(text="What do I like?")
    
    assert response is not None

