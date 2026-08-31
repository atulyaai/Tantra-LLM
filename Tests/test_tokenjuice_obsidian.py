"""
Tests for TokenJuice data squeezing & Obsidian offline Knowledge Graph vault.
"""
import os
import tempfile
import torch
import pytest

from Tantra.tokenjuice import TokenJuiceEngine
from Tantra.obsidian import ObsidianVaultEngine, KnowledgeNode, KnowledgeGraph


def test_tokenjuice_entropy_and_squeezing():
    engine = TokenJuiceEngine(entropy_threshold=0.3, enrichment_rate=0.5)
    
    # Low-entropy repeated sequence
    low_entropy = [1, 1, 1, 1, 1, 1, 1, 1]
    e_low = engine.compute_token_entropy(low_entropy)
    assert e_low < 0.2
    
    # High-entropy diverse sequence
    high_entropy = [1, 250, 4829, 991, 12, 592, 1024, 881]
    e_high = engine.compute_token_entropy(high_entropy)
    assert e_high > e_low
    
    # Test batch enrichment
    engine.register_synthetic_pair([100, 101, 102], [200, 201, 202])
    x = torch.zeros((2, 8), dtype=torch.long)
    y = torch.zeros((2, 8), dtype=torch.long)
    
    x_out, y_out = engine.enrich_batch(x, y)
    assert x_out.shape == (2, 8)
    assert y_out.shape == (2, 8)
    
    # Test loss weights
    targets = torch.tensor([1, 100, 5, 101])
    weights = engine.compute_dynamic_loss_weights(targets, high_priority_ids=[100, 101])
    assert weights[1].item() == 2.5
    assert weights[3].item() == 2.5
    assert weights[0].item() == 1.0


def test_obsidian_knowledge_graph_and_vault(tmp_path):
    vault_dir = tmp_path / "obsidian_vault"
    vault_dir.mkdir()
    
    # Create test markdown file
    note1 = vault_dir / "Tantra_Architecture.md"
    note1.write_text("# Tantra Architecture\n\nTantra uses ALRA linear attention and [[BitNet]] 1.58-bit ternary quantization.\n#architecture #ai")
    
    note2 = vault_dir / "BitNet.md"
    note2.write_text("# BitNet\n\nBitNet is a 1.58-bit ternary quantization model for extreme CPU speed.")
    
    engine = ObsidianVaultEngine(str(vault_dir))
    assert len(engine.graph.nodes) == 2
    
    node = engine.graph.get_node("Tantra_Architecture")
    assert node is not None
    assert "BitNet" in node.links
    assert "architecture" in node.tags
    
    # Test offline RAG prompt augmentation
    augmented = engine.augment_prompt("Tell me about Tantra Architecture")
    assert "[Knowledge Vault Context]" in augmented
    assert "Tantra Architecture" in augmented
