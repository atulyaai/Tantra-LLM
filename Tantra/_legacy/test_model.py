#!/usr/bin/env python
"""Test Tantra model building and forward pass."""

import sys
sys.path.insert(0, '.')

import torch
from Tantra.config import load_taxonomy
from Tantra.model.tantra_model import build_tantra_model

print("Loading taxonomy...")
taxonomy = load_taxonomy('Tantra/config/taxonomy.yaml')

print("Building model (this may take a moment)...")
model = build_tantra_model(
    'Tantra/config/taxonomy.yaml',
    'Model/best.pt',
    device='cpu',
)

print(f"\nModel built successfully!")
print(f"Experts: {model.n_experts}")
print(f"Expert names (first 10): {model.expert_names[:10]}")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
print("\nTesting forward pass...")
dummy_input = torch.randint(0, 1000, (1, 10))
with torch.no_grad():
    logits, analysis = model(dummy_input, return_router_analysis=True)

print(f"Logits shape: {logits.shape}")
print(f"Router analysis keys: {analysis.keys() if analysis else 'None'}")
print(f"Top experts: {analysis['top_experts'][:3] if analysis else 'N/A'}")

print("\n✅ All tests passed!")