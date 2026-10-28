"""Small smoke test for ProductKeyExperts (run with: python -m pytest tests/test_peer_smoke.py -q)."""
import torch
from Tantra.model.peer import ProductKeyExperts


def test_peer_forward_backward():
    m = ProductKeyExperts(d_model=128, n_keys=16, heads=2, top_k=4, track_activations=True)
    x = torch.randn(8, 128)
    y = m(x)
    assert y.shape == x.shape
    loss = y.sum()
    loss.backward()
    assert m.down.weight.grad is not None
    stats = m.usage_stats()
    assert stats is not None
    assert stats["n_experts"] == 16 * 16
    print("PEER smoke OK:", stats)


if __name__ == "__main__":
    test_peer_forward_backward()
