"""Small smoke test for ProductKeyExperts (run with: PYTHONPATH=. python tests/test_peer_smoke.py)."""
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


def test_low_rank_int8_and_compact():
    torch.manual_seed(0)
    m = ProductKeyExperts(d_model=64, n_keys=8, heads=2, top_k=4, track_activations=True)
    with torch.no_grad():
        m.down.weight.normal_()
        m.up.weight.normal_()
    x = torch.randn(4, 64)
    with torch.no_grad():
        ref = m(x).clone()
    m.train()
    _ = m(x)
    report = m.compact_used_(min_activations=1)
    assert report["dropped"] > 0 and report["ratio"] > 1
    with torch.no_grad():
        y = m(x)
    assert torch.allclose(y, ref, atol=1e-4), (y - ref).abs().max().item()

    # Tables that really are low-rank must reconstruct. Random tables do not.
    m2 = ProductKeyExperts(d_model=32, n_keys=8, heads=1, top_k=2)
    basis = torch.randn(4, 32)
    code = torch.randn(m2.n_experts, 4)
    with torch.no_grad():
        m2.down.weight.copy_(code @ basis)
        m2.up.weight.copy_(code @ basis)
        info = m2.factor_low_rank_(rank=4)
    assert info["down_rel_rmse"] < 1e-4 and info["up_rel_rmse"] < 1e-4
    assert info["stored_bytes"] < 64 * m2.n_experts * 4

    m3 = ProductKeyExperts(d_model=32, n_keys=8, heads=1, top_k=2)
    with torch.no_grad():
        m3.down.weight.normal_()
        m3.up.weight.normal_()
    x3 = torch.randn(4, 32)
    with torch.no_grad():
        base = m3(x3).clone()
        before = m3.stored_bytes()
        q = m3.quantize_int8_()
        out = m3(x3)
    assert q["ratio"] > 2.5
    assert (out - base).abs().max().item() < 0.2
    print("compress OK", {"compact": report, "low_rank": info, "int8": q, "before": before})


if __name__ == "__main__":
    test_peer_forward_backward()
    test_low_rank_int8_and_compact()
