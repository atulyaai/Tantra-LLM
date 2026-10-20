import torch
from Tantra.model.peer import ProductKeyExperts


def test_shapes_noop_start_and_routing_range():
    torch.manual_seed(0)
    layer = ProductKeyExperts(d_model=32, n_keys=16, heads=2, top_k=4, key_dim=16)
    x = torch.randn(3, 5, 32)
    assert layer.n_experts == 256
    assert torch.equal(layer(x), torch.zeros_like(x))   # up=0: adding it to a trained model changes nothing
    scores, ids = layer.route(x.reshape(-1, 32))
    assert ids.shape == (15, 2, 4) and int(ids.min()) >= 0 and int(ids.max()) < 256


def test_product_keys_find_the_true_top_k():
    torch.manual_seed(1)
    layer = ProductKeyExperts(d_model=16, n_keys=8, heads=1, top_k=4, key_dim=8)
    x = torch.randn(6, 16)
    scores, ids = layer.route(x)
    q = layer.query(x).view(6, 1, 2, 4)
    full = (q[:, 0, 0] @ layer.keys[0, 0].T).unsqueeze(-1) + (q[:, 0, 1] @ layer.keys[0, 1].T).unsqueeze(-2)
    best = full.flatten(-2).topk(4, -1)
    assert torch.allclose(scores[:, 0], best.values, atol=1e-5)
    assert torch.equal(ids[:, 0].sort(-1).values, best.indices.sort(-1).values)


def test_gradients_touch_only_selected_experts():
    torch.manual_seed(2)
    layer = ProductKeyExperts(d_model=16, n_keys=32, heads=1, top_k=4, key_dim=8)
    with torch.no_grad():
        layer.up.weight.normal_()
    x = torch.randn(2, 16)
    layer(x).sum().backward()
    _, ids = layer.route(x)
    grad = layer.up.weight.grad
    assert grad.is_sparse
    touched = set(grad.coalesce().indices()[0].tolist())
    assert touched == set(ids.flatten().tolist()) and len(touched) <= 8 < layer.n_experts
