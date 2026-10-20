import pytest
import torch
from Tantra.config import ModelConfig
from Tantra.inference.expert_cache import ExpertCache, ExpertMismatch
from Tantra.model.tantra_model import TantraModel, quantize_for_cpu, save_expert

NAMES = [f'expert{i:02d}' for i in range(30)]


def tiny_cfg(**kw):
    return ModelConfig(vocab_size=64, d_model=16, n_layers=2, n_heads=2, n_kv_heads=1, mlp_mult=2, max_seq_len=32, lora_rank=4, **kw)


@pytest.fixture
def saved(tmp_path):
    torch.manual_seed(0)
    model = TantraModel(tiny_cfg())
    for name in NAMES:
        model.add_expert(name)
    with torch.no_grad():  # make every adapter non-trivial
        for n, p in model.named_parameters():
            if 'lora_B' in n:
                p.normal_(0, 0.1)
    model.save(tmp_path / 'base.pt', tmp_path / 'experts')
    return tmp_path, model


def lora_count(model):
    return sum(1 for n, _ in model.named_parameters() if 'lora_' in n)


def test_lazy_load_starts_empty_and_respects_capacity(saved):
    root, _ = saved
    model = TantraModel.load(root / 'base.pt', lazy=True)
    assert model.expert_names == NAMES and lora_count(model) == 0
    cache = ExpertCache(model, root / 'experts', capacity=8)
    cache.activate([('expert00', 0.6), ('expert01', 0.4)])
    assert list(cache.resident) == ['expert00', 'expert01']
    for i in range(2, 12):
        cache.activate([(NAMES[i], 1.0)])
    assert len(cache.resident) == 8 and cache.evictions == 4
    per_expert = lora_count(model) // 8
    assert per_expert == 2 * 2 * len(model.cfg.lora_targets)  # A and B per target module per layer


def test_lazy_outputs_match_eager(saved):
    root, _ = saved
    eager = TantraModel.load(root / 'base.pt', root / 'experts').eval()
    lazy = TantraModel.load(root / 'base.pt', lazy=True).eval()
    cache = ExpertCache(lazy, root / 'experts', capacity=2)
    x = torch.randint(0, 64, (1, 10))
    route = [('expert05', 0.7), ('expert17', 0.3)]
    eager.set_active(route); cache.activate(route)
    with torch.no_grad():
        assert torch.allclose(eager(x), lazy(x), atol=1e-5)
    cache.activate([('expert09', 1.0)]); cache.activate(route)  # evict and reload: still identical
    with torch.no_grad():
        eager.set_active(route)
        assert torch.allclose(eager(x), lazy(x), atol=1e-5)


def test_mismatched_adapter_is_refused(saved, tmp_path):
    root, _ = saved
    bigger = TantraModel(ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, mlp_mult=2, max_seq_len=32, lora_rank=4))
    bigger.add_expert('expert03')
    save_expert(bigger, 'expert03', root / 'experts')  # overwrite with an adapter for another base
    model = TantraModel.load(root / 'base.pt', lazy=True)
    cache = ExpertCache(model, root / 'experts', capacity=4)
    with pytest.raises(ValueError, match='does not fit'):
        cache.activate([('expert03', 1.0)])
    assert 'expert03' not in cache.resident and lora_count(model) == 0


def test_hash_mismatch_is_refused(saved):
    root, model = saved
    save_expert(model, 'expert04', root / 'experts', base_sha256='a' * 64)
    lazy = TantraModel.load(root / 'base.pt', lazy=True)
    cache = ExpertCache(lazy, root / 'experts', capacity=4, base_sha256='b' * 64)
    with pytest.raises(ExpertMismatch, match='different base'):
        cache.activate([('expert04', 1.0)])


def test_unknown_route_names_are_ignored_and_quantized_base_accepts_experts(saved):
    root, _ = saved
    model = quantize_for_cpu(TantraModel.load(root / 'base.pt', lazy=True).eval())
    cache = ExpertCache(model, root / 'experts', capacity=2)
    assert cache.activate([('not-an-expert', 1.0), ('expert00', 1.0)]) == [('expert00', 1.0)]
    with torch.no_grad():
        assert model(torch.randint(0, 64, (1, 5))).shape == (1, 5, 64)
