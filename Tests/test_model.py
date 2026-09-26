"""Model correctness: attention math, generation cache, growth, categories, loading."""
import torch

from Tantra.config import ALRAConfig, NeuroCoreConfig
from Tantra.evolution import AutoGrowthController
from Tantra.model import ALRAAttention, CausalSelfAttention, NeuroCoreModel, load_model


def tiny(local_every=2, vocab=300):
    cfg = NeuroCoreConfig.tiny()
    cfg.vocab.vocab_size = cfg.vocab.byte_bpe_vocab = vocab
    cfg.block.alra.local_attn_every = local_every
    cfg.block.alra.local_window = 6
    return NeuroCoreModel(cfg, use_mtp=False).eval()


def test_alra_chunked_equals_recurrence():
    torch.manual_seed(0)
    a = ALRAAttention(ALRAConfig(dim=64, num_heads=4, head_dim=16))
    T = 70
    Q, K = torch.rand(1, 4, T, 16) + 0.1, torch.rand(1, 4, T, 16) + 0.1
    V, g = torch.randn(1, 4, T, 16), torch.rand(1, 4, T) * 0.5 + 0.5
    st, ref = {}, []
    for t in range(T):
        o, st = a._sequential_forward(Q[:, :, t:t + 1], K[:, :, t:t + 1], V[:, :, t:t + 1], g[:, :, t:t + 1], st)
        ref.append(o)
    out, S, _ = a._chunked_forward(Q, K, V, g, None, None, chunk=16)
    assert torch.allclose(out, torch.cat(ref, 2), atol=1e-3)
    assert torch.allclose(S, st["S"], atol=1e-4)


def test_sliding_window_cache_matches_full_and_stays_bounded():
    torch.manual_seed(0)
    attn = CausalSelfAttention(ALRAConfig(dim=64, num_heads=4, head_dim=16), window=5).eval()
    x = torch.randn(1, 14, 64)
    full, _ = attn(x)
    state, outs = {}, []
    for t in range(14):
        o, state = attn(x[:, t:t + 1], state=state)
        outs.append(o)
    assert torch.allclose(full, torch.cat(outs, 1), atol=1e-5)
    assert state["k"].shape[2] == 4


def test_hybrid_layout_and_prefill_cache_exact():
    torch.manual_seed(0)
    m = tiny()
    assert [type(l.attn).__name__ for l in m.layers] == ["ALRAAttention", "CausalSelfAttention"] * 2
    ids = torch.randint(21, 300, (1, 30))
    with torch.no_grad():
        full = m(ids, use_latent_reasoning=False)[0]
        states = [{} for _ in m.layers]
        first, states = m(ids[:, :20], states=states, use_latent_reasoning=False)
        rest = []
        for t in range(20, 30):
            o, states = m(ids[:, t:t + 1], states=states, use_latent_reasoning=False)
            rest.append(o)
    assert torch.allclose(full, torch.cat([first] + rest, 1), atol=1e-4)


def test_generate_stops_at_eos_and_returns_prompt_plus_tokens():
    m = tiny()
    ids = torch.randint(21, 300, (1, 5))
    out = m.generate(ids, max_new_tokens=7, temperature=0.0, eos_token_id=None)
    assert out.shape == (1, 12) and torch.equal(out[:, :5], ids)
    assert len(list(m.generate_stream(ids, max_new_tokens=4, temperature=0.8))) == 4


def test_growth_adds_identity_layer():
    torch.manual_seed(0)
    m = tiny(local_every=0)
    ids = torch.randint(21, 300, (1, 10))
    with torch.no_grad():
        before = m(ids, use_latent_reasoning=False)[0]
    assert AutoGrowthController().grow_capacity(m)
    with torch.no_grad():
        after = m(ids, use_latent_reasoning=False)[0]
    assert len(m.layers) == 5 and torch.equal(before, after)


def test_new_category_layer_is_identity_until_trained():
    m = tiny()
    ids = torch.randint(21, 300, (1, 10))
    with torch.no_grad():
        base = m(ids, use_latent_reasoning=False)[0]
        m.add_category_layers(["greetings"], depth=1, clone_layer_index=3)
        routed = m(ids, use_latent_reasoning=False, adapter_name="greetings")[0]
    assert torch.allclose(base, routed)
    m.freeze_for_category("greetings")
    trainable = {n.split(".")[0] for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"category_layers", "category_gates"}


def test_load_model_rebuilds_grown_model_with_categories(tmp_path):
    m = tiny()
    AutoGrowthController().grow_capacity(m)
    m.add_category_layers(["code"], depth=2)
    path = tmp_path / "m.pt"
    torch.save({"model_state_dict": m.state_dict(), "config": m.config}, path)
    loaded, _ = load_model(str(path))
    assert len(loaded.layers) == 5 and loaded.category_depth("code") == 2
    ids = torch.randint(21, 300, (1, 8))
    with torch.no_grad():
        assert torch.allclose(m(ids)[0], loaded(ids)[0], atol=1e-6)
    q, _ = load_model(str(path), int8=True)
    with torch.no_grad():
        assert q(ids)[0].shape == (1, 8, 300)


def test_moe_and_mtp_forward():
    cfg = NeuroCoreConfig.tiny()
    cfg.vocab.vocab_size = 300
    cfg.moe.num_experts, cfg.moe.real_top1 = 2, True
    m = NeuroCoreModel(cfg, use_mtp=True, use_moe=True)
    (main, mtp), _ = m(torch.randint(0, 300, (2, 9)), return_mtp=True)
    assert main.shape == mtp.shape == (2, 9, 300)
    assert m.get_aux_loss().item() >= 0
