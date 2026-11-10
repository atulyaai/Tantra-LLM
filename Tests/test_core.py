"""
Base model, PEER experts, router lifetime, engine, reasoning and memory.

Merged: test_tantra.py, test_peer.py, test_peer_smoke.py, test_lazy_experts.py, test_engine_lazy.py, test_batched_sft.py, test_reasoning.py, test_memory.py.
"""

# ──────────────────────────────────────────────────────────────────────  test_tantra.py  ──────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Tantra import ModelConfig, estimate_params, get_preset, load_taxonomy, validate_taxonomy
from Tantra.train import EncodedDataset, read_jsonl
from Tantra.conversation_train import sc_generate as generate, generate_mixed, write_seed_conversation as write_all
from Tantra.conversation_train import write_seed_knowledge as write_knowledge
from Tantra.train import TantraModel, quantize_for_cpu
from Tantra.tokenizer import TantraTokenizer
from Tantra.train import TantraTrainer, TrainArgs

TINY = ModelConfig(vocab_size=300, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)


def _model():
    torch.manual_seed(0)
    return TantraModel(ModelConfig.from_dict(TINY.to_dict())).eval()


def test_forward_shape_and_loss():
    m = _model()
    ids = torch.randint(0, 300, (2, 10))
    assert m(ids).shape == (2, 10, 300)
    loss = m.loss(ids, ids)
    loss.backward()
    assert torch.isfinite(loss)


def test_new_expert_is_noop_until_trained():
    m = _model()
    ids = torch.randint(0, 300, (1, 8))
    base = m(ids)
    m.add_expert("greeting", rank=4)
    m.set_active([("greeting", 1.0)])
    assert torch.allclose(m(ids), base, atol=1e-6)  # LoRA B starts at zero


def test_experts_are_isolated():
    m = _model()
    m.add_expert("a", 4); m.add_expert("b", 4)
    pa, pb = m.expert_parameters("a"), m.expert_parameters("b")
    assert pa and pb and not {id(p) for p in pa} & {id(p) for p in pb}
    m.set_active([("a", 1.0)])
    for p in m.parameters():
        p.requires_grad = False
    for p in pa:
        p.requires_grad = True
    ids = torch.randint(0, 300, (1, 8))
    m.loss(ids, ids).backward()
    assert all(p.grad is None for p in pb)


def test_kv_cache_matches_full_forward():
    m = _model()
    ids = torch.randint(0, 300, (1, 12))
    full = m(ids)[:, -1]
    caches = [{} for _ in m.layers]
    m(ids[:, :-1], caches, 0)
    step = m(ids[:, -1:], caches, 11)[:, -1]
    assert torch.allclose(full, step, atol=1e-4)


def test_grow_depth_preserves_output():
    m = _model()
    m.add_expert("x", 4)
    with torch.no_grad():
        for p in m.expert_parameters("x"):
            p.normal_(0, 0.02)
    m.set_active([("x", 1.0)])
    ids = torch.randint(0, 300, (1, 10))
    before = m(ids)
    m.grow_depth(2)
    m.set_active([("x", 1.0)])
    assert len(m.layers) == 4
    assert torch.allclose(m(ids), before, atol=1e-5)


def test_router_grow_and_remove():
    m = _model()
    for n in ["a", "b", "c"]:
        m.add_expert(n, 4)
    assert m.router.head.out_features == 3
    m.remove_expert("b")
    assert m.expert_names == ["a", "c"] and m.router.head.out_features == 2
    r = m.route(torch.randint(0, 300, (1, 5)), max_experts=2, cum_prob=1.0, min_prob=0)
    assert len(r) == 2 and abs(sum(w for _, w in r) - 1) < 1e-5


def test_save_load_roundtrip(tmp_path):
    m = _model()
    m.add_expert("g", 4)
    with torch.no_grad():
        for p in m.expert_parameters("g"):
            p.normal_(0, 0.02)
    m.save(tmp_path / "base.pt", tmp_path / "experts")
    m2 = TantraModel.load(tmp_path / "base.pt", tmp_path / "experts").eval()
    ids = torch.randint(0, 300, (1, 6))
    m.set_active([("g", 1.0)]); m2.set_active([("g", 1.0)])
    assert torch.allclose(m(ids), m2(ids), atol=1e-6)


def test_int8_quantization_runs():
    m = quantize_for_cpu(_model())
    out = list(m.generate(torch.randint(0, 300, (1, 5)), max_new_tokens=5, temperature=0))
    assert len(out) == 5


def test_seed_replies_match_questions(tmp_path):
    rows = generate("identity_limitations", 300)
    for r in rows:
        u, a = r["messages"][-2]["content"].lower(), r["messages"][-1]["content"].lower()
        if "internet" in u or "web" in u:
            assert "internet" in a or "web" in a, (u, a)
    assert all(not any("ऀ" <= c <= "ॿ" for m in r["messages"] for c in m["content"]) for r in rows)
    mixed = generate_mixed("greeting", "identity_self", 20)
    assert len(mixed) == 20 and mixed[0]["experts"] == ["greeting", "identity_self"]
    counts = write_all(tmp_path, n=50, mixed_n=10)
    assert counts["identity_self"] == 50 + 10  # own rows + greeting/identity mix


def test_knowledge_seed_is_correct(tmp_path):
    import re
    write_knowledge(tmp_path, n=200)
    for r in read_jsonl(tmp_path / "knowledge" / "gk_math.jsonl"):
        a, op, b, res = re.match(r"(\d+) ([+\-x]) (\d+) = (-?\d+)", r["messages"][1]["content"]).groups()
        assert int(res) == {"+": int(a) + int(b), "-": int(a) - int(b), "x": int(a) * int(b)}[op]
    geo = read_jsonl(tmp_path / "knowledge" / "gk_geography.jsonl")
    assert any("Paris" in r["messages"][1]["content"] for r in geo) or len(geo) == 200


def test_junk_rows_filtered(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"text": "<|user|>\\nhi\\n<|assistant|> Sample 1 of 10000."}\n'
                 '{"text": "<|user|>\\nhi\\n<|assistant|> Hello!"}\n'
                 '{"text": "<|user|>\\nhi\\n<|assistant|> Hello!"}\n', encoding="utf-8")
    rows = read_jsonl(f)
    assert rows == [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]}]


def test_tokenizer_chat_mask():
    tok = TantraTokenizer.train(["hello there", "who are you", "I am Tantra"] * 20, vocab_size=300)
    ids, mask = tok.encode_chat([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "I am Tantra"}])
    assert tok.decode([i for i, k in zip(ids, mask) if k]) == "I am Tantra"
    assert ids[-1] == tok.end_id and mask[-1] == 1


def test_presets_and_taxonomy():
    assert 0.8e9 < estimate_params(get_preset("1b")) < 1.3e9
    assert validate_taxonomy(load_taxonomy()) == []


@pytest.mark.slow
def test_end_to_end_tiny_pipeline(tmp_path):
    data = tmp_path / "data"
    write_all(data, n=150)
    write_knowledge(data, n=150)
    gen = tmp_path / "general"
    gen.mkdir()
    story = "Once upon a time a cat sat in the sun and was happy all day long.\n<|endoftext|>\n"
    (gen / "stories.txt").write_text(story * 50, encoding="utf-8")
    t = TantraTrainer(tmp_path / "model", [data], general_dir=gen)
    assert len(t.general_rows()) == 50
    t.train_tokenizer(vocab_size=600)
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128)
    model = t.train_base(cfg, TrainArgs(steps=30, batch_size=8, max_len=96, log_every=1000),
                         experts=["greeting", "identity_self"], resume=False)
    r = t.train_expert("identity_self", model, TrainArgs(steps=40, lr=5e-3, warmup=5, batch_size=8, log_every=1000))
    assert r["loss_expert"] < r["loss_base"]
    t.train_expert("greeting", model, TrainArgs(steps=10, batch_size=8, log_every=1000))
    acc = t.train_router(model, steps=300)
    assert acc > 0.6
    from Tantra.main import TantraEngine
    eng = TantraEngine(tmp_path / "model")
    assert eng.route("who are you?")[0][0] == "identity_self"
    assert isinstance(eng.chat([{"role": "user", "content": "hi"}], max_new_tokens=8), str)


def test_adaptive_routing_picks_count():
    m = _model()
    for n in ["a", "b", "c"]:
        m.add_expert(n, 4)
    with torch.no_grad():
        m.router.head.weight.zero_()
        m.router.head.bias.copy_(torch.tensor([10.0, 0.0, 0.0]))  # confident -> 1 expert
    assert [n for n, _ in m.route(torch.tensor([[1, 2]]))] == ["a"]
    with torch.no_grad():
        m.router.head.bias.copy_(torch.tensor([1.0, 1.0, -5.0]))  # split -> 2 experts
    assert sorted(n for n, _ in m.route(torch.tensor([[1, 2]]))) == ["a", "b"]

# ──────────────────────────────────────────────────────────────────────  test_peer.py  ──────────────────────────────────────────────────────────────────────

import torch
from Tantra.train import ProductKeyExperts


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

# ──────────────────────────────────────────────────────────────────────  test_peer_smoke.py  ──────────────────────────────────────────────────────────────────────

"""Small smoke test for ProductKeyExperts."""
import torch
from Tantra.train import ProductKeyExperts


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

# ──────────────────────────────────────────────────────────────────────  test_lazy_experts.py  ──────────────────────────────────────────────────────────────────────

import pytest
import torch
from Tantra import ModelConfig
from Tantra.main import ExpertCache, ExpertMismatch
from Tantra.train import TantraModel, quantize_for_cpu, save_expert

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


def test_no_repeat_ngram_blocks_repeated_trigrams():
    torch.manual_seed(0)
    model = TantraModel(tiny_cfg()).eval()
    with torch.no_grad():  # force a model that always prefers tokens 5, 6, 7 in a loop
        model.embed.weight.zero_(); model.embed.weight[5, 0] = model.embed.weight[6, 1] = model.embed.weight[7, 2] = 1.0
    ids = torch.tensor([[5, 6, 7]])
    free = list(model.generate(ids, 12, temperature=0, repetition_penalty=1.0))
    blocked = list(model.generate(ids, 12, temperature=0, repetition_penalty=1.0, no_repeat_ngram=3))
    trigrams = [tuple(blocked[i:i + 3]) for i in range(len(blocked) - 2)]
    assert len(trigrams) == len(set(trigrams))
    assert len(blocked) == len(free) == 12


def test_int8_head_matches_fp32_closely():
    torch.manual_seed(0)
    fp = TantraModel(tiny_cfg()).eval()
    q = TantraModel(tiny_cfg()).eval(); q.load_state_dict(fp.state_dict())
    quantize_for_cpu(q, layers=False)
    x = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        a, b = fp(x), q(x)
    assert '_q_head' in q.__dict__ and '_q_head' not in fp.__dict__
    assert (a.argmax(-1) == b.argmax(-1)).float().mean() >= 0.9
    assert set(q.state_dict()) == set(fp.state_dict())  # the INT8 copy never leaks into checkpoints

# ──────────────────────────────────────────────────────────────────────  test_engine_lazy.py  ──────────────────────────────────────────────────────────────────────

import torch
from Tantra import ModelConfig
from Tantra.main import TantraEngine
from Tantra.train import TantraModel
from Tantra.tokenizer import TantraTokenizer


def test_engine_loads_experts_on_demand_and_reports_timing(tmp_path):
    tok = TantraTokenizer.train(['Hello hi welcome thanks I am Tantra. Who made you?'] * 3, vocab_size=280, min_frequency=1)
    tok.save(tmp_path / 'tokenizer.json')
    model = TantraModel(ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
                                    mlp_mult=2, max_seq_len=64, lora_rank=2))
    for name in ('greeting', 'identity_self', 'farewell'):
        model.add_expert(name)
    model.save(tmp_path / 'base.pt', tmp_path / 'experts')
    eng = TantraEngine(tmp_path, max_experts=1, expert_cache=1)
    assert eng.cache is not None and len(eng.cache.resident) == 0
    reply = eng.chat([{'role': 'user', 'content': 'Hello'}], max_new_tokens=4, experts=[('identity_self', 1.0)])
    assert isinstance(reply, str)
    assert list(eng.cache.resident) == ['identity_self']
    assert eng.last_timing['experts_loaded'] == 1 and eng.last_timing['tokens'] >= 1
    eng.chat([{'role': 'user', 'content': 'Hi'}], max_new_tokens=2, experts=[('farewell', 1.0)])
    assert list(eng.cache.resident) == ['farewell'] and eng.cache.evictions == 1
    assert eng.benchmark(n_tokens=2)['experts_resident'] <= 1



def test_router_none_class_means_base_only(tmp_path):
    tok = TantraTokenizer.train(['Hello hi welcome thanks I am Tantra.'] * 3, vocab_size=280, min_frequency=1)
    tok.save(tmp_path / 'tokenizer.json')
    model = TantraModel(ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
                                    mlp_mult=2, max_seq_len=64, lora_rank=2))
    model.add_expert('greeting'); model.add_expert('identity_self')
    model.save(tmp_path / 'base.pt', tmp_path / 'experts')
    eng = TantraEngine(tmp_path, max_experts=2)

    class Stub:
        names = ['greeting', 'identity_self', 'none']
        def __init__(self, p): self.p = torch.tensor(p)
        def probs(self, text): return self.p

    eng.ngram = Stub([0.2, 0.08, 0.72])        # "none" wins -> base only, not a partial greeting
    assert eng.route('my exam went well') == []
    eng.ngram = Stub([0.55, 0.05, 0.40])       # expert wins; "none" never becomes an expert
    route = eng.route('hello')
    assert [n for n, _ in route] == ['greeting'] and abs(sum(w for _, w in route) - 1.0) < 1e-6

# ──────────────────────────────────────────────────────────────────────  test_batched_sft.py  ──────────────────────────────────────────────────────────────────────

import torch
from Tantra import ModelConfig
from Tantra.train import collate
from Tantra.train import TantraModel


def test_one_padded_batch_equals_token_weighted_per_row_passes():
    torch.manual_seed(0)
    model = TantraModel(ModelConfig(vocab_size=50, d_model=16, n_layers=2, n_heads=2, n_kv_heads=1, mlp_mult=2, max_seq_len=32))
    rows = []
    for length, start in ((7, 2), (11, 5), (4, 1)):  # different lengths -> right padding in the batch
        ids = torch.randint(3, 50, (length,)).tolist()
        labels = [-100] * start + ids[start:]           # supervise only the "assistant" tail
        rows.append((ids, labels))
    # Old path: one forward per row, loss weighted by its share of supervised tokens.
    model.zero_grad()
    batches = [collate([r], 0) for r in rows]
    count = sum(int((y[:, 1:] != -100).sum()) for _, y, _ in batches)
    for x, y, _ in batches:
        (model.loss(x, y) * int((y[:, 1:] != -100).sum()) / count).backward()
    old = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    # New path: one right-padded batch.
    model.zero_grad()
    x, y, _ = collate(rows, 0)
    model.loss(x, y).backward()
    for n, p in model.named_parameters():
        if n in old:
            assert torch.allclose(old[n], p.grad, atol=1e-6), n

# ──────────────────────────────────────────────────────────────────────  test_reasoning.py  ──────────────────────────────────────────────────────────────────────

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Tantra.reasoning as R


class FakeEngine:
    """Stands in for TantraEngine: fixed route + scripted replies."""

    def __init__(self, route, replies):
        self.route_ = route
        self.replies = list(replies)

    def route(self, q):
        return self.route_

    def chat(self, messages, **kw):
        return self.replies.pop(0) if self.replies else "I don't know."


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "LEARNED", tmp_path / "learned" / "facts.jsonl")
    monkeypatch.setattr(R, "AUTO_FEEDBACK", tmp_path / "auto")
    data = tmp_path / "curated" / "knowledge"
    data.mkdir(parents=True)
    row = {"messages": [{"role": "user", "content": "What is the capital of Japan?"},
                        {"role": "assistant", "content": "The capital of Japan is Tokyo."}]}
    (data / "gk_geography.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return [tmp_path / "curated"]


def test_math_tool_exact():
    assert R.math_answer("what is 12 times 7?") == "12 x 7 = 84."
    assert R.math_answer("45 plus 38") == "45 + 38 = 83."
    assert R.math_answer("100 divided by 8") == "100 / 8 = 12.5."
    assert R.math_answer("who are you") is None


def test_fast_path_for_conversation(tmp_path, monkeypatch):
    dirs = _isolate(tmp_path, monkeypatch)
    r = R.Reasoner(FakeEngine([("greeting", 1.0)], ["Hello!"]), dirs, use_web=False, use_teacher=False)
    out = r.answer([{"role": "user", "content": "hi"}])
    assert out["reply"] == "Hello!" and out["source"] == "model"


def test_wrong_math_draft_is_fixed_and_learned(tmp_path, monkeypatch):
    dirs = _isolate(tmp_path, monkeypatch)
    eng = FakeEngine([("gk_math", 1.0)], ["12 x 7 = 74.", "12 x 7 = 72.", "12 x 7 = 82."])
    out = R.Reasoner(eng, dirs, use_web=False, use_teacher=False).answer([{"role": "user", "content": "what is 12 times 7?"}])
    assert out["reply"] == "12 x 7 = 84." and out["source"] == "calculator"
    assert "84" in (tmp_path / "auto" / "gk_math.jsonl").read_text(encoding="utf-8")


def test_correct_draft_is_verified(tmp_path, monkeypatch):
    dirs = _isolate(tmp_path, monkeypatch)
    eng = FakeEngine([("gk_geography", 1.0)], ["Tokyo is the capital of Japan."])
    out = R.Reasoner(eng, dirs, use_web=False, use_teacher=False).answer(
        [{"role": "user", "content": "what is the capital of Japan?"}])
    assert out["source"] == "model (verified)"


def test_web_answer_is_learned_then_offline(tmp_path, monkeypatch):
    dirs = _isolate(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(R, "search_web", lambda q: calls.append(q) or
                        ("Canberra is the capital city of Australia.", "https://en.wikipedia.org/wiki/Canberra"))
    q = [{"role": "user", "content": "what is the capital of Australia?"}]
    r = R.Reasoner(FakeEngine([("gk_geography", 1.0)], ["Sydney."] * 3), dirs, use_teacher=False)
    assert "Canberra" in r.answer(q)["reply"] and len(calls) == 1
    # new session, no internet allowed: answered from what was learned
    r2 = R.Reasoner(FakeEngine([("gk_geography", 1.0)], ["Sydney."] * 3), dirs, use_web=False, use_teacher=False)
    out = r2.answer(q)
    assert "Canberra" in out["reply"] and len(calls) == 1


def test_contraction_question_triggers_fact_check(tmp_path, monkeypatch):
    # regression: "whats X" (no apostrophe) failed the old `what\\b` regex and
    # skipped the knowledge index entirely -> model hallucination
    dirs = _isolate(tmp_path, monkeypatch)
    eng = FakeEngine([("smalltalk", 1.0)], ["I can chat in English and Hindi."] * 3)
    out = R.Reasoner(eng, dirs, use_web=False, use_teacher=False).answer(
        [{"role": "user", "content": "whats the capital of japan"}])
    assert "Tokyo" in out["reply"] and out["source"].startswith("local")


def test_chatty_expert_skips_fact_path(tmp_path, monkeypatch):
    # "tell me a joke" matches "tell me" but must NOT hit the knowledge path
    dirs = _isolate(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(R, "search_web", lambda q: calls.append(q) or ("joke wiki", "u"))
    eng = FakeEngine([("humor", 1.0)], ["Why did the computer go to the doctor? Because it had a virus!"])
    out = R.Reasoner(eng, dirs, use_teacher=False).answer([{"role": "user", "content": "tell me a joke"}])
    assert out["source"] == "model" and not calls


def test_ngram_router_handles_typos_and_mixes():
    from Tantra.main import route_probs_to_experts, train_ngram_router
    data = {"greeting": ["hi", "hello", "hey there", "good morning", "how are you"],
            "identity_self": ["who are you", "what is your name", "who made you", "who created you"],
            "gk_geography": ["what is the capital of france", "capital of japan", "capital of india"]}
    names = list(data)
    texts, labels = [], []
    for i, n in enumerate(names):
        for t in data[n] * 30:
            texts.append(t); labels.append(i)
    for _ in range(40):
        texts += ["hi, who made you"] * 2; labels += [0, 1]
    r, _ = train_ngram_router(texts, labels, names, holdout=0.0)
    top = lambda t: route_probs_to_experts(r.probs(t), names)[0][0]
    assert top("whats india capital") == "gk_geography"
    assert top("who mad u") == "identity_self"
    assert top("helo") == "greeting"
    assert {n for n, _ in route_probs_to_experts(r.probs("hi, who made you"), names)} == {"greeting", "identity_self"}

# ──────────────────────────────────────────────────────────────────────  test_memory.py  ──────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from Tantra.evaluate_conversation import TantraMemoryStore
from Tantra.evaluate_conversation import InvertedBM25Index, tokenize_query
from Tantra.evaluate_conversation import DenseVectorIndex
import torch


def test_tokenize_query():
    toks = tokenize_query("What is the capital of India?")
    assert "capital" in toks
    assert "india" in toks
    assert "what" not in toks
    assert "the" not in toks


def test_inverted_bm25_search():
    index = InvertedBM25Index()
    index.add_document(0, "The capital of France is Paris.")
    index.add_document(1, "The capital of Japan is Tokyo.")
    index.add_document(2, "Quantum mechanics describes atomic physics.")

    results = index.search("capital France", top_k=2)
    assert len(results) > 0
    assert results[0][0] == 0  # doc 0 matches best


def test_dense_vector_index():
    idx = DenseVectorIndex(dim=4)
    v0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    idx.add(v0, 100)
    idx.add(v1, 200)

    q = torch.tensor([0.9, 0.1, 0.0, 0.0])
    res = idx.search(q, top_k=1)
    assert res[0][0] == 100


def test_memory_store_hybrid_and_save_load(tmp_path):
    store = TantraMemoryStore(dim=64, alpha=0.6)
    store.add_fact("What is the capital of India?", "New Delhi is the capital of India.", "gk_geography")
    store.add_fact("Who developed theory of relativity?", "Albert Einstein developed it.", "gk_science")
    store.add_fact("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।", "gk_geography")

    # Exact search
    res = store.search("capital of India")
    assert res is not None
    q, a, expert, score = res
    assert "New Delhi" in a
    assert expert == "gk_geography"

    # Hindi Unicode search
    res_hi = store.search("भारत की राजधानी")
    assert res_hi is not None
    assert "नई दिल्ली" in res_hi[1]

    # Save and load roundtrip
    p = tmp_path / "memory.pt"
    store.save(p)
    loaded = TantraMemoryStore.load(p)
    assert len(loaded) == 3
    res_loaded = loaded.search("relativity Einstein")
    assert res_loaded is not None
    assert "Einstein" in res_loaded[1]


def test_memory_store_dataset_ingest(tmp_path):
    import json
    folder=tmp_path/'curated'/'knowledge'
    folder.mkdir(parents=True)
    row={'messages':[{'role':'user','content':'What is the capital of India?'},{'role':'assistant','content':'New Delhi'}]}
    (folder/'gk_geography.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
    store = TantraMemoryStore()
    n = store.load_from_datasets([folder.parent], learned_path=None, max_per_file=50)
    assert n == 1
    res = store.search("capital")
    assert res is not None


def test_triple_store_dna_and_indic(tmp_path):
    from Tantra.evaluate_conversation import TripleStore

    store = TripleStore()
    store.add_triple("India", "capital", "New Delhi")
    store.add_triple("India", "currency", "Rupee")
    store.add_triple("Sanskrit", "author", "Panini")
    store.add_triple("भारत", "capital", "नई दिल्ली")

    # Relational queries
    res = store.query(subject="India", predicate="capital")
    assert len(res) == 1 and res[0][2] == "New Delhi"

    # Natural language parsing
    hit = store.query_natural("What is the capital of India?")
    assert hit is not None and hit[2] == "New Delhi"

    # Indic Sanskrit query
    hit_sanskrit = store.query_natural("Who is the author of Sanskrit grammar?")
    assert hit_sanskrit is not None and hit_sanskrit[2] == "Panini"

    # Save and load roundtrip
    p = tmp_path / "triples.json"
    store.save(p)
    loaded = TripleStore.load(p)
    assert len(loaded) == 4
    assert loaded.query("India", "currency")[0][2] == "Rupee"
