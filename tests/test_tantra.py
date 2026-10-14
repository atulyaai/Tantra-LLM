import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Tantra.config import ModelConfig, estimate_params, get_preset, load_taxonomy, validate_taxonomy
from Tantra.data.expert_dataset import EncodedDataset, read_jsonl
from Tantra.data.seed_conversation import generate, generate_mixed, write_all
from Tantra.data.seed_knowledge import write_all as write_knowledge
from Tantra.model.tantra_model import TantraModel, quantize_for_cpu
from Tantra.tokenizer import TantraTokenizer
from Tantra.training.tantra_trainer import TantraTrainer, TrainArgs

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
    from Tantra.inference.inference_engine import TantraEngine
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
