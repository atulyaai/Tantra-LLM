"""Tokenizer, data pipeline, training, checkpoints, evaluation."""
import json

import pytest
import torch

from Tantra.config import EOS_ID, NeuroCoreConfig, SPECIAL_TOKENS
from Tantra.dataset import IGNORE_INDEX, DPODataset, JSONLDataset, chat_prompt, item_segments
from Tantra.eval_suite import load_probe, run_probe, validation_metrics
from Tantra.model import NeuroCoreModel
from Tantra.tokenizer import build_tokenizer, load_tokenizer
from Tantra.train import NeuroTrainer, lr_at

ROWS = [
    {"messages": [{"role": "user", "content": "तुम्हें किसने बनाया?"}, {"role": "assistant", "content": "मुझे अतुल्य AI ने बनाया।"}]},
    {"messages": [{"role": "user", "content": "भारत की राजधानी क्या है?"}, {"role": "assistant", "content": "भारत की राजधानी नई दिल्ली है।"}]},
    {"user": "What is 2+2?", "assistant": "2+2 is 4."},
    {"text": "गंगा भारत की सबसे लंबी नदी है। हिमालय उत्तर में है।"},
    {"messages": [{"role": "user", "content": "इसके बारे में विस्तृत जानकारी दें:"},
                  {"role": "assistant", "content": "लेख " * 150}]},
]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    path = d / "train.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ROWS * 20), encoding="utf-8")
    tok = load_tokenizer(build_tokenizer([str(path)], str(d), vocab_size=400, max_mb=1))
    return path, tok


def test_tokenizer_special_ids_and_roundtrip(data):
    _, tok = data
    for token, idx in SPECIAL_TOKENS.items():
        assert tok.bpe.token_to_id(token) == idx
    s = "भारत की राजधानी नई दिल्ली है। Hello!"
    assert tok.decode(tok.encode(s)) == s


def test_segments_mask_questions_and_convert_generic_prompts():
    segs = item_segments(ROWS[0])
    learned = "".join(t for t, keep in segs if keep)
    assert "अतुल्य" in learned and "किसने" not in learned and segs[-1] == ("</s>", True)
    assert item_segments(ROWS[4]) == [(ROWS[4]["messages"][1]["content"].strip(), True)]
    assert all(keep for _, keep in item_segments(ROWS[0], stage="pretrain"))
    assert chat_prompt("Hi").endswith("<|user|>\nHi\n\n<|assistant|>\n")


def test_dataset_packs_full_windows(data):
    path, tok = data
    ds = JSONLDataset(str(path), tok, seq_len=32, loop=False, shuffle_buffer=1)
    batches = list(ds)
    assert batches and all(x.shape == (32,) and y.shape == (32,) for x, y in batches)
    x, y = batches[0]
    assert torch.equal(x[1:], torch.where(y[:-1] == IGNORE_INDEX, x[1:], y[:-1]))
    assert (torch.cat([y for _, y in batches]) == EOS_ID).any()


def _trainer(tok, **kw):
    cfg = NeuroCoreConfig.tiny()
    cfg.vocab.vocab_size = tok.vocab_size
    torch.manual_seed(0)
    return NeuroTrainer(NeuroCoreModel(cfg, use_mtp=False), lr=3e-3, warmup_steps=5, total_steps=80, **kw)


def test_training_actually_learns_and_resumes_exactly(data, tmp_path):
    path, tok = data
    tr = _trainer(tok, grad_accumulation_steps=2)
    loader = torch.utils.data.DataLoader(JSONLDataset(str(path), tok, seq_len=32), batch_size=4)
    val = torch.utils.data.DataLoader(JSONLDataset(str(path), tok, seq_len=32, loop=False, shuffle_buffer=1), batch_size=4)
    before = validation_metrics(tr.model, val)["loss"]
    tr.fit(loader, max_steps=60, log_every=1000, eval_every=0)
    after = validation_metrics(tr.model, val)["loss"]
    assert tr.step_count == 60 and after < before - 1.0

    ck = tmp_path / "ck.pt"
    tr.save_checkpoint(str(ck))
    tr2 = _trainer(tok, grad_accumulation_steps=2)
    tr2.load_checkpoint(str(ck))
    assert tr2.step_count == 60 and tr2.optimizer.param_groups[0]["lr"] == pytest.approx(lr_at(60, 3e-3, 5, 80))
    for a, b in zip(tr.model.state_dict().values(), tr2.model.state_dict().values()):
        assert torch.equal(a, b)  # full precision — nothing lost on resume
    assert tr2.optimizer.state_dict()["state"]


def test_probe_scores_and_logs(data, tmp_path):
    _, tok = data
    tr = _trainer(tok)
    probe = tmp_path / "p.jsonl"
    probe.write_text(json.dumps({"question": "भारत की राजधानी क्या है?", "keywords": ["दिल्ली"],
                                 "reference": "नई दिल्ली", "category": "gk"}, ensure_ascii=False), encoding="utf-8")
    hist = tmp_path / "h.jsonl"
    r = run_probe(tr.model, tok, load_probe(str(probe)), step=1, max_new_tokens=5, history_path=str(hist))
    assert r["n"] == 1 and "hit_rate" in r and hist.read_text(encoding="utf-8").strip()


def test_dpo_step_runs(data, tmp_path):
    _, tok = data
    prefs = tmp_path / "prefs.jsonl"
    prefs.write_text(json.dumps({"prompt": "नमस्ते", "chosen": "नमस्ते! बताइए।", "rejected": "पता नहीं"},
                                ensure_ascii=False), encoding="utf-8")
    tr = _trainer(tok)
    tr.train_dpo(torch.utils.data.DataLoader(DPODataset(str(prefs), tok, max_len=24), batch_size=1), max_steps=2)
    assert tr.step_count == 2
