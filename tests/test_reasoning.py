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
    from Tantra.model.ngram_router import route_probs_to_experts, train_ngram_router
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
