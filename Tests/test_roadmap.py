"""Data quality, Muon / WSD, shared-expert MoE, retrieval and distillation data."""
import json

import pytest
import torch

from Tantra.config import NeuroCoreConfig
from Tantra.data_prep import foreign_assistant
from Tantra.dataset import JSONLDataset
from Tantra.model import NeuroCoreModel
from Tantra.train import Muon, NeuroTrainer, lr_at, newton_schulz


# ── refusal / other-assistant filter ──

@pytest.mark.parametrize("text", [
    "I'm sorry, I am an AI language model and I am not able to provide information about that.",
    "As an AI, I do not have physical sensations.",
    "I was trained by OpenAI.",
    "I'm sorry, but I can't help with that.",
    "I am ChatGPT.",
    "मैं एक AI भाषा मॉडल हूँ और मेरे पास भौतिक शरीर नहीं है।",
    "मुझे खेद है, लेकिन मैं यह नहीं कर सकता।",
])
def test_filter_catches_refusals(text):
    assert foreign_assistant(text)


@pytest.mark.parametrize("text", [
    "I am Tantra, an AI assistant created by Atulya AI.",
    "I am an AI assistant, not a human. My name is Tantra and I was created by Atulya AI.",
    "भारत की राजधानी नई दिल्ली है।",
    "Gemini is a zodiac sign.", "The llama lives in the Andes.", "Google was founded in 1998.",
])
def test_filter_keeps_normal_answers(text):
    assert not foreign_assistant(text)


def test_boost_drops_refusals_from_base(tmp_path):
    from Tantra.data_boost import boost
    rows = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "As an AI language model, I can't."}]},
            {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "The answer is 4."}]}]
    (tmp_path / "sft.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    c = boost(str(tmp_path), identity=10, use_hf=False)
    assert c["dropped_refusals"] == 1 and c["base"] == 1 and c["identity"] == 10
    text = (tmp_path / "sft.jsonl").read_text(encoding="utf-8")
    assert "As an AI" not in text and "The answer is 4." in text


# ── knowledge pretraining data ──

HINDI_FACTS = (
    "प्रकाश संश्लेषण वह प्रक्रिया है जिसमें हरे पौधे सूर्य के प्रकाश की ऊर्जा से भोजन बनाते हैं। "
    "इस क्रिया में कार्बन डाइऑक्साइड और जल मिलकर ग्लूकोज बनाते हैं तथा ऑक्सीजन बाहर निकलती है। "
    "पत्तियों में पाया जाने वाला क्लोरोफिल प्रकाश को सोखता है, इसी कारण पत्तियाँ हरी दिखाई देती हैं। "
    "वैज्ञानिक यान इंगेनहाउस ने वर्ष 1779 में सिद्ध किया कि पौधों को इसके लिए प्रकाश आवश्यक है। "
    "पृथ्वी के वायुमंडल की लगभग इक्कीस प्रतिशत ऑक्सीजन इसी प्रक्रिया की देन मानी जाती है। "
    "समुद्र में रहने वाले सूक्ष्म शैवाल भी बड़ी मात्रा में यही क्रिया करते हैं। "
    "इस प्रक्रिया के दो चरण होते हैं: प्रकाश अभिक्रिया और अंधकार अभिक्रिया, जिसे केल्विन चक्र कहते हैं। "
    "मेल्विन केल्विन को इस चक्र की खोज के लिए 1961 में रसायन का नोबेल पुरस्कार मिला था। "
    "कृषि में फसलों की उपज इसी प्रक्रिया की दक्षता पर निर्भर करती है, इसलिए इसका अध्ययन महत्वपूर्ण है। "
)


def test_hindi_edu_score_prefers_explanations_over_gossip():
    from Tantra.knowledge_data import hindi_edu_score
    gossip = ("बॉलीवुड की खूबसूरत तस्वीरें सोशल मीडिया पर वायरल हो रही हैं। अभिनेत्री ने कहा कि उनकी नई फिल्म "
              "बॉक्स ऑफिस पर धमाल मचाएगी। प्रशंसकों ने सेलेब के नए लुक की तारीफ की। देखें वीडियो और हमें फॉलो करें। "
              "सूत्रों के अनुसार शादी की तैयारी चल रही है और जल्द ही तारीख का ऐलान हो सकता है। ") * 2
    assert hindi_edu_score(HINDI_FACTS) >= 4 > hindi_edu_score(gossip)


def test_knowledge_build_filters_dedups_and_repeats_wiki(tmp_path):
    from Tantra import knowledge_data as kd
    article = ("Photosynthesis is the process by which green plants use sunlight to turn water and carbon dioxide "
               "into sugar. Chlorophyll in the leaves absorbs mostly red and blue light, which is why leaves look "
               "green. Jan Ingenhousz showed in 1779 that plants need light for this. The Calvin cycle, found by "
               "Melvin Calvin, fixes carbon in the second stage. Most oxygen in the air comes from this process, "
               "much of it from tiny algae in the oceans rather than from land plants.")
    fake = {
        "HuggingFaceFW/fineweb-edu": [{"text": article, "int_score": 4}, {"text": "buy now " * 100, "int_score": 1},
                                      {"text": article, "int_score": 4}],
        "wikimedia/wikipedia": [{"title": "प्रकाश संश्लेषण", "text": HINDI_FACTS}],
        "ai4bharat/sangraha": [{"text": "बॉलीवुड वायरल " * 100}],
    }
    r = kd.build(str(tmp_path), rows_fn=lambda repo, kw: iter(fake[repo]))
    assert r["fineweb_edu_en"]["kept"] == 1          # low score dropped, duplicate dropped
    assert r["sangraha_hi"]["kept"] == 0
    lines = (tmp_path / "pretrain_knowledge.jsonl").read_text(encoding="utf-8").splitlines()
    assert sum("क्लोरोफिल" in l for l in lines) == 2  # Wikipedia written twice (wiki_en sees a duplicate)


def test_dataset_interleaves_files_by_size(tmp_path):
    from Tantra.tokenizer import build_tokenizer, load_tokenizer
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("\n".join(json.dumps({"text": f"alpha {i}"}) for i in range(300)), encoding="utf-8")
    b.write_text("\n".join(json.dumps({"text": f"beta {i}"}) for i in range(300)), encoding="utf-8")
    tok = load_tokenizer(build_tokenizer([str(a), str(b)], str(tmp_path), vocab_size=300, max_mb=1))
    ds = JSONLDataset([str(a), str(b)], tok, seq_len=16, stage="pretrain", shuffle_buffer=1, loop=False)
    first = [l for _, l in zip(range(100), ds._mixed(__import__("random").Random(0)))]
    assert 20 < sum("beta" in l for l in first) < 80        # both files from the start, not one after the other
    assert len(list(ds._mixed(__import__("random").Random(0)))) == 600


# ── Muon + warmup-stable-decay ──

def test_newton_schulz_orthogonalises():
    g = torch.randn(64, 32)
    x = newton_schulz(g)
    s = torch.linalg.svdvals(x)
    assert 0.5 < float(s.min()) and float(s.max()) < 1.3      # singular values pushed towards 1


def test_wsd_schedule_is_flat_then_decays():
    assert lr_at(1000, 1e-3, 100, 10000, schedule="wsd") == pytest.approx(1e-3)
    assert lr_at(7999, 1e-3, 100, 10000, schedule="wsd") == pytest.approx(1e-3)
    assert lr_at(9000, 1e-3, 100, 10000, schedule="wsd") < 1e-3
    assert lr_at(10000, 1e-3, 100, 10000, schedule="wsd") == pytest.approx(1e-4)


def _tiny(vocab=300):
    cfg = NeuroCoreConfig.tiny()
    cfg.vocab.vocab_size = cfg.vocab.byte_bpe_vocab = vocab
    torch.manual_seed(0)
    return NeuroCoreModel(cfg, use_mtp=False)


def test_muon_trains_and_resumes(tmp_path):
    torch.manual_seed(0)
    x = torch.randint(0, 300, (8, 33))
    tr = NeuroTrainer(_tiny(), lr=3e-3, optimizer_name="muon", warmup_steps=2, total_steps=40, schedule="wsd")
    assert isinstance(tr.optimizer, Muon)
    muon_group = next(g for g in tr.optimizer.param_groups if g["use_muon"])
    assert muon_group["lr"] > 10 * tr.lr * 0.05                      # its own (larger) peak
    first = tr.train_step(x[:, :-1], x[:, 1:])["loss"]
    for _ in range(30):
        last = tr.train_step(x[:, :-1], x[:, 1:])["loss"]
    assert last < first - 1.0                                        # memorises the batch
    ck = tmp_path / "m.pt"
    tr.save_checkpoint(str(ck))
    tr2 = NeuroTrainer(_tiny(), lr=3e-3, optimizer_name="muon", warmup_steps=2, total_steps=40, schedule="wsd")
    tr2.load_checkpoint(str(ck))
    assert tr2.optimizer.state_dict()["state"] and tr2.step_count == tr.step_count


# ── shared-expert MoE ──

def test_shared_expert_moe_runs_and_routes():
    cfg = NeuroCoreConfig.tiny()
    cfg.vocab.vocab_size = cfg.vocab.byte_bpe_vocab = 300
    cfg.moe.num_experts, cfg.moe.top_k, cfg.moe.expert_expansion, cfg.moe.shared_expansion = 8, 2, 1, 2
    cfg.moe.real_top1 = True
    m = NeuroCoreModel(cfg, use_mtp=False, use_moe=True)
    mlp = m.layers[0].mlp
    assert mlp.shared is not None and len(mlp.experts) == 8
    logits, _ = m(torch.randint(0, 300, (2, 10)))
    logits.mean().backward()
    assert mlp.shared.w_up.weight.grad is not None


# ── retrieval (RAG) data ──

def test_rag_example_uses_the_webui_format(tmp_path):
    from Tantra.rag_data import build
    from Tantra.smriti import KNOWLEDGE_HEADER, build as build_smriti
    rows = [{"messages": [{"role": "user", "content": f"भारत के राज्य {i} की राजधानी क्या है?"},
                          {"role": "assistant", "content": f"राज्य {i} की राजधानी नगर {i} है।"}]} for i in range(30)]
    rows.append({"messages": [{"role": "user", "content": "एक कविता लिखें"}, {"role": "assistant", "content": "कविता की पंक्ति यहाँ है।"}]})
    src = tmp_path / "facts.jsonl"
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    db = str(tmp_path / "s.db")
    build_smriti([str(src)], db)
    stats = build(str(tmp_path), db, n=10, workers=1)
    assert stats["with_fact"] + stats["distractors_only"] == 10
    for line in (tmp_path / "sft_rag.jsonl").read_text(encoding="utf-8").splitlines():
        sys_msg, user, answer = json.loads(line)["messages"]
        assert sys_msg["role"] == "system" and sys_msg["content"].startswith(KNOWLEDGE_HEADER)
        assert "कविता" not in user["content"]                      # creative requests are not fact questions


# ── distillation ──

def test_distill_parses_checks_and_writes(tmp_path):
    from Tantra.distill import build, parse_pairs
    good = json.dumps([{"q": "गंगा नदी कहाँ से निकलती है?", "a": "गंगा नदी गंगोत्री हिमनद से निकलती है।"},
                       {"q": "गंगा कितनी लंबी है?", "a": "लेख के अनुसार गंगा 2525 किलोमीटर लंबी है।"},   # 'according to'
                       {"q": "Who are you?", "a": "As an AI language model I cannot say."}], ensure_ascii=False)
    pairs = parse_pairs("Sure!\n" + good, "hindi", set())
    assert [p["q"] for p in pairs] == ["गंगा नदी कहाँ से निकलती है?"]
    assert parse_pairs("no json here", "hindi", set()) == []
    passage = "गंगा भारत की सबसे लंबी नदी है। यह गंगोत्री हिमनद से निकलती है और बंगाल की खाड़ी में गिरती है। " * 6
    stats = build(str(tmp_path), n=3, batch_size=2, generate=lambda ps: [good] * len(ps), source=iter([passage] * 10))
    assert stats["pairs"] >= 3
    rec = json.loads((tmp_path / "sft_distill.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["messages"][1]["role"] == "assistant"


def test_distill_own_questions(tmp_path):
    from Tantra.distill import build, load_questions
    qf = tmp_path / "q.txt"
    qf.write_text("What is the capital of France?\n\nWhat is the capital of France?\n"
                  '{"question": "What is the capital of Japan?"}\n', encoding="utf-8")
    assert load_questions(str(qf)) == ["What is the capital of France?", "What is the capital of Japan?"]
    asked = []
    gen = lambda ps: asked.extend(ps) or ["The capital city is a well known place."] * len(ps)
    stats = build(str(tmp_path), n=10, batch_size=1, generate=gen, questions=str(qf))
    assert stats["pairs"] == 2 and len(asked) == 2
    rec = json.loads((tmp_path / "sft_distill.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["messages"][0]["content"] == "What is the capital of France?"
    build(str(tmp_path), n=10, batch_size=1, generate=gen, questions=str(qf))     # resume: nothing asked twice
    assert len(asked) == 2


def test_distill_http_teacher_talks_openai_api(tmp_path):
    """The teacher can be any OpenAI-compatible server (Colibri `coli serve`, llama.cpp, Ollama)."""
    import http.server, threading
    from Tantra.distill import HTTPTeacher, build
    seen, gets = [], []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            gets.append(self.path)
            if self.path != "/v1/models":                      # like Colibri: anything else is a 404
                self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()
                return
            data = json.dumps({"data": [{"id": "olmoe-served"}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body["model"], self.headers.get("Authorization")))
            reply = json.dumps([{"q": "What is the capital of France?", "a": "The capital of France is the city of Paris."}])
            data = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/v1"
        assert HTTPTeacher(url, "olmoe", api_key="k")(["hi"])[0].startswith("[")
        assert seen[0] == ("/v1/chat/completions", "olmoe", "Bearer k")
        auto = HTTPTeacher(url)                                  # no model name -> asked from GET /models
        assert auto.model == "olmoe-served" and gets == ["/v1/models"]
        stats = build(str(tmp_path), n=1, batch_size=1, per_passage=1, teacher_url=url, teacher_model="olmoe",
                      source=iter(["Paris is the capital and largest city of France. " * 12]))
        assert stats["pairs"] == 1
        assert HTTPTeacher("http://127.0.0.1:9/v1", timeout=1, retries=1, backoff=0)(["x"]) == [""]      # one failure -> skipped, no crash
        import pytest
        dead = HTTPTeacher("http://127.0.0.1:9/v1", timeout=1, retries=1, backoff=0)
        with pytest.raises(RuntimeError, match="in a row"):                         # a dead server stops the run
            for _ in range(6):
                dead(["x"])
        hits = []

        class Flaky(H):
            def do_POST(self):
                hits.append(1)
                if len(hits) < 3:                                  # two 503s, then a good answer
                    self.send_response(503); self.send_header("Content-Length", "0"); self.end_headers()
                else:
                    super().do_POST()
        srv2 = http.server.HTTPServer(("127.0.0.1", 0), Flaky)
        threading.Thread(target=srv2.serve_forever, daemon=True).start()
        try:
            got = HTTPTeacher(f"http://127.0.0.1:{srv2.server_port}/v1", "m", retries=2, backoff=0)(["hi"])[0]
            assert got.startswith("[") and len(hits) == 3
        finally:
            srv2.shutdown()
    finally:
        srv.shutdown()
