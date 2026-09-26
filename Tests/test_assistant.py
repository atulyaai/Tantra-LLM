"""Skills, personal memory, reminders, taught answers, documents and the assistant API."""
import datetime as dt
import json

from fastapi.testclient import TestClient

import WebUI.server as server
from Tantra.documents import Documents, extract_text
from Tantra.memory import Memory
from Tantra.skills import handle
from Tantra.smriti import Smriti, build as build_smriti


def test_calculator_units_time():
    assert handle("250 × 18 + 5%").card["result"] == "4,725"
    assert handle("3*120 + 2*180").text.endswith("720")
    assert handle("15% of 2400").card["result"] == "360"
    assert "0.5 million" in handle("5 lakh in million").text
    assert "37" in handle("98.6 f in c").text
    now = dt.datetime(2026, 3, 14, 21, 5)
    assert "रात के 9:05" in handle("कितने बजे हैं?", {"now": now}).text
    assert "Saturday" in handle("what is the date today", {"now": now}).text
    assert handle("hello") is None and handle("भारत की राजधानी क्या है?") is None


def test_memory_reminders_and_teaching(tmp_path):
    m = Memory(str(tmp_path / "memory.json"))
    ctx = {"memory": m, "now": dt.datetime(2026, 3, 14, 10, 0)}
    r = handle("याद रखो: मेरी बेटी का जन्मदिन 14 मार्च को है", ctx)
    assert r.name == "memory" and m.list()[0]["category"] == "family"
    assert m.relevant("बेटी का जन्मदिन कब है")[0]["text"].startswith("मेरी बेटी")
    rem = handle("10 मिनट बाद याद दिलाना चाय बनानी है", ctx)
    assert rem.name == "reminder" and rem.card["reminder"]["text"] == "चाय बनानी है"
    assert handle("remind me tomorrow at 9 am to send the report", ctx).card["reminder"]["text"] == "send the report"
    m.data["reminders"][0]["due"] = 0            # the first is due now, the second far in the future
    m.data["reminders"][1]["due"] = 4e9
    assert [x["text"] for x in m.due()] == ["चाय बनानी है"] and m.due() == []   # announced once
    assert handle("भूल जाओ बेटी का जन्मदिन", ctx).name == "memory" and m.list() == []
    m.teach("तुम्हारा नाम क्या है?", "मेरा नाम Tantra है।")
    assert m.taught_answer("तुम्हारा नाम क्या है")["answer"] == "मेरा नाम Tantra है।"
    assert Memory(str(tmp_path / "memory.json")).data["taught"]                   # saved to disk


def test_best_answer_is_strict(tmp_path):
    rows = [{"messages": [{"role": "user", "content": "भारत की राजधानी क्या है?"},
                          {"role": "assistant", "content": "नई दिल्ली भारत की राजधानी है।"}]},
            {"messages": [{"role": "user", "content": "Translate this to Hindi:\nThe capital city is big."},
                          {"role": "assistant", "content": "राजधानी बड़ी है।"}]},
            {"text": "दिल्ली एक बड़ा शहर है। यहाँ बहुत लोग रहते हैं और राजधानी की इमारतें हैं।"}]
    data = tmp_path / "k.jsonl"
    data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), "utf-8")
    build_smriti([str(data)], str(tmp_path / "k.db"))
    s = Smriti(str(tmp_path / "k.db"), readonly=True)
    assert s.best_answer("भारत की राजधानी बताओ")["text"] == "नई दिल्ली भारत की राजधानी है।"
    assert s.best_answer("hello kaise ho") is None                     # too little to look up
    assert s.best_answer("दिल्ली में कितने लोग रहते हैं") is None       # loose article overlap: not direct
    t = "Lunch is at 1:30 pm. The meeting is on Friday in room 204."
    assert Smriti.best_sentences(t, "when is the meeting?", 1) == "The meeting is on Friday in room 204."


def test_documents_roundtrip(tmp_path):
    d = Documents(str(tmp_path / "docs.db"))
    item = d.add("policy.txt", "The monthly team meeting is on the first Friday in room 204.".encode())
    assert item["pieces"] == 1 and d.list()[0]["name"] == "policy.txt"
    assert "204" in d.search("team meeting room")[0]["text"]
    d.remove("policy.txt")
    assert d.list() == [] and d.search("team meeting") == []
    assert "Hello" in extract_text("a.html", b"<html><script>x()</script><p>Hello</p></html>")


def test_assistant_api(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(server, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(server, "CHATS_FILE", str(tmp_path / "chats.json"))
    server.state.update(memory=None, model=None)
    c = TestClient(server.app)
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "12 guna 12"}]}).json()
    assert r["skill"] == "calculator" and "144" in r["choices"][0]["message"]["content"]   # works with no model at all
    s = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "याद रखो: मुझे चाय पसंद है"}], "stream": True})
    assert '"skill": "memory"' in s.text and s.text.strip().endswith("[DONE]")
    assert c.get("/api/memory").json()["memories"][0]["text"] == "मुझे चाय पसंद है"
    c.post("/api/feedback", json={"question": "Tantra kaun hai?", "bad": "x", "correct": "Main Tantra hoon."})
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Tantra kaun hai?"}]}).json()
    assert r["skill"] == "taught" and (tmp_path / "feedback.jsonl").exists()
    up = c.post("/api/docs", files={"file": ("notes.txt", "Room 204 has the projector for meetings.".encode(), "text/plain")})
    assert up.status_code == 200 and c.get("/api/docs").json()["docs"][0]["name"] == "notes.txt"
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "which room has the projector?"}],
                                             "knowledge_first": True}).json()
    assert r["skill"] == "knowledge" and "204" in r["choices"][0]["message"]["content"]
    assert c.post("/api/code/run", json={"code": "print(6*7)"}).json()["stdout"].strip() == "42"
    assert c.post("/api/open", json={"name": "not-an-app"}).status_code == 404


def test_money_date_word_tools_follow_the_region():
    from Tantra.skills import set_region
    now = {"now": dt.datetime(2026, 9, 27, 10, 0)}
    try:
        set_region("IN")
        assert handle("EMI for 5 lakh at 9% for 3 years", now).text.startswith("EMI: ₹15,899.87")
        assert handle("simple interest on 50000 at 7% for 2 years", now).card["result"] == "₹57,000"
        assert "₹450" in handle("18% GST on 2500", now).text
        assert "Base ₹2,500" in handle("2950 including 18% GST", now).text
        assert handle("20% discount on 1499", now).card["result"] == "₹1,199.2"
        assert handle("days between 1 Jan 2026 and 26 Jan 2026", now).card["result"] == "25 days"
        assert "शनिवार" in handle("15 August 2026 ko kaun sa din hai", now).text
        assert handle("how many days until 25 December", now).card["result"] == "89 days"
        assert handle("12,34,567 in words", now).text.startswith("बारह लाख चौंतीस हज़ार पाँच सौ सड़सठ")
        assert "normal" in handle("BMI 70 kg 175 cm", now).text
        assert handle("2 acre in bigha", now).text.endswith("3.2 bigha")
        set_region("US")
        assert handle("tax on 2500", now).text == "sales tax $200, total $2,700"
        assert "One million two hundred" in handle("1234567 in words", now).text
        assert "4 Mar 2026" in handle("03/04/2026 ko kaun sa din hai", now).text
        set_region("GB")
        assert handle("tax on 2500", now).text.startswith("VAT £500")
    finally:
        set_region("IN")
