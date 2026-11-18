import json
import pytest
from Tantra.multilingual_data import approve_review_queue, build_review_queue

SOURCE={"id":"sample-open-corpus","revision":"commit-123","license":"CC-BY-4.0","url":"https://example.org/dataset"}

def write_jsonl(path, values):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(v,ensure_ascii=False)+"\n" for v in values),encoding="utf-8")

def test_intake_pending_and_quarantine(tmp_path):
    src=tmp_path/"source.jsonl"
    write_jsonl(src,[
        {"id":"1","family_id":"conv:1","language":"hi","user":"आप कैसे हैं?","assistant":"मैं ठीक हूँ, धन्यवाद।"},
        {"id":"2","family_id":"conv:2","language":"hinglish","user":"Aaj kya plan hai?","assistant":"Abhi decide nahi kiya."},
        {"id":"3","family_id":"conv:3","language":"hi","user":"How are you?","assistant":"I am fine."},
        {"id":"4","family_id":"conv:4","language":"fr","user":"Bonjour","assistant":"Salut"}])
    out=tmp_path/"queue"; report=build_review_queue(src,out,expert="smalltalk",source=SOURCE)
    candidates=[json.loads(x) for x in (out/"candidates.jsonl").read_text().splitlines()]
    assert report["pending_review_rows"]==2 and report["quarantined_rows"]==2
    assert all(x["review"]["status"]=="pending" for x in candidates)
    assert all(x["source"]["url"]==SOURCE["url"] for x in candidates)
    assert report["training_ready"] is False

def test_intake_requires_license_and_clean_output(tmp_path):
    src=tmp_path/"source.jsonl"; write_jsonl(src,[])
    with pytest.raises(ValueError,match="license"):
        build_review_queue(src,tmp_path/"q",expert="smalltalk",source={**SOURCE,"license":"unknown"})
    out=tmp_path/"existing"; out.mkdir(); (out/"keep").write_text("x")
    with pytest.raises(ValueError,match="new or empty"):
        build_review_queue(src,out,expert="smalltalk",source=SOURCE)

def test_approval_requires_full_review_and_native_language_check(tmp_path):
    src=tmp_path/"source.jsonl"
    write_jsonl(src,[
        {"id":"hi1","family_id":"hi:1","language":"hi","user":"आप कैसे हैं?","assistant":"मैं ठीक हूँ।"},
        {"id":"en1","family_id":"en:1","language":"en","user":"How are you?","assistant":"I am fine."}])
    q=tmp_path/"queue"; build_review_queue(src,q,expert="smalltalk",source=SOURCE)
    records=[json.loads(x) for x in (q/"candidates.jsonl").read_text().splitlines()]
    decisions=tmp_path/"decisions.jsonl"
    write_jsonl(decisions,[{"record_id":r["record_id"],"decision":"approve","reviewer":"reviewer","note":"checked","fluent_language_review":False} for r in records])
    with pytest.raises(ValueError,match="fluent_language_review"): approve_review_queue(q,decisions,tmp_path/"incoming")
    write_jsonl(decisions,[{"record_id":r["record_id"],"decision":"approve","reviewer":"reviewer","note":"checked","fluent_language_review":r["language"]=="hi"} for r in records])
    result=approve_review_queue(q,decisions,tmp_path/"incoming")
    assert result["approved"]==2
    final=[json.loads(x) for x in (tmp_path/"incoming"/"smalltalk"/"multilingual-reviewed.jsonl").read_text().splitlines()]
    assert {r["review"]["status"] for r in final}=={"reviewed"}

def test_approval_rejects_missing_duplicate_decisions(tmp_path):
    src=tmp_path/"source.jsonl"; write_jsonl(src,[{"id":"1","family_id":"f1","language":"en","user":"Hi there","assistant":"Hello there"}])
    q=tmp_path/"queue"; build_review_queue(src,q,expert="greeting",source=SOURCE)
    row=json.loads((q/"candidates.jsonl").read_text()); decisions=tmp_path/"decisions.jsonl"
    write_jsonl(decisions,[])
    with pytest.raises(ValueError,match="missing"): approve_review_queue(q,decisions,tmp_path/"incoming")
    decision={"record_id":row["record_id"],"decision":"approve","reviewer":"R","note":"ok"}
    write_jsonl(decisions,[decision,decision])
    with pytest.raises(ValueError,match="duplicate"): approve_review_queue(q,decisions,tmp_path/"incoming")
