import json
import pytest
from Tantra.hinglish_export import export_hinglish_rows

def test_export_hinglish_preserves_scenario_persona_families_and_provenance(tmp_path):
    rows=[
        {"user_message":"Yaar, aaj kya plan hai?","assistant_response":"Abhi decide nahi kiya, bhai.","category":"casual","script":"roman","scenario":"weekend plan","persona":"friend"},
        {"user_message":"Bhai, weekend ka kya scene?","assistant_response":"Chalo movie dekhte hain, yaar.","category":"casual","script":"roman","scenario":"weekend plan","persona":"friend"},
        {"user_message":"Explain the database index.","assistant_response":"A database index speeds up queries.","category":"question","script":"roman","scenario":"database","persona":"engineer"},
        {"user_message":"आप कैसे हैं?","assistant_response":"मैं ठीक हूँ।","category":"casual","script":"devanagari","scenario":"greeting","persona":"friend"},
    ]
    out=tmp_path/"hinglish.jsonl"
    result=export_hinglish_rows(rows,out)
    records=[json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert result["rows"]==2
    assert result["synthetic"] is True and result["training_eligible"] is False
    assert records[0]["family_id"]==records[1]["family_id"]
    assert {row["expert"] for row in records}=={"smalltalk"}
    assert {row["domain"] for row in records}=={"conversation"}
    manifest=json.loads((tmp_path/"hinglish.jsonl.manifest.json").read_text())
    assert manifest["revision"] and manifest["output_sha256"]

def test_export_hinglish_refuses_non_train_split_and_overwrite(tmp_path):
    with pytest.raises(ValueError,match="only source train"):
        export_hinglish_rows([],tmp_path/"test.jsonl",split="test")
    path=tmp_path/"existing.jsonl"; path.write_text("keep")
    with pytest.raises(ValueError,match="overwrite"):
        export_hinglish_rows([],path)

def test_export_hinglish_rejects_empty_subset(tmp_path):
    with pytest.raises(ValueError,match="no usable"):
        export_hinglish_rows([{"user_message":"Hello","assistant_response":"Hi","scenario":"x","persona":"y","script":"roman"}],tmp_path/"empty.jsonl")


def test_hinglish_sample_cap_keeps_whole_scenario_persona_families(tmp_path):
    rows=[
        {"user_message":"Yaar, aaj kya plan hai?","assistant_response":"Abhi decide nahi kiya, bhai.","category":"casual","script":"roman","scenario":"same scenario","persona":"friend"},
        {"user_message":"Bhai, weekend ka kya scene?","assistant_response":"Chalo movie dekhte hain, yaar.","category":"casual","script":"roman","scenario":"same scenario","persona":"friend"},
        {"user_message":"Kya haal hai bhai?","assistant_response":"Main theek hoon yaar.","category":"casual","script":"roman","scenario":"other scenario","persona":"friend"},
    ]
    result=export_hinglish_rows(rows,tmp_path/"sample.jsonl",max_rows_per_expert=1)
    records=[json.loads(x) for x in (tmp_path/"sample.jsonl").read_text(encoding="utf-8").splitlines()]
    original_counts={}
    for row in rows:
        family=f"hinglish-scenario:{row['scenario'].casefold()}:{row['persona'].casefold()}"
        original_counts[family]=original_counts.get(family,0)+1
    for family in {row["family_id"] for row in records}:
        assert sum(row["family_id"]==family for row in records)==original_counts[family]
    assert result["max_rows_per_expert"]==1
