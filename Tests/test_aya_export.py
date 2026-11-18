import json
import pytest
from Tantra.aya_export import export_aya_training_rows

def test_aya_export_keeps_only_english_hindi_training_rows_and_is_pending_input(tmp_path):
    output=tmp_path/"aya.jsonl"
    rows=[
        {"language_code":"eng","inputs":"Explain gravity.","targets":"Gravity attracts masses.","annotation_type":"original_annotations"},
        {"language_code":"hin","inputs":"गुरुत्वाकर्षण क्या है?","targets":"यह वस्तुओं के बीच आकर्षण है।","annotation_type":"original_annotations"},
        {"language_code":"fra","inputs":"Bonjour","targets":"Salut"},
        {"language_code":"eng","inputs":"","targets":"missing prompt"},
    ]
    result=export_aya_training_rows(rows,output)
    records=[json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["rows"]==2
    assert result["by_language"]=={"en":1,"hi":1}
    assert result["training_eligible"] is False
    assert result["output_sha256"]
    manifest=json.loads((tmp_path/"aya.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_split"]=="train" and manifest["revision"]
    assert {row["language"] for row in records}=={"en","hi"}
    assert {row["expert"] for row in records}=={"lang_en","lang_hi"}
    assert {row["domain"] for row in records}=={"multilingual"}
    assert records[0]["family_id"] != records[1]["family_id"]

def test_aya_export_refuses_test_split_and_overwrite(tmp_path):
    with pytest.raises(ValueError,match="only Aya train"):
        export_aya_training_rows([],tmp_path/"test.jsonl",split="test")
    path=tmp_path/"existing.jsonl"; path.write_text("keep")
    with pytest.raises(ValueError,match="overwrite"):
        export_aya_training_rows([],path)

def test_aya_export_rejects_empty_supported_subset(tmp_path):
    with pytest.raises(ValueError,match="no supported"):
        export_aya_training_rows([{"language_code":"tam","inputs":"வணக்கம்","targets":"வணக்கம்"}],tmp_path/"empty.jsonl")
