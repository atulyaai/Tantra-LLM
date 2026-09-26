"""Data cleaning (Tantra/data_prep.py) and the Smriti knowledge store (Tantra/smriti.py)."""
import json
from collections import Counter

from Tantra.data_prep import build, clean_article, convert_master_row
from Tantra.smriti import Smriti, build as build_smriti, keywords


def _conv(*pairs):
    return {"messages": [{"role": r, "content": c} for r, c in pairs]}


# A realistic Hindi article (unique sentences) that won't trip the repetition filter.
_REALISTIC_ARTICLE = (
    "गंगा भारत की सबसे पवित्र और लंबी नदियों में से एक है। "
    "इसका उद्गम उत्तराखंड के गंगोत्री हिमनद से होता है। "
    "यह नदी उत्तर प्रदेश, बिहार और पश्चिम बंगाल से होकर बहती है। "
    "गंगा का जल लाखों लोगों की आजीविका का स्रोत है। "
    "इसके किनारे वाराणसी, इलाहाबाद और हरिद्वार जैसे प्रसिद्ध शहर बसे हैं। "
    "गंगा की सहायक नदियों में यमुना, गोमती, घाघरा और सोन प्रमुख हैं। "
    "गंगा डॉल्फिन इस नदी में पाई जाने वाली एक दुर्लभ प्रजाति है। "
    "नमामि गंगे योजना के तहत इस नदी की सफाई का कार्य चल रहा है। "
    "गंगा का धार्मिक महत्व हिंदू संस्कृति में अत्यधिक है। "
    "अंत में यह नदी बंगाल की खाड़ी में मिल जाती है।"
)


def test_filler_question_becomes_plain_text():
    out = list(convert_master_row(_conv(("user", "इसके बारे में विस्तृत जानकारी दें:"), ("assistant", _REALISTIC_ARTICLE)), Counter()))
    assert out and out[0][0] == "pretrain" and "गंगा" in out[0][1]["text"]


def test_shifted_roles_are_dropped_or_repaired():
    stats = Counter()
    broken = _conv(("user", "आप गणित के शिक्षक हैं।"), ("assistant", "26 और 46 का योग ज्ञात कीजिए।"))
    assert list(convert_master_row(broken, stats)) == []
    fixable = _conv(("user", "आप गणित के शिक्षक हैं।"), ("assistant", "2 और 3 का योग?"), ("user", "2 + 3 = 5"))
    (kind, rec), = convert_master_row(fixable, stats)
    assert kind == "sft" and rec["messages"][0] == {"role": "user", "content": "2 और 3 का योग?"}
    assert rec["messages"][1]["role"] == "assistant"
    assert stats["dropped: shifted roles, no answer"] == 1 and stats["fixed: shifted roles"] == 1


def test_boilerplate_removed():
    s = clean_article("डिजिटल डेस्क, नई दिल्ली: आज बारिश हुई।\nइस पृष्ठ को प्रिंट करें\nदेखें https://x.com/a")
    assert "प्रिंट" not in s and "http" not in s and "बारिश" in s


def test_build_splits_dedups_and_reports(tmp_path):
    rows = [_conv(("user", f"प्रश्न {i}: भारत की राजधानी?"), ("assistant", f"नई दिल्ली {i}")) for i in range(300)]
    rows += rows[:50]                                                       # duplicates
    rows += [_conv(("user", "इसके बारे में विस्तृत जानकारी दें:"),
                   ("assistant", f"लेख {i}: " + _REALISTIC_ARTICLE))
              for i in range(100)]
    (tmp_path / "master_train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), "utf-8")
    rep = build(str(tmp_path), raw_dir=str(tmp_path / "none"), val_fraction=0.05)
    assert rep["rows"]["sft"] + rep["val_rows"].get("sft", 0) == 300
    assert rep["rows"]["pretrain"] + rep["val_rows"].get("pretrain", 0) == 100
    assert rep["cleaning"]["dropped: duplicate"] == 50
    train = {l for l in (tmp_path / "sft.jsonl").read_text("utf-8").splitlines()}
    val = {l for l in (tmp_path / "val_sft.jsonl").read_text("utf-8").splitlines()}
    assert train and not (train & val)                                      # nothing leaks into validation


def test_smriti_finds_hindi_and_english_facts(tmp_path):
    data = tmp_path / "facts.jsonl"
    rows = [_conv(("user", "भारत की राजधानी क्या है?"), ("assistant", "भारत की राजधानी नई दिल्ली है।")),
            _conv(("user", "What is the capital of France?"), ("assistant", "The capital of France is Paris.")),
            {"text": "गंगा नदी हिमालय से निकलती है और बंगाल की खाड़ी में गिरती है। " * 3}]
    data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), "utf-8")
    stats = build_smriti([str(data)], str(tmp_path / "smriti.db"))
    assert stats["facts"] == 3
    s = Smriti(str(tmp_path / "smriti.db"), readonly=True)
    assert "दिल्ली" in s.search("भारत की राजधानी बताओ")[0]["text"]
    assert "Paris" in s.search("capital of france?")[0]["text"]
    ctx, hits = s.context("गंगा कहाँ से निकलती है")
    assert "हिमालय" in ctx and hits[0]["kind"] == "text"
    assert s.search("zzzz qqqq") == [] and keywords("क्या है?") == []
