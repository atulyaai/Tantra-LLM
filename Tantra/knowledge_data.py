"""
Tantra/knowledge_data.py — Knowledge-dense pretraining text (python main.py --mode knowledge).

Why: the probe shows the model learns identity and translation but almost no general knowledge
(GK 3/25). A small model only learns facts it has seen many times in clean, explanatory text, so
this builds Datasets/pretrain_knowledge.jsonl from sources chosen for that:

  fineweb_edu_en  HuggingFaceFW/fineweb-edu: English web pages already scored for educational
                  value by a classifier (the data behind SmolLM). Only int_score >= 3 is kept.
  wiki_hi         Hindi Wikipedia — every article, shown twice (facts need repetition).
  wiki_en         English Wikipedia — shown twice.
  sangraha_hi     ai4bharat/sangraha (verified Hindi web text). There is no Hindi edu classifier,
                  so hindi_edu_score() copies the idea with rules: explanatory, encyclopedic text
                  scores high; news, gossip, ads and repetitive text score low. Kept at >= 4.

Budgets are in millions of characters (--scale multiplies them all). Sources that cannot be
downloaded are skipped with a warning. Training mixes this file with pretrain.jsonl automatically
(by size), and `--mode pack` uploads it with the rest.
"""
from __future__ import annotations

import json
import os
import re
import zlib
from typing import Callable, Dict, Iterator, List, Optional

from Tantra.data_prep import DEV, _Buckets, _wiki_text, bad_text, clean_article, key, norm
from Tantra.utils import get_logger

log = get_logger("tantra.knowledge")

# name -> (repo, load_dataset kwargs, budget in million chars, times each text is written)
SOURCES: Dict[str, tuple] = {
    "fineweb_edu_en": ("HuggingFaceFW/fineweb-edu", {"name": "sample-10BT"}, 900, 1),
    "wiki_hi": ("wikimedia/wikipedia", {"name": "20231101.hi"}, 700, 2),
    "wiki_en": ("wikimedia/wikipedia", {"name": "20231101.en"}, 500, 2),
    "sangraha_hi": ("ai4bharat/sangraha", {"data_dir": "verified/hin"}, 900, 1),
}

EDU_WORDS = ("इतिहास", "विज्ञान", "सिद्धांत", "प्रक्रिया", "उदाहरण", "परिभाषा", "अर्थ", "कारण", "प्रकार",
             "स्थित", "जन्म", "राजधानी", "नदी", "गणित", "भूगोल", "संविधान", "अध्ययन", "विकास", "संरचना",
             "तत्व", "ऊर्जा", "शरीर", "रोग", "भाषा", "साहित्य", "वंश", "साम्राज्य", "स्वतंत्रता", "आविष्कार",
             "पृथ्वी", "सूर्य", "जनसंख्या", "क्षेत्रफल", "अर्थव्यवस्था", "कहलाता", "कहते हैं", "को कहा जाता")
NOISE_WORDS = ("ने कहा", "बताया कि", "सूत्रों", "पुलिस", "गिरफ्तार", "वायरल", "बॉलीवुड", "राशिफल", "शेयर बाजार",
               "सेंसेक्स", "ब्रेकिंग", "लाइव अपडेट", "क्लिक करें", "डाउनलोड करें", "ऑफर", "छूट", "सब्सक्राइब",
               "फॉलो करें", "मैच में", "बॉक्स ऑफिस", "प्रेमिका", "खूबसूरत तस्वीरें", "देखें वीडियो",
               "सेलेब", "रिजल्ट जारी", "भाजपा", "कांग्रेस", "चुनाव में", "प्रधानमंत्री मोदी", "मुख्यमंत्री")
DATELINE = re.compile(r"^\s*[A-Zऀ-ॿ][\wऀ-ॿ ]{1,25}(\(.{1,20}\))?\s*[:：]\s")   # "KANPUR: ", "लंदनः"
NEWS_WORDS = ("समाचार", "संवाददाता", "रिपोर्ट", "खबर", "अधिकारियों ने", "मंत्री ने", "बुधवार", "गुरुवार", "सोमवार")
YEAR = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-2][0-9])(?!\d)")


def hindi_edu_score(text: str) -> int:
    """0..5 rule-based stand-in for an educational-value classifier (Hindi)."""
    if not text:
        return 0
    deva = len(DEV.findall(text)) / max(1, len(text))
    score = 0
    score += len(text) >= 800
    score += deva >= 0.6
    score += text.count("है।") + text.count("थे।") + text.count("था।") + text.count("हैं।") >= 5
    score += sum(w in text for w in EDU_WORDS) >= 4
    score += len(YEAR.findall(text)) >= 2 or "प्रतिशत" in text or "किलोमीटर" in text
    score -= min(2, sum(w in text for w in NOISE_WORDS))    # gossip, politics, ads
    if DATELINE.match(text) or sum(w in text[:1500] for w in NEWS_WORDS) >= 2:   # day-to-day news
        score -= 1
    raw = text.encode()
    if len(raw) > 2000 and len(zlib.compress(raw)) / len(raw) < 0.18:   # boilerplate / repeated lines
        score -= 2
    return max(0, min(5, score))


def _texts(name: str, row: dict) -> List[str]:
    if name.startswith("wiki_"):
        return _wiki_text(row)
    if name == "fineweb_edu_en":
        if (row.get("int_score") or 0) < 3:
            return []
        t = norm(row.get("text"))
        return [t] if len(t) >= 300 else []
    if name == "sangraha_hi":
        t = clean_article(row.get("text") or "")
        return [t] if hindi_edu_score(t) >= 4 else []
    return []


def stream_rows(repo: str, kwargs: dict) -> Iterator[dict]:
    """Streamed (no full download) and shuffled, so a budget takes a sample of the whole set,
    not the first articles alphabetically."""
    from datasets import load_dataset
    yield from load_dataset(repo, split="train", streaming=True, **kwargs).shuffle(seed=0, buffer_size=10_000)


def build(data_dir: str, scale: float = 1.0, only: Optional[List[str]] = None,
          rows_fn: Callable[[str, dict], Iterator[dict]] = stream_rows, seed: int = 0) -> Dict:
    out_path = os.path.join(data_dir, "pretrain_knowledge.jsonl")
    out = _Buckets(out_path, seed=seed)
    seen: set = set()
    report: Dict[str, dict] = {}
    for name, (repo, kwargs, budget_m, repeat) in SOURCES.items():
        if only and name not in only:
            continue
        budget = int(budget_m * 1e6 * scale)
        stat = {"rows_read": 0, "kept": 0, "chars": 0}
        report[name] = stat
        try:
            rows = rows_fn(repo, kwargs)
            log.info(f"{name}: up to {budget / 1e6:,.0f}M chars from {repo}")
            for row in rows:
                stat["rows_read"] += 1
                for t in _texts(name, row):
                    if bad_text(t, 150):
                        continue
                    k = key(t)
                    if k in seen:
                        continue
                    seen.add(k)
                    for _ in range(repeat):
                        out.add({"text": t})
                    stat["kept"] += 1
                    stat["chars"] += len(t)
                if stat["chars"] >= budget:
                    break
                if stat["rows_read"] % 100_000 == 0:
                    log.info(f"  {name}: {stat['rows_read']:,} read, {stat['chars'] / 1e6:,.0f}M chars kept")
        except Exception as e:  # noqa: BLE001 — a missing source must not stop the others
            stat["error"] = str(e).splitlines()[0][:200]
            log.warning(f"Skipping the rest of {name}: {stat['error']}")
        log.info(f"{name}: kept {stat['kept']:,} texts, {stat['chars'] / 1e6:,.1f}M chars")
    out.close()
    report["total_rows_written"] = out.count
    with open(os.path.join(data_dir, "knowledge_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log.info(f"Wrote {out_path}: {out.count:,} rows")
    return report
