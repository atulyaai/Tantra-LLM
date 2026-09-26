"""
Tantra/data_prep.py — Clean and mix all training data (python main.py --mode data).

Inputs   Datasets/master_train.jsonl, master_val.jsonl   (the original Hindi data)
         Datasets/raw/<name>/*.parquet                    (open datasets, see SOURCES)
Outputs  Datasets/pretrain.jsonl      {"text": ...}      plain text: articles, Wikipedia
         Datasets/sft.jsonl           {"messages": ...}  conversations: questions -> answers
         Datasets/val_pretrain.jsonl, Datasets/val_sft.jsonl   held out, never trained on
         Datasets/data_report.json    what was kept / dropped and the language mix

Fixes applied to the original data (found by auditing all 1.33M rows):
  * ~635k rows were an article behind one of 5 generic questions ("इसके बारे में विस्तृत
    जानकारी दें:" ...) or "विषय: <headline> ... संपूर्ण जानकारी" -> kept as plain text, not as
    fake Q&A, with news-site boilerplate removed.
  * ~80k rows had shifted roles: the system prompt was stored as the USER message and the
    question as the ASSISTANT answer (e.g. user "आप गणित के शिक्षक हैं।" -> assistant
    "26 और 46 का योग ज्ञात कीजिए।"). They taught the model to answer with a question.
    Repaired when the real answer follows, dropped otherwise.
  * exact and near duplicates, empty / broken / highly repetitive rows removed.
Everything is globally shuffled (sources are interleaved, not one after another).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
import unicodedata
import zlib
from collections import Counter
from typing import Dict, Iterator, List, Optional, Tuple

from Tantra.utils import get_logger

log = get_logger("tantra.data")

FILLER_PREFIXES = (
    "इसके मुख्य बिंदु क्या हैं? स्पष्ट कीजिए:", "इसके बारे में विस्तृत जानकारी दें:",
    "कृपया निम्नलिखित विषय को विस्तार से समझाएं:", "इस विषय पर प्रकाश डालें और मुख्य तथ्य बताएं:",
    "ज्ञानवर्धन हेतु इसका विवरण प्रस्तुत करें:",
)
VISHAY = re.compile(r"^विषय:\s*(.+?)\s*\n+\s*कृपया इसके बारे में संपूर्ण जानकारी प्रदान करें।?\s*$", re.S)
PERSONA = re.compile(r"^(आप|तुम)\s[^?\n]{2,60}(हैं|हो|है)[।.]?$")   # "आप गणित के शिक्षक हैं।"
BOILERPLATE = re.compile(
    r"(इस पृष्ठ को प्रिंट करें|🔊\s*पूरी खबर सुने|टिप्पणियों पर जाएँ|हर दिन कुछ नया सीखें|ये भी पढ़ें[:\-]?.*|यह भी पढ़ें[:\-]?.*|"
    r"Also Read.*|Advertisement|विज्ञापन|Click here.*|https?://\S+|www\.\S+|\S+@\S+\.\w+|"
    r"^\s*(डिजिटल डेस्क|जागरण संवाददाता|जेएनएन|ब्यूरो|संवाद सूत्र|एजेंसी)[^।\n:]{0,40}?\s[:\-]\s*|"
    r"^\s*(डिजिटल डेस्क|जागरण संवाददाता|जेएनएन|ब्यूरो|संवाद सूत्र|एजेंसी)[^।\n:]{0,40}?:\s*|\((भाषा|निप्र|एजेंसी|वार्ता)\))",
    re.M)
JUNK_TITLES = ("डिजिटल डेस्क", "इस पृष्ठ", "टिप्पणियों", "हर दिन कुछ नया", "पूरी खबर", "जागरण संवाददाता", "जेएनएन", "(भाषा)")
DEV = re.compile(r"[ऀ-ॿ]")
LAT = re.compile(r"[A-Za-z]")

# name -> (kind, how many rows at most, repeat factor)
SOURCES = {
    "wiki_simple_en": ("pretrain", None, 1),
    "wiki_sa": ("pretrain", None, 1),
    "alpaca_en": ("sft", None, 1),
    "alpaca_hi": ("sft", None, 1),
    "alpaca_hi_hinglish": ("sft", None, 1),
    "hinglish_conv": ("sft", 150_000, 1),
    "gsm8k": ("sft", None, 2),
    "python_code": ("sft", None, 1),
}


# ── text helpers ─────────────────────────────────────────────────────────────

def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").replace("\r\n", "\n").replace("\r", "\n").replace("​", "")
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def clean_article(s: str) -> str:
    s = BOILERPLATE.sub("", norm(s))
    lines = [ln.strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if len(ln) > 1]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def bad_text(s: str, min_chars: int = 2) -> bool:
    if len(s) < min_chars or "�" in s:
        return True
    letters = len(DEV.findall(s)) + len(LAT.findall(s))
    if len(s) > 40 and letters / len(s) < 0.35:           # mostly symbols / numbers / markup
        return True
    if len(s) > 300 and len(zlib.compress(s.encode())) / len(s.encode()) < 0.12:   # very repetitive
        return True
    return False


HINGLISH_WORDS = set("hai hain ka ki ke ko mein me se aur nahi nahin kya kaise kyun kyon yaar bhai hum tum aap mujhe "
                     "tera mera tumhara hoga hogi raha rahi karo karna kar kuch bahut accha acha thik theek abhi "
                     "bhi toh lekin agar jab tab wala wali ho gaya gayi".split())


def lang_of(s: str) -> str:
    d, l = len(DEV.findall(s)), len(LAT.findall(s))
    t = d + l or 1
    if d / t > 0.8:
        return "hindi"
    if l / t > 0.8:
        words = re.findall(r"[a-z]+", s.lower()[:2000])
        if words and sum(w in HINGLISH_WORDS for w in words) / len(words) > 0.12:
            return "hinglish"   # Hindi written in Latin letters
        return "english"
    return "mixed"


def key(s: str, n: int = 400) -> str:
    return hashlib.md5(re.sub(r"\W+", "", s.lower())[:n].encode()).hexdigest()[:16]


# ── the original data ────────────────────────────────────────────────────────

def convert_master_row(o: dict, stats: Counter) -> Iterator[Tuple[str, dict]]:
    msgs = [{"role": m.get("role"), "content": norm(m.get("content") or "")} for m in o.get("messages") or []
            if m.get("role") in ("system", "user", "assistant")]
    if not msgs:
        stats["dropped: empty"] += 1
        return
    system = next((m["content"] for m in msgs if m["role"] == "system"), None)
    turns = [m for m in msgs if m["role"] != "system"]

    # shifted roles: persona sentence as the user, the question as the assistant
    if len(turns) >= 2 and turns[0]["role"] == "user" and PERSONA.match(turns[0]["content"]):
        if len(turns) >= 3:
            system = turns[0]["content"]
            fixed = [{"role": "user" if i % 2 == 0 else "assistant", "content": t["content"]} for i, t in enumerate(turns[1:])]
            if len(fixed) % 2:
                fixed = fixed[:-1]
            if fixed:
                stats["fixed: shifted roles"] += 1
                turns = fixed
            else:
                stats["dropped: shifted roles, no answer"] += 1
                return
        else:
            stats["dropped: shifted roles, no answer"] += 1
            return

    if len(turns) < 2 or turns[0]["role"] != "user" or turns[1]["role"] != "assistant":
        stats["dropped: not a conversation"] += 1
        return
    q, a = turns[0]["content"], turns[1]["content"]

    # generic question + article -> plain text
    for p in FILLER_PREFIXES:
        if q.startswith(p):
            body = clean_article((q[len(p):] + "\n\n" + a).strip())
            if len(body) < 200 or bad_text(body):
                stats["dropped: short/broken article"] += 1
                return
            stats["article (filler question -> text)"] += 1
            yield "pretrain", {"text": body}
            return
    m = VISHAY.match(q)
    if m:
        title = m.group(1).strip()
        body = clean_article(a)
        if len(body) < 200 or bad_text(body):
            stats["dropped: short/broken article"] += 1
            return
        if not any(j in title for j in JUNK_TITLES) and len(title) > 8:
            body = f"{title}\n\n{body}"
        stats["article (विषय headline -> text)"] += 1
        yield "pretrain", {"text": body}
        return

    if bad_text(a) or bad_text(q, 1):
        stats["dropped: broken text"] += 1
        return
    out = []
    if system and not PERSONA.match(system or "x"):
        out.append({"role": "system", "content": system})
    for t in turns if len(turns) % 2 == 0 else turns[:-1]:
        out.append(t)
    stats["conversation" + (" (translation)" if q.startswith("Translate this to") else "")] += 1
    yield "sft", {"messages": out}


# ── open datasets (parquet) ──────────────────────────────────────────────────

def _rows(name: str, raw_dir: str) -> Iterator[dict]:
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(os.path.join(raw_dir, name, "*.parquet"))):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=4096):
            yield from batch.to_pylist()


def _wiki_text(r: dict) -> List[str]:
    text = norm(r.get("text"))
    text = re.split(r"\n(References|Related pages|Other websites|Notes|Sources|सन्दर्भाः|बाह्यसम्पर्कतन्तुः)\s*\n", text)[0]
    if len(text) < 150:
        return []
    title = norm(r.get("title"))
    out, buf = [], f"{title}\n\n" if title else ""
    for para in text.split("\n\n"):          # long pages -> pieces of ~6k chars
        if len(buf) + len(para) > 6000 and len(buf) > 500:
            out.append(buf.strip())
            buf = f"{title} (जारी)\n\n" if DEV.search(title or "") else f"{title} (continued)\n\n"
        buf += para + "\n\n"
    if len(buf.strip()) > 150:
        out.append(buf.strip())
    return out


def convert_source_row(name: str, r: dict) -> Iterator[Tuple[str, dict]]:
    def qa(q: str, a: str) -> Iterator[Tuple[str, dict]]:
        q, a = norm(q), norm(a)
        if q and a and not bad_text(a) and not bad_text(q, 1):
            yield "sft", {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}

    if name.startswith("wiki_"):
        for t in _wiki_text(r):
            yield "pretrain", {"text": t}
    elif name in ("alpaca_en", "alpaca_hi"):
        yield from qa(r["instruction"] + ("\n\n" + r["input"] if r.get("input") else ""), r["output"])
    elif name == "alpaca_hi_hinglish":
        yield from qa(r["input"], r["output"])                       # Hindi
        yield from qa(r["input_hinglish"], r["output_hinglish"])     # Hinglish (roman)
    elif name == "hinglish_conv":
        yield from qa(r["input"], r["output"])
    elif name == "gsm8k":
        ans = re.sub(r"<<[^>]*>>", "", r["answer"])
        ans = re.sub(r"\n?#### *(.+)$", r"\n\nAnswer: \1", ans.strip())
        yield from qa(r["question"], ans)
    elif name == "python_code":
        q = r["instruction"] + ("\n\nInput: " + r["input"] if r.get("input") else "")
        yield from qa(q, "```python\n" + r["output"].strip() + "\n```")


# ── build ────────────────────────────────────────────────────────────────────

class _Buckets:
    """Spread records over N temp files at random, then shuffle each in memory -> global shuffle."""

    def __init__(self, path: str, n: int = 64, seed: int = 0) -> None:
        self.path, self.n = path, n
        self.rng = random.Random(seed)
        self.files = [open(f"{path}.part{i}", "w", encoding="utf-8") for i in range(n)]
        self.count = 0

    def add(self, rec: dict) -> None:
        self.files[self.rng.randrange(self.n)].write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.count += 1

    def close(self) -> None:
        for f in self.files:
            f.close()
        with open(self.path + ".tmp", "w", encoding="utf-8") as out:
            for i in range(self.n):
                part = f"{self.path}.part{i}"
                with open(part, encoding="utf-8") as f:
                    lines = f.readlines()
                self.rng.shuffle(lines)
                out.writelines(lines)
                os.remove(part)
        os.replace(self.path + ".tmp", self.path)


def build(data_dir: str, raw_dir: Optional[str] = None, val_fraction: float = 0.004,
          max_val: int = 4000, seed: int = 0) -> Dict:
    raw_dir = raw_dir or os.path.join(data_dir, "raw")
    out = {k: _Buckets(os.path.join(data_dir, f"{k}.jsonl"), seed=seed) for k in ("pretrain", "sft")}
    val = {k: open(os.path.join(data_dir, f"val_{k}.jsonl"), "w", encoding="utf-8") for k in ("pretrain", "sft")}
    val_n = Counter()
    seen: set = set()
    stats: Counter = Counter()
    chars: Counter = Counter()
    by_source: Dict[str, Counter] = {}

    def emit(source: str, kind: str, rec: dict) -> None:
        text = rec["text"] if kind == "pretrain" else "\n".join(m["content"] for m in rec["messages"])
        k = key(text)
        if k in seen:
            stats["dropped: duplicate"] += 1
            return
        seen.add(k)
        lang = lang_of(text)
        chars[f"{kind}/{lang}"] += len(text)
        by_source.setdefault(source, Counter())[kind] += 1
        # deterministic hold-out: the same row always lands on the same side
        if int(k[:6], 16) / 0xFFFFFF < val_fraction and val_n[kind] < max_val:
            val[kind].write(json.dumps(rec, ensure_ascii=False) + "\n")
            val_n[kind] += 1
        else:
            out[kind].add(rec)

    for fname in ("master_train.jsonl", "master_val.jsonl"):
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            continue
        log.info(f"Cleaning {fname} ...")
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    o = json.loads(line)
                except ValueError:
                    stats["dropped: bad json"] += 1
                    continue
                for kind, rec in convert_master_row(o, stats):
                    emit(fname, kind, rec)
                if i and i % 200_000 == 0:
                    log.info(f"  {i:,} rows")

    for name, (kind_hint, limit, repeat) in SOURCES.items():
        if not glob.glob(os.path.join(raw_dir, name, "*.parquet")):
            log.warning(f"Skipping {name}: not downloaded (Datasets/raw/{name}/)")
            continue
        log.info(f"Adding {name} ...")
        rows = list(_rows(name, raw_dir))
        random.Random(seed).shuffle(rows)
        n = 0
        for r in rows:
            if limit and n >= limit:
                break
            recs = list(convert_source_row(name, r))
            for _ in range(repeat):
                for kind, rec in recs:
                    if _:   # repeated copies skip dedup on purpose
                        out[kind].add(rec)
                    else:
                        emit(name, kind, rec)
            n += 1

    for v in val.values():
        v.close()
    for b in out.values():
        b.close()
    total = sum(chars.values()) or 1
    report = {
        "rows": {k: b.count for k, b in out.items()}, "val_rows": dict(val_n),
        "language_mix_percent_of_chars": {k: round(100 * v / total, 1) for k, v in sorted(chars.items())},
        "chars_million": {k: round(v / 1e6, 1) for k, v in sorted(chars.items())},
        "by_source": {k: dict(v) for k, v in by_source.items()},
        "cleaning": dict(stats.most_common()),
    }
    with open(os.path.join(data_dir, "data_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log.info(json.dumps(report, ensure_ascii=False, indent=1))
    return report
