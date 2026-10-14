"""
Think -> check -> search -> fix -> learn
=======================================
A small model can't "think longer" reliably, so accuracy comes from code
around it:

  1. route          conversation experts take the FAST path (1 draft, no checks)
  2. draft          factual / math questions: the model writes N drafts
  3. check          math: exact Python arithmetic
                    facts: compare drafts with the local knowledge index
  4. search         if nothing local: local teacher model (Ollama, if running)
                    -> internet (Wikipedia)
  5. fix            return the verified / found answer
  6. learn          every correction or web answer is saved to
                      Datasets/learned/facts.jsonl            (instant lookup, offline next time)
                      Datasets/feedback/auto/<expert>.jsonl   (training data for the expert)

Once a fact is learned, the internet is never needed for it again.
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LEARNED = Path("Datasets/learned/facts.jsonl")
AUTO_FEEDBACK = Path("Datasets/feedback/auto")
USER_AGENT = "Tantra-LLM/0.4 (https://github.com/atulyaai/Tantra-LLM; offline-first research assistant)"

STOP = set("a an the is are was were of in on at to for and or what whats what's which who whom how many much "
           "tell me please do does did can you your i it its this that be by with from as about".split())
FACT_Q = re.compile(r"^\s*(what'?s|what|which|who'?s|who|where'?s|where|when|how many|how much|how far|"
                    r"how fast|how big|how tall|why|capital|name|define|tell me)\b", re.I)
FACT_WORDS = re.compile(r"\b(capital|population|currency|president|prime minister|largest|smallest|"
                        r"highest|lowest|longest|shortest|invented|discovered|founded|distance between|"
                        r"wrote|written|creator|composed|sutra|theory|formula|element|"
                        r"grammar|father|author|scientist|mission|temple|river|mountain|headquarters)\b", re.I)
INDIC_FACT_WORDS = ("राजधानी", "मुद्रा", "राष्ट्रपति", "प्रधानमंत्री", "आविष्कार", "रचयिता", "मुख्यालय",
                    "वेद", "वेदव्यास", "पाणिनि", "आर्यभट्ट", "चंद्रयान", "रामानुजन", "पशु", "पक्षी", "क्या", "कहाँ", "कौन", "किसने", "कब", "क्यों")


def _toks(s: str) -> List[str]:
    return [w for w in re.findall(r"[\w']+", s.lower()) if w not in STOP]


# ── tools ────────────────────────────────────────────────────────────────────

_OPS = {"plus": "+", "add": "+", "minus": "-", "subtract": "-", "times": "*", "x": "*",
        "multiplied by": "*", "into": "*", "divided by": "/", "over": "/"}


def math_answer(q: str) -> Optional[str]:
    """Exact arithmetic for 'what is 12 times 7', '45+38', '100 / 4'."""
    s = q.lower()
    for w, o in sorted(_OPS.items(), key=lambda kv: -len(kv[0])):
        s = re.sub(rf"\b{w}\b", f" {o} ", s)
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/×])\s*(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    a, op, b = float(m.group(1)), m.group(2).replace("×", "*"), float(m.group(3))
    if op == "/" and b == 0:
        return "You can't divide by zero."
    r = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else math.nan}[op]
    fmt = lambda v: str(int(v)) if float(v).is_integer() else f"{v:.4g}"
    sym = {"*": "x", "/": "/", "+": "+", "-": "-"}[op]
    return f"{fmt(a)} {sym} {fmt(b)} = {fmt(r)}."


# ── local knowledge index ────────────────────────────────────────────────────

class KnowledgeIndex:
    """Hybrid High-Density Memory Store over curated knowledge Q/A + learned facts."""

    def __init__(self, data_dirs=("Datasets/curated",), learned: Optional[Path] = None, memory_path: Optional[Path] = None):
        learned = learned or LEARNED
        from Tantra.memory import TantraMemoryStore
        loaded = False
        is_default_dirs = [str(d).replace("\\", "/") for d in data_dirs] == ["Datasets/curated"]
        candidates = [Path(memory_path)] if memory_path else ([Path("Model/memory.pt")] if is_default_dirs else [])
        for p in candidates:
            if p and p.exists():
                try:
                    self.store = TantraMemoryStore.load(p)
                    loaded = True
                    break
                except Exception:
                    pass
        if not loaded:
            self.store = TantraMemoryStore(dim=64, alpha=0.6)
            self.store.load_from_datasets(data_dirs, learned_path=learned)
        self.docs = [(q, a, e) for q, a, e, _ in self.store.documents.values()]

    def add(self, q: str, a: str, expert: str) -> None:
        self.store.add_fact(q, a, expert)
        self.docs.append((q, a, expert))

    def search(self, q: str) -> Optional[Tuple[str, str, str, float]]:
        """Return (question, answer, expert, score) or None."""
        return self.store.search(q)


# ── outside help ─────────────────────────────────────────────────────────────

def _get_json(url: str, timeout: float = 6.0, data: Optional[bytes] = None) -> Optional[dict]:
    try:
        headers = {"User-Agent": USER_AGENT}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def ask_local_teacher(q: str, host: str = "http://localhost:11434") -> Optional[Tuple[str, str]]:
    """Use a bigger local model if one is running (Ollama). Returns (answer, source)."""
    tags = _get_json(f"{host}/api/tags", timeout=1.0)
    if not tags or not tags.get("models"):
        return None
    model = tags["models"][0]["name"]
    body = json.dumps({"model": model, "stream": False,
                       "prompt": f"Answer in one or two short, factual sentences.\nQuestion: {q}\nAnswer:"}).encode()
    r = _get_json(f"{host}/api/generate", timeout=60, data=body)
    ans = (r or {}).get("response", "").strip()
    return (ans, f"ollama:{model}") if ans else None


def _stem(w: str) -> str:
    return w[:5]  # crude: invented/inventor, capital/capitals


def search_web(q: str) -> Optional[Tuple[str, str]]:
    """Wikipedia in 2 requests: (1) full-text search for the question,
    (2) intro text of the top results in one batch. Picks the article whose
    first sentences best answer the question. Returns (answer, url)."""
    qt = set(_toks(q))
    if not qt:
        return None
    res = _get_json("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": q, "srlimit": 6, "format": "json"}))
    titles = [h["title"] for h in (res or {}).get("query", {}).get("search", [])]
    if not titles:
        return None
    pages = _get_json("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "exsentences": 2,
         "redirects": 1, "titles": "|".join(titles), "format": "json"}))
    extracts = {p.get("title"): p.get("extract", "") for p in (pages or {}).get("query", {}).get("pages", {}).values()}
    qs = {_stem(t) for t in qt}
    asks_entity = bool(re.match(r"\s*(who|which|where|what is the \w+ of|name the)\b", q, re.I))
    best, best_score = None, 0.0
    for rank, title in enumerate(titles):
        text = re.sub(r"\s*\([^()]*\)", "", extracts.get(title, ""))  # drop "( KAN-brə; ...)" asides
        text = re.sub(r"\s+", " ", text).strip()
        if not text or "may refer to" in text:
            continue
        first = re.split(r"(?<=[.!?])\s", text)[0]
        at = [_stem(t) for t in _toks(text)]
        ft = [_stem(t) for t in _toks(first)]
        tt = set(_toks(title))
        score = (len(qs & set(at)) / len(qs)                       # covers the question
                 + 0.3 * (len(qs & set(ft)) / max(1, len(ft)))    # dense, direct first sentence
                 + 0.3 * (bool(tt) and tt <= qt)                  # article named in question ("what is X")
                 + 0.4 * (bool(tt) and not (tt & qt) and asks_entity)  # a NEW name answers "who/which ..."
                 - 0.03 * rank)
        if score > best_score:
            best, best_score = (text, "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))), score
    return best if best_score >= 0.5 else None


# ── learning ─────────────────────────────────────────────────────────────────

def learn(q: str, a: str, expert: str, source: str) -> None:
    row = {"q": q, "a": a, "expert": expert, "source": source, "time": datetime.now().isoformat(timespec="seconds")}
    LEARNED.parent.mkdir(parents=True, exist_ok=True)
    with open(LEARNED, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    AUTO_FEEDBACK.mkdir(parents=True, exist_ok=True)
    msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    with open(AUTO_FEEDBACK / f"{expert}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"messages": msgs, "source": source}, ensure_ascii=False) + "\n")


# ── the loop ─────────────────────────────────────────────────────────────────

def _key_terms(answer: str, question: str) -> List[str]:
    """Words in the trusted answer that are not in the question (e.g. 'Tokyo', '84').

    Falls back to the answer's numbers when everything else overlaps (e.g.
    '144 / 12 = 12' vs 'what is 144 divided by 12?' -> ['12']); never returns
    empty, because all([]) is True and would verify ANY draft.
    """
    q = set(_toks(question))
    keys = [t for t in _toks(answer) if t not in q][:4]
    if not keys:
        nums = [t for t in _toks(answer) if any(c.isdigit() for c in t)]
        keys = nums[-1:] or _toks(answer)[-1:]
    return keys


class Reasoner:
    def __init__(self, engine, data_dirs=("Datasets/curated",), drafts: int = 3, use_web: bool = True,
                 use_teacher: bool = True, learn_new: bool = True):
        self.eng = engine
        self.index = KnowledgeIndex(data_dirs)
        self.drafts = drafts
        self.use_web, self.use_teacher, self.learn_new = use_web, use_teacher, learn_new
        self.trace: List[str] = []

    CHATTY = ("humor", "greeting", "farewell", "empathy", "apology")

    def needs_check(self, q: str, route) -> bool:
        top = route[0][0] if route else ""
        if top.startswith("gk_") or math_answer(q) or FACT_WORDS.search(q) or any(w in q for w in INDIC_FACT_WORDS):
            return True
        if top.startswith("identity"):
            # Only fast-path if user is asking about the bot itself (you/your/tantra)
            if any(w in q.lower() for w in ("you", "your", "yourself", "tantra")):
                return False
            return bool(FACT_Q.match(q))
        if top.startswith(self.CHATTY):
            return False  # chatty experts never go to the knowledge path
        return bool(FACT_Q.match(q))

    def answer(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_new_tokens: int = 60) -> Dict:
        q = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        route = self.eng.route(q)
        self.trace = [f"route: {', '.join(f'{n}:{w:.2f}' for n, w in route) or 'base'}"]

        if not self.needs_check(q, route):  # FAST path
            reply = self.eng.chat(messages, temperature=temperature, max_new_tokens=max_new_tokens, experts=route)
            self.trace.append("fast path (no check needed)")
            return {"reply": reply, "source": "model", "route": route, "trace": self.trace}

        expert = route[0][0] if route and route[0][0].startswith("gk_") else "gk_general"
        # trusted reference: exact tool > local knowledge
        ref, ref_src = math_answer(q), "calculator"
        if ref:
            expert = "gk_math"
        else:
            hit = self.index.search(q)
            if hit and hit[3] >= 0.75:
                ref, ref_src = hit[1], f"local:{hit[2]}"
                expert = hit[2] if hit[2].startswith("gk_") else expert
        self.trace.append(f"reference: {ref_src if ref else 'none locally'}")

        # drafts (low temperature, short)
        drafts = []
        for i in range(self.drafts):
            d = self.eng.chat(messages, temperature=0.0 if i == 0 else 0.6, max_new_tokens=max_new_tokens,
                              experts=route)
            drafts.append(d)
            if ref and all(k in _toks(d) for k in _key_terms(ref, q)):
                self.trace.append(f"draft {i + 1} verified ✓")
                return {"reply": d, "source": "model (verified)", "route": route, "trace": self.trace}
            self.trace.append(f"draft {i + 1} rejected: {d[:60]!r}")

        if ref:  # model was wrong -> correct it and learn
            if self.learn_new and not ref_src.startswith("local"):
                learn(q, ref, expert, ref_src)
            elif self.learn_new:
                learn(q, ref, expert, "correction")  # expert still got it wrong -> more training data
            self.trace.append("fixed with reference, saved for training")
            return {"reply": ref, "source": ref_src, "route": route, "trace": self.trace}

        # nothing local: ask outside, then learn so it's offline next time
        found = None
        if self.use_teacher:
            found = ask_local_teacher(q)
            self.trace.append("local teacher: " + (found[1] if found else "not available"))
        if not found and self.use_web:
            found = search_web(q)
            self.trace.append("web search: " + (found[1] if found else "nothing found"))
        if found:
            ans, src = found
            if self.learn_new:
                learn(q, ans, expert, src)
                self.index.add(q, ans, expert)
                self.trace.append(f"learned -> {expert} (offline next time)")
            return {"reply": ans, "source": src, "route": route, "trace": self.trace}

        self.trace.append("no verified answer; returning best draft")
        return {"reply": drafts[0] + " (I'm not sure about this.)", "source": "model (unverified)",
                "route": route, "trace": self.trace}
