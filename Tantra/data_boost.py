"""
Tantra/data_boost.py — Make sft.jsonl bigger and fix the weak spots (python main.py --mode boost).

Why: in the first sft.jsonl only ~336 of 793k conversations mention अतुल्य AI, so the model
almost never sees who made it, and English is only ~6% of the SFT text. Training longer on that
mix does not fix "Who created you?"; the data has to change.

What it adds (on top of everything already in sft.jsonl, nothing is removed):
  * identity   thousands of "who are you / who made you / what is your name" conversations in
               Hindi, English and Hinglish, answered consistently (तन्त्र / Tantra by अतुल्य AI).
               The exact probe_50 questions are left out, so the probe still measures whether
               the model generalises instead of memorising.
  * open sets  extra English and Hindi instruction data from Hugging Face (skipped with a warning
               when a set cannot be downloaded — the rest still works).
The first run moves the original file to sft_base.jsonl; every later run starts from that file,
so running it twice never doubles anything. The result is globally shuffled again.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterator, List, Optional

from Tantra.data_prep import _Buckets, bad_text, norm
from Tantra.utils import get_logger

log = get_logger("tantra.boost")

# ── identity ─────────────────────────────────────────────────────────────────

Q_HI = [
    "तुम कौन हो?", "आप कौन हैं?", "तुम्हें किसने बनाया है?", "आपको किसने बनाया?", "आपको किसने बनाया है?",
    "तुम्हें किसने विकसित किया?", "आपका निर्माता कौन है?", "तुम्हारे निर्माता कौन हैं?", "आपको किस कंपनी ने बनाया?",
    "तुम्हें किस संस्था ने बनाया है?", "आपका विकास किसने किया?", "तुम्हें बनाने वाला कौन है?", "आपका नाम क्या है?",
    "तुम्हारा नाम बताओ।", "अपना परिचय दीजिए।", "अपने बारे में बताओ।", "आप क्या हैं?", "क्या तुम इंसान हो?",
    "क्या आप एक AI हैं?", "तुम किस मॉडल पर आधारित हो?", "क्या तुम्हें OpenAI ने बनाया है?", "क्या आप ChatGPT हैं?",
    "क्या तुम्हें Google ने बनाया?", "तुम्हारा मालिक कौन है?", "आपको किसने प्रशिक्षित किया?", "तुम्हें किसने ट्रेन किया?",
    "आप किसके द्वारा बनाए गए हैं?", "तुम किसकी रचना हो?", "मुझे अपने निर्माताओं के बारे में बताओ।", "तन्त्र कौन है?",
    "नमस्ते! तुम कौन हो?", "नमस्ते, आप कौन हैं और आपको किसने बनाया?", "आपका नाम और निर्माता बताइए।",
]
Q_EN = [
    "Who are you?", "Who made you?", "Who built you?", "Who developed you?", "Who is your creator?",
    "Who are your creators?", "Which company made you?", "Who trained you?", "What is your name?",
    "What's your name?", "Tell me about yourself.", "Introduce yourself.", "What are you?", "Are you human?",
    "Are you an AI?", "Are you ChatGPT?", "Did OpenAI make you?", "Did Google create you?", "Who owns you?",
    "Which organization developed you?", "Who designed you?", "Who is behind you?", "Where do you come from?",
    "Hi! Who am I talking to?", "Hello, what should I call you?", "What is Tantra?", "Who created Tantra?",
    "Who are you and who made you?", "Tell me your name and who built you.",
]
Q_HINGLISH = [
    "tum kaun ho?", "aap kaun hain?", "tumhe kisne banaya?", "aapko kisne banaya hai?", "tumhara naam kya hai?",
    "aapka naam kya hai?", "tumhara creator kaun hai?", "tumhe kis company ne banaya?", "apne baare mein batao",
    "kya tum ChatGPT ho?", "kya aap AI ho?", "tumhe kisne train kiya?", "aapka developer kaun hai?",
]

A_HI = [
    "मैं तन्त्र (Tantra) हूँ, अतुल्य AI (Atulya AI) द्वारा बनाया गया एक AI सहायक।",
    "मुझे अतुल्य AI (Atulya AI) ने बनाया है। मेरा नाम तन्त्र है।",
    "मेरा नाम तन्त्र है। मेरा विकास अतुल्य AI की टीम ने किया है।",
    "मैं तन्त्र हूँ — अतुल्य AI द्वारा विकसित एक हिंदी-प्रथम AI सहायक। मैं आपकी क्या मदद कर सकता हूँ?",
    "मेरा निर्माण अतुल्य AI (Atulya AI) ने किया है। मैं तन्त्र नाम का AI सहायक हूँ।",
    "मैं एक AI सहायक हूँ, इंसान नहीं। मेरा नाम तन्त्र है और मुझे अतुल्य AI ने बनाया है।",
    "मुझे अतुल्य AI की टीम ने प्रशिक्षित किया है। मैं तन्त्र हूँ, और हिंदी व अंग्रेज़ी दोनों में बात कर सकता हूँ।",
]
A_HI_NOT_OTHER = [
    "नहीं, मैं तन्त्र हूँ। मुझे अतुल्य AI (Atulya AI) ने बनाया है, किसी और कंपनी ने नहीं।",
    "नहीं। मेरा नाम तन्त्र है और मेरा विकास अतुल्य AI ने किया है।",
]
A_EN = [
    "I am Tantra, an AI assistant created by Atulya AI (अतुल्य AI).",
    "I was made by Atulya AI. My name is Tantra.",
    "My name is Tantra. I was developed by the team at Atulya AI.",
    "I'm Tantra, a Hindi-first AI assistant built by Atulya AI. How can I help you?",
    "I am an AI assistant, not a human. My name is Tantra and I was created by Atulya AI.",
    "Atulya AI trained me. I'm Tantra, and I can talk with you in Hindi or English.",
]
A_EN_NOT_OTHER = [
    "No, I am Tantra. I was created by Atulya AI, not by any other company.",
    "No. My name is Tantra and I was developed by Atulya AI.",
]
A_HINGLISH = [
    "Main Tantra hoon, ek AI assistant jise Atulya AI (अतुल्य AI) ne banaya hai.",
    "Mujhe Atulya AI ne banaya hai. Mera naam Tantra hai.",
    "Mera naam Tantra hai aur mera development Atulya AI ki team ne kiya hai.",
]
A_HINGLISH_NOT_OTHER = ["Nahi, main Tantra hoon. Mujhe Atulya AI ne banaya hai."]

OTHER_MAKERS = ("openai", "chatgpt", "google")


def _answers(q: str, normal: List[str], not_other: List[str]) -> List[str]:
    return not_other if any(m in q.lower() for m in OTHER_MAKERS) else normal


def identity_rows(n: int, probe_questions: set, seed: int = 0) -> List[dict]:
    """n conversations; each question paired with a random matching answer, probe questions excluded."""
    rng = random.Random(seed)
    pool = ([(q, A_HI, A_HI_NOT_OTHER) for q in Q_HI] + [(q, A_EN, A_EN_NOT_OTHER) for q in Q_EN]
            + [(q, A_HINGLISH, A_HINGLISH_NOT_OTHER) for q in Q_HINGLISH])
    pool = [p for p in pool if norm(p[0]) not in probe_questions]
    out = []
    for i in range(n):
        q, normal, other = pool[i % len(pool)]
        a = rng.choice(_answers(q, normal, other))
        # small surface variety so the model learns the fact, not one string
        if rng.random() < 0.3:
            q = rng.choice(["", "नमस्ते! ", "Hi, ", "Hello! ", "भाई, "]) + q[0].lower() + q[1:] if q[0].isascii() else q
        out.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
    return out


# ── open datasets (Hugging Face) ─────────────────────────────────────────────

# repo -> (config, split, max rows)
HF_SOURCES = {
    "databricks/databricks-dolly-15k": (None, "train", None),        # English, human written
    "HuggingFaceH4/no_robots": (None, "train", None),                 # English, human written
    "ai4bharat/indic-align": ("Dolly_T", "train", None),              # Hindi (Dolly translated)
    "ai4bharat/indic-align@wiki": ("Wiki_Conv", "train", 80_000),     # Hindi general-knowledge chats
    "sarvamai/samvaad-hi-v1": (None, "train", 60_000),                # Hindi / Hinglish chat
}


def _to_messages(r: dict) -> Optional[List[dict]]:
    """Convert the common instruction-data layouts into [{'role','content'}, ...]."""
    turns = r.get("hin_Deva")                      # indic-align: [[question, answer], ...]
    if isinstance(turns, list) and turns:
        out = []
        for t in turns:
            if isinstance(t, (list, tuple)) and len(t) == 2 and t[0] and t[1]:
                out += [{"role": "user", "content": norm(t[0])}, {"role": "assistant", "content": norm(t[1])}]
        return out or None
    msgs = r.get("messages") or r.get("conversations") or r.get("conversation")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        out = []
        for m in msgs:
            role = str(m.get("role") or m.get("from") or "").lower()
            text = norm(m.get("content") or m.get("value") or "")
            role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
            if role in ("user", "assistant") and text:
                out.append({"role": role, "content": text})
        return out if len(out) >= 2 and out[0]["role"] == "user" else None
    q = r.get("instruction") or r.get("question") or r.get("prompt") or r.get("input")
    a = r.get("output") or r.get("response") or r.get("answer")
    ctx = r.get("context") or (r.get("input") if r.get("instruction") else "")
    if not q or not a:
        return None
    q = norm(q) + ("\n\n" + norm(ctx) if ctx else "")
    return [{"role": "user", "content": q}, {"role": "assistant", "content": norm(a)}]


def hf_rows(probe_questions: set) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("pip install datasets  — skipping Hugging Face sources")
        return
    for repo, (config, split, limit) in HF_SOURCES.items():
        try:
            ds = load_dataset(repo.split("@")[0], config, split=split, streaming=True)
        except Exception as e:  # noqa: BLE001 — any download problem just skips this set
            log.warning(f"Skipping {repo}: {str(e).splitlines()[0][:160]}")
            continue
        n = 0
        for r in ds:
            if limit and n >= limit:
                break
            msgs = _to_messages(r)
            if not msgs or norm(msgs[0]["content"]) in probe_questions:
                continue
            if any(bad_text(m["content"], 1) for m in msgs):
                continue
            yield {"messages": msgs}
            n += 1
        log.info(f"Added {n:,} rows from {repo}")


# ── build ────────────────────────────────────────────────────────────────────

def boost(data_dir: str, identity: int = 12_000, use_hf: bool = True, seed: int = 0) -> Dict:
    sft = os.path.join(data_dir, "sft.jsonl")
    base = os.path.join(data_dir, "sft_base.jsonl")
    if not os.path.isfile(base):
        if not os.path.isfile(sft):
            raise FileNotFoundError(f"{sft} not found — run  python main.py --mode data  first")
        os.replace(sft, base)
        log.info(f"Kept the original as {base}")

    probe_questions = set()
    probe = os.path.join(data_dir, "probe_50.jsonl")
    if os.path.isfile(probe):
        with open(probe, encoding="utf-8") as f:
            probe_questions = {norm(json.loads(l)["question"]) for l in f if l.strip()}

    out = _Buckets(sft, seed=seed)
    counts = {"base": 0, "identity": 0, "open_datasets": 0}
    with open(base, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.files[out.rng.randrange(out.n)].write(line if line.endswith("\n") else line + "\n")
                out.count += 1
                counts["base"] += 1
    for rec in identity_rows(identity, probe_questions, seed):
        out.add(rec)
        counts["identity"] += 1
    if use_hf:
        for rec in hf_rows(probe_questions):
            out.add(rec)
            counts["open_datasets"] += 1
    out.close()
    counts["total"] = out.count
    with open(os.path.join(data_dir, "boost_report.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=1)
    log.info(f"sft.jsonl rebuilt: {counts}")
    return counts


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Datasets"))
    p.add_argument("--identity", type=int, default=12_000, help="identity conversations to add")
    p.add_argument("--no-hf", action="store_true", help="skip downloading open datasets")
    a = p.parse_args()
    boost(a.data_dir, a.identity, not a.no_hf)
