"""Download open-source datasets for LLM training. Downloads from Hugging Face
and direct URLs, converts to unified JSONL format (system/user/assistant/category).

Usage:
    python tools/download_data.py --samples 50000
    python tools/download_data.py --list
    python tools/download_data.py --categories code,instruction
    python tools/download_data.py --force
"""

from __future__ import annotations

import gzip
import io
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    import pyarrow.parquet as pq
    import requests
    HAS_PQ = True
except ImportError:
    HAS_PQ = False

DATA_DIR = Path("Download")
SYSTEM_DEFAULT = "You are Atulya. Answer clearly and accurately."
SYSTEM_CODE = "You are Atulya. Write correct, clean code."
SYSTEM_REASON = "You are Atulya. Explain your reasoning step by step."


# ── Datasets registry ──

DATASETS: list[dict[str, Any]] = [
    # ═══════════════════════════════════════════════════════
    # INSTRUCTION
    # ═══════════════════════════════════════════════════════
    dict(id="databricks/databricks-dolly-15k", cat="instruction", split="train", samples=15_011,
         fields={"instruction": "instruction", "response": "response"}, fmt="instruction",
         desc="Dolly 15k: human-generated instruction responses"),
    dict(id="OpenAssistant/oasst1", cat="instruction", split="train", samples=30_000,
         fields={"text": "text", "role": "role"}, fmt="oasst",
         desc="OpenAssistant: human multi-turn conversations"),
    dict(id="HuggingFaceH4/ultrachat_200k", cat="instruction", split="train_sft", samples=50_000,
         fields={"messages": "messages"}, fmt="messages",
         desc="UltraChat: diverse synthetic chat"),
    dict(id="HuggingFaceTB/smoltalk", cat="instruction", config="all", split="train", samples=10_000,
         fields={"messages": "messages"}, fmt="messages",
         desc="SmolTalk: clean multi-turn chat"),
    dict(id="HuggingFaceH4/no_robots", cat="instruction", split="train", samples=5_000,
         fields={"messages": "messages"}, fmt="messages",
         desc="No Robots: clean SFT dataset"),
    dict(id="tatsu-lab/alpaca", cat="instruction", split="train", samples=10_000,
         fields={"instruction": "instruction", "response": "output"}, fmt="instruction",
         desc="Alpaca: instruction-following (Stanford)"),
    dict(id="teknium/OpenHermes-2.5", cat="instruction", split="train", samples=100_000,
         fields={"conversations": "conversations", "conversation": "conversations"}, fmt="sharegpt", use_parquet_fallback=True,
         desc="OpenHermes 2.5: 1M+ high-quality instructions (Teknium)"),
    dict(id="HuggingFaceTB/cosmopedia", cat="factual", config="auto_math_text", split="train", samples=100_000,
         fields={"text": "text", "markdown": "markdown", "topic": "topic"}, fmt="textbook",
         use_parquet_fallback=True,
         desc="Cosmopedia: clean synthetic textbook-style knowledge"),
    dict(id="yahma/alpaca-cleaned", cat="instruction", split="train", samples=50_000,
         fields={"instruction": "instruction", "response": "output"}, fmt="instruction",
         desc="Alpaca Cleaned: 52k cleaned Alpaca"),
    dict(id="Intel/orca_dpo_pairs", cat="instruction", split="train", samples=30_000,
         fields={"question": "question", "chosen": "chosen"}, fmt="qa",
         desc="Orca DPO Pairs: preference pairs from Orca"),
    dict(id="nvidia/HelpSteer", cat="instruction", split="train", samples=20_000,
         fields={"prompt": "prompt", "response": "response"}, fmt="qa",
         desc="HelpSteer: human-rated helpfulness responses"),
    dict(id="GAIR/lima", cat="instruction", split="train", samples=5_000,
         fields={"conversations": "conversations"}, fmt="sharegpt",
         desc="LIMA: less-is-more-alignment"),
    dict(id="WizardLM/WizardLM_evol_instruct_70k", cat="instruction", split="train", samples=50_000,
         fields={"instruction": "instruction", "output": "output"}, fmt="instruction",
         desc="WizardLM Evol-Instruct 70k"),
    dict(id="LDJnr/Pure-Dove", cat="instruction", split="train", samples=15_000,
         fields={"messages": "messages"}, fmt="messages",
         desc="Pure-Dove: clean general instruction & chat"),

    # ═══════════════════════════════════════════════════════
    # CODE
    # ═══════════════════════════════════════════════════════
    dict(id="google-research-datasets/mbpp", cat="code", split="train", samples=974,
         fields={"instruction": "text", "response": "code"}, fmt="instruction",
         desc="MBPP: Python programming problems"),
    dict(id="microsoft/orca-math-word-problems-200k", cat="code", split="train", samples=20_000,
         fields={"question": "question", "answer": "answer"}, fmt="qa",
         desc="Orca Math: math word problems"),
    dict(id="nuprl/MultiPL-E", cat="code", split="train", samples=10_000,
         fields={"instruction": "prompt", "response": "canonical_solution"}, fmt="instruction",
         desc="MultiPL-E: multi-language code exercises",
         note="may fail if no parquet fallback"),
    dict(id="bigcode/gigacode", cat="code", split="train", samples=10_000,
         fields={"instruction": "text", "response": "code"}, fmt="instruction", use_parquet_fallback=True,
         desc="GigaCode: synthetic Python coding problems"),
    dict(id="open-phi/programming_books_llama", cat="code", split="train", samples=100_000,
         fields={"text": "markdown", "topic": "topic"}, fmt="textbook",
         use_parquet_fallback=True,
         desc="Programming books: textbook-style code and CS reasoning"),
    dict(id="nampdn-ai/tiny-codes", cat="code", split="train", samples=50_000,
         fields={"instruction": "prompt", "response": "response", "code": "code", "text": "text"},
         fmt="flex_code", use_parquet_fallback=True,
         desc="Tiny Codes: compact coding instruction examples"),

    # ═══════════════════════════════════════════════════════
    # REASONING
    # ═══════════════════════════════════════════════════════
    dict(id="openai/gsm8k", cat="reasoning", config="main", split="train", samples=7_473,
         fields={"question": "question", "answer": "answer"}, fmt="qa",
         desc="GSM8K: grade school math"),
    dict(id="TIGER-Lab/MathInstruct", cat="reasoning", split="train", samples=50_000,
         fields={"instruction": "instruction", "response": "output"}, fmt="instruction", use_parquet_fallback=True,
         desc="MathInstruct: diverse math instruction data"),
    dict(id="math_qa", cat="reasoning", split="train", samples=10_000,
         fields={"question": "question", "answer": "answer"}, fmt="qa", use_parquet_fallback=True,
         desc="MathQA: math word problems"),
    dict(id="camel-ai/math", cat="reasoning", split="train", samples=50_000,
         fields={"question": "message_1", "answer": "message_2", "topic": "topic", "sub_topic": "sub_topic"},
         fmt="camel_math", use_parquet_fallback=True,
         desc="CAMEL Math: step-by-step math conversations"),
    dict(id="garage-bAInd/Open-Platypus", cat="reasoning", split="train", samples=25_000,
         fields={"instruction": "instruction", "input": "input", "response": "output"},
         fmt="instruction", use_parquet_fallback=True,
         desc="Open-Platypus: reasoning and STEM instruction data"),

    # ═══════════════════════════════════════════════════════
    # FACTUAL
    # ═══════════════════════════════════════════════════════
    dict(id="wikimedia/wikipedia", cat="factual", config="20231101.en", split="train", samples=200_000,
         fields={"title": "title", "text": "text"}, fmt="wikitext", use_parquet_fallback=True,
         desc="Wikipedia: encyclopedic articles"),
    dict(id="monology/pile-uncopyrighted", cat="general", split="train", samples=200_000,
         fields={"text": "text"}, fmt="raw_text", use_parquet_fallback=True,
         desc="The Pile: large diverse text corpus"),
    dict(id="wiki_qa", cat="factual", split="train", samples=5_000,
         fields={"question": "question", "answer": "answer"}, fmt="qa", use_parquet_fallback=True,
         desc="WikiQA: Wikipedia-based QA"),
    dict(id="wikiwhy", cat="factual", split="train", samples=5_000,
         fields={"question": "question", "answer": "answer"}, fmt="qa", use_parquet_fallback=True,
         desc="WikiWhy: open-ended why questions"),
    dict(id="openbookqa", cat="factual", split="train", samples=5_000,
         fields={"question_stem": "question_stem", "answerKey": "answerKey"}, fmt="qa", use_parquet_fallback=True,
         desc="OpenBookQA: science QA"),

    # ═══════════════════════════════════════════════════════
    # GENERAL / CHAT
    # ═══════════════════════════════════════════════════════
    dict(id="HuggingFaceFW/fineweb", cat="general", config="CC-MAIN-2024-10", split="train", samples=200_000,
         fields={"text": "text", "url": "url"}, fmt="raw_text", use_parquet_fallback=True,
         desc="FineWeb: filtered web text"),
    dict(id="HuggingFaceFW/fineweb-2", cat="general", config="CC-MAIN-2025-07", split="train", samples=200_000,
         fields={"text": "text", "url": "url"}, fmt="raw_text", use_parquet_fallback=True,
         desc="FineWeb-2: filtered web text 2025"),
    dict(id="mlfoundations/dclm-baseline-1.0-parquet", cat="general", config="default", split="train", samples=500_000,
         fields={"text": "text"}, fmt="raw_text", use_parquet_fallback=True,
         desc="DCLM: massive filtered web corpus"),
    dict(id="LDJnr/Capybara", cat="chat", split="train", samples=10_000,
         fields={"conversation": "conversation"}, fmt="sharegpt", use_parquet_fallback=True,
         desc="Capybara: diverse chat conversations"),

    # ═══════════════════════════════════════════════════════
    # NEW COGNITIVE LAYERS
    # ═══════════════════════════════════════════════════════
    dict(id="dair-ai/emotion", cat="emotion", split="train", samples=10_000,
         fields={"text": "text"}, fmt="raw_text", use_parquet_fallback=True,
         desc="Emotion: emotion classification dataset"),
    dict(id="agiresearch/RoboQA", cat="action", split="train", samples=5_000,
         fields={"question": "question", "answer": "answer"}, fmt="qa", use_parquet_fallback=True,
         desc="RoboQA: robotics and action planning"),
    dict(id="spatial_reasoning", cat="spatial", split="train", samples=5_000,
         fields={"question": "question", "answer": "answer"}, fmt="qa", use_parquet_fallback=True,
         desc="Spatial Reasoning: geometry and map routing"),
]


# ── Normalization ──

BAD_PATTERNS = re.compile(
    r"(?i)(thought:|action:|tool_name|api_name|<human>:|<bot>:|"
    r"available apis:|near me|current weather|weather today|"
    r"unsupported claims|connect the answer)"
)
PII_PATTERNS = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|"  # email
    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|"                         # phone
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"                             # credit card
)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    text = PII_PATTERNS.sub("[REDACTED]", text)
    return text


def is_quality(text: str, min_words: int = 5) -> bool:
    if len(text) < 20:
        return False
    if BAD_PATTERNS.search(text):
        return False
    words = re.findall(r"[A-Za-z]+", text)
    return len(words) >= min_words


# ── Converters ──

def norm(t: str) -> str:
    return clean_text(t)


def make(system: str, user: str, assistant: str, category: str) -> dict | None:
    user = norm(user)
    assistant = norm(assistant)
    if not user or not assistant or len(user) < 3 or len(assistant) < 3:
        return None
    if len(user) > 4096 or len(assistant) > 8192:
        return None
    return {"system": system, "user": user, "assistant": assistant, "category": category}


def convert_row(spec: dict, row: dict) -> list[dict]:
    fmt, cat = spec["fmt"], spec["cat"]
    f = spec["fields"]
    system = SYSTEM_CODE if cat == "code" else SYSTEM_REASON if cat == "reasoning" else SYSTEM_DEFAULT
    out = []

    if fmt == "instruction":
        inp = row.get(f["instruction"], "")
        out_ = row.get(f["response"], "")
        if inp and out_:
            out.append(make(system, inp, out_, cat))

    elif fmt == "qa":
        q = row.get(f["question"], "")
        a = row.get(f["answer"], "")
        if q and a:
            out.append(make(system, q, a, cat))

    elif fmt == "messages":
        msgs = row.get(f["messages"], [])
        for i in range(0, len(msgs) - 1, 2):
            u = msgs[i].get("content", "") if msgs[i].get("role") in ("user", "human") else ""
            a = msgs[i + 1].get("content", "") if i + 1 < len(msgs) and msgs[i + 1].get("role") in ("assistant", "bot", "gpt") else ""
            if u and a:
                out.append(make(system, u, a, cat))

    elif fmt == "sharegpt":
        convs = row.get(f.get("conversations", "conversations"), [])
        if not convs:
            convs = row.get(f.get("conversation", "conversation"), [])
        for i in range(0, len(convs) - 1, 2):
            u = convs[i].get("value", "") if convs[i].get("from") in ("user", "human") else convs[i].get("content", "")
            a = convs[i + 1].get("value", "") if i + 1 < len(convs) and convs[i + 1].get("from") in ("assistant", "bot", "gpt") else convs[i + 1].get("content", "")
            if u and a:
                out.append(make(system, u, a, cat))

    elif fmt == "oasst":
        text = row.get(f["text"], "")
        if text and is_quality(text):
            out.append(make(system, "Continue this conversation.", text, cat))

    elif fmt == "wikitext":
        text = row.get(f["text"], "")
        title = row.get(f["title"], "")
        if text and is_quality(text, 20):
            out.append(make(system, f"Tell me about {title}.", text[:3000], cat))

    elif fmt == "textbook":
        text = row.get(f.get("text", "text"), "") or row.get(f.get("markdown", "markdown"), "")
        topic = row.get(f.get("topic", "topic"), "")
        if text and is_quality(text, 20):
            prompt = f"Explain this topic clearly: {topic}" if topic else "Continue the following educational text:"
            out.append(make(system, prompt, str(text)[:3000], cat))

    elif fmt == "camel_math":
        question = row.get(f["question"], "")
        answer = row.get(f["answer"], "")
        topic = row.get(f.get("topic", "topic"), "")
        sub_topic = row.get(f.get("sub_topic", "sub_topic"), "")
        if question and answer and is_quality(str(answer), 5):
            user = str(question)
            if topic or sub_topic:
                user = f"{topic} {sub_topic}: {user}".strip()
            out.append(make(SYSTEM_REASON, user, str(answer), cat))

    elif fmt == "flex_code":
        instruction = row.get(f.get("instruction", "prompt"), "") or row.get("instruction", "")
        response = row.get(f.get("response", "response"), "") or row.get(f.get("code", "code"), "")
        text = row.get(f.get("text", "text"), "")
        if instruction and response and is_quality(str(response), 5):
            out.append(make(SYSTEM_CODE, str(instruction), str(response), cat))
        elif text and is_quality(str(text), 20):
            out.append(make(SYSTEM_CODE, "Continue this code or programming explanation:", str(text)[:3000], cat))

    elif fmt == "raw_text":
        text = row.get(f["text"], "")
        if text and is_quality(text, 20):
            out.append(make(system, "Continue the following text:", text[:3000], cat))

    return [r for r in out if r is not None]


# ── Parquet-based fallback downloader ──

PARQUET_CACHE_DIR = Path.home() / ".cache" / "hf_parquet_dl"


def _fetch_parquet_urls(ds_id: str) -> list[dict]:
    """Use HuggingFace Datasets Server API to discover Parquet file URLs."""
    api_url = f"https://datasets-server.huggingface.co/parquet?dataset={ds_id}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "opencode/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        files = data.get("parquet_files", [])
        return files
    except Exception as e:
        print(f"[parquet API fail] {e}")
        return []


def download_via_parquet(spec: dict, max_samples: int) -> list[dict]:
    """Download dataset directly via Datasets Server parquet URLs."""
    ds_id = spec["id"]
    print(f"  [parquet fallback] {ds_id}...", end=" ", flush=True)
    t0 = time.time()

    if not HAS_PQ:
        print(f"SKIP (need pyarrow+requests)")
        return []

    parquet_files = _fetch_parquet_urls(ds_id)
    if not parquet_files:
        print(f"NO parquet files found")
        return []

    # Determine target config/split
    target_config = spec.get("config")
    target_split = spec.get("split", "train")

    records: list[dict] = []
    seen: set = set()

    for pf in parquet_files:
        if len(records) >= max_samples:
            break
        cfg = pf.get("config", "default")
        spl = pf.get("split", "train")
        if target_config and cfg != target_config:
            continue
        if spl != target_split:
            continue

        url = pf.get("url", "")
        if not url:
            continue

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "opencode/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                table = pq.read_table(io.BytesIO(resp.read()))
        except Exception as e:
            print(f"  [parquet error {url[-40:]}] {e}")
            continue

        for batch in table.to_batches(max_chunksize=1024):
            if len(records) >= max_samples:
                break
            for i in range(batch.num_rows):
                row = {col: batch.column(col)[i].as_py() for col in batch.column_names}
                converted = convert_row(spec, row)
                for r in converted:
                    key = (r["user"][:80], r["assistant"][:80])
                    if key not in seen:
                        seen.add(key)
                        records.append(r)

    elapsed = time.time() - t0
    print(f"{len(records)} rows ({elapsed:.0f}s)")
    return records


# ── Main download ──

def download_and_convert(spec: dict, max_samples: int, out_dir: Path, force: bool = False) -> dict:
    ds_id = spec["id"]
    cat = spec["cat"]
    out_path = out_dir / cat / f"{ds_id.replace('/', '_')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        with open(out_path, "r", encoding="utf-8", errors="replace") as f:
            existing = sum(1 for _ in f)
        return {"status": "skipped", "id": ds_id, "rows": existing}

    samples = min(spec["samples"], max_samples)
    print(f"  [{cat}] {ds_id} ({samples} samples)...", end=" ", flush=True)
    t0 = time.time()
    records: list[dict] = []

    # Strategy 1: try load_dataset
    if load_dataset is not None:
        try:
            kwargs = {"split": spec["split"], "streaming": True}
            if spec.get("config"):
                kwargs["name"] = spec["config"]
            ds = load_dataset(ds_id, **kwargs)
            seen = set()
            for i, row in enumerate(ds):
                if len(records) >= samples:
                    break
                converted = convert_row(spec, row)
                for r in converted:
                    key = (r["user"][:80], r["assistant"][:80])
                    if key not in seen:
                        seen.add(key)
                        records.append(r)
        except Exception as e:
            print(f"\n  [load_dataset fail: {e}]", end=" ")

    # Strategy 2: try direct parquet fallback
    if not records and spec.get("use_parquet_fallback"):
        records = download_via_parquet(spec, samples)

    if not records:
        print(f"ZERO rows")
        return {"status": "error", "id": ds_id, "error": "no rows"}

    random.Random(42).shuffle(records)
    with open(out_path, "w", encoding="utf-8", newline="\n", errors="replace") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mb = out_path.stat().st_size / 1_000_000
    print(f"{len(records)} rows, {mb:.1f} MB ({time.time()-t0:.0f}s)")
    return {"status": "ok", "id": ds_id, "rows": len(records), "path": str(out_path)}


# ── CLI ──

def list_datasets():
    print(f"{'Dataset':45s} {'Cat':14s} {'Rows':>7s} {'Desc'}")
    print("-" * 90)
    for s in DATASETS:
        print(f"{s['id']:45s} {s['cat']:14s} {s['samples']:>7d} {s['desc'][:50]}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download opensource datasets for LLM training")
    parser.add_argument("--samples", type=int, default=50_000, help="Max per dataset")
    parser.add_argument("--categories", help="Comma-separated: code,instruction,reasoning,factual,general,chat")
    parser.add_argument("--dataset", help="Download specific dataset ID only")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    selected = list(DATASETS)
    if args.dataset:
        selected = [s for s in selected if args.dataset in s["id"]]
    elif args.categories:
        cats = {c.strip() for c in args.categories.split(",")}
        selected = [s for s in selected if s["cat"] in cats]

    if not selected:
        print("No matching datasets. Use --list to see available.")
        sys.exit(1)

    print(f"Downloading {len(selected)} datasets (max {args.samples}/dataset)...\n")
    results = []
    for spec in selected:
        r = download_and_convert(spec, args.samples, DATA_DIR, args.force)
        results.append(r)

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]
    total_rows = sum(r["rows"] for r in ok) + sum(r.get("rows", 0) for r in skipped)

    print("\n" + "=" * 50)
    print(f"OK: {len(ok)}  Skipped: {len(skipped)}  Errors: {len(errors)}")
    print(f"Total rows across all: {total_rows:,}")
    print(f"\nNext: python tools/clean_and_combine.py --max-rows 500000")


if __name__ == "__main__":
    main()
