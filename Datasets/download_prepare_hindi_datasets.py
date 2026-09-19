#!/usr/bin/env python3
"""
Download and prepare Hindi datasets from 6 open-source sources.

Sources:
  1. AI4Bharat IndicCorp v2 (Hindi) — tokenizer training
  2. OSCAR / CC-100 Hindi — tokenizer + base LM diversity
  3. AI4Bharat IndicInstruct / Cohere Aya — conversation fine-tune
  4. AI4Bharat Samanantar — En-Hi translation pairs
  5. Hindi Wikipedia dump — general knowledge
  6. CodeAlpaca-Hindi — code with Hindi explanations

Usage:
  pip install datasets huggingface_hub requests tqdm
  python Datasets/download_prepare_hindi_datasets.py          # download all
  python Datasets/download_prepare_hindi_datasets.py --only indicorp  # one source
  python Datasets/download_prepare_hindi_datasets.py --tokenizer-only  # just #1 and #2

Output goes to Datasets/raw/ as .jsonl files, then this script
builds the final expert_*.jsonl files from them.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).parent
RAW_DIR = DATASETS_DIR / "raw"
RAW_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# HuggingFace dataset configs
# Each entry: (source_key, hf_dataset_id, hf_subset, split, max_rows, output_name)
# ---------------------------------------------------------------------------
HF_SOURCES = {
    "indicorp": {
        "dataset_id": "ai4bharat/IndicCorpV2",
        "subset": "hin_Deva",
        "split": "train",
        "max_rows": 500_000,
        "output": "indicorp_hindi.jsonl",
        "text_field": "text",
        "description": "AI4Bharat IndicCorp v2 — Hindi monolingual web text (tokenizer)",
    },
    "oscar": {
        "dataset_id": "oscar-corpus/OSCAR-2301",
        "subset": "hi",
        "split": "train",
        "max_rows": 500_000,
        "output": "oscar_hindi.jsonl",
        "text_field": "text",
        "description": "OSCAR 2301 — Hindi (tokenizer + base LM)",
    },
    "cc100": {
        "dataset_id": "cc100",
        "subset": "hi",
        "split": "train",
        "max_rows": 500_000,
        "output": "cc100_hindi.jsonl",
        "text_field": "text",
        "description": "CC-100 Hindi (tokenizer diversity)",
    },
    "indicinstruct": {
        "dataset_id": "ai4bharat/IndicInstruct",
        "subset": None,  # iterate all Hindi subsets
        "split": "train",
        "max_rows": 100_000,
        "output": "indicinstruct_hindi.jsonl",
        "description": "AI4Bharat IndicInstruct — Hindi instruction following",
    },
    "aya": {
        "dataset_id": "CohereForMultilingual/aya_dataset",
        "subset": None,
        "split": "train",
        "max_rows": 100_000,
        "output": "aya_hindi.jsonl",
        "description": "Cohere Aya — Hindi conversation/instruction",
    },
    "samanantar": {
        "dataset_id": "ai4bharat/samanantar",
        "subset": None,
        "split": "train",
        "max_rows": 200_000,
        "output": "samanantar_hindi.jsonl",
        "description": "AI4Bharat Samanantar — En-Hi translation pairs",
    },
}

# ---------------------------------------------------------------------------
# Wikipedia dump URL
# ---------------------------------------------------------------------------
WIKI_DUMP_URL = "https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles1.xml-p1p41242.bz2"
WIKI_OUTPUT = "wikipedia_hindi.jsonl"

# CodeAlpaca — we download the English version and generate Hindi explanations
CODEALPACA_HF = "tatsu-lab/alpaca"
CODEALPACA_OUTPUT = "codealpaca_hindi.jsonl"


# ===========================================================================
# Helpers
# ===========================================================================

def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _is_hindi_heavy(text: str) -> bool:
    """Return True if >= 30% of non-space characters are Devanagari."""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 10:
        return False
    deva = sum(1 for c in chars if "\u0900" <= c <= "\u097F")
    return deva / len(chars) >= 0.30


def _write_jsonl(path: Path, rows: Iterator[dict]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count % 10_000 == 0:
                log.info(f"  Written {count:,} rows to {path.name}")
    return count


def _load_existing(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


# ===========================================================================
# Source 1: IndicCorp v2
# ===========================================================================

def download_indicorp(max_rows: int = 500_000) -> Path:
    """Download Hindi text from AI4Bharat IndicCorp v2."""
    out = RAW_DIR / "indicorp_hindi.jsonl"
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading IndicCorp v2 (Hindi)...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "ai4bharat/IndicCorpV2",
            "hin_Deva",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        log.warning(f"  HuggingFace load failed: {e}")
        log.info("  Trying direct URL fallback...")
        return _download_text_fallback(
            "https://objectstore.e2core.dev/indicnlp/indiccorp/v2/2023-04-27/indiccorp.hin_Deva.tar.gz",
            out, max_rows
        )

    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            text = str(row.get("text", "")).strip()
            if len(text) < 50 or not _is_hindi_heavy(text):
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_rows:
                break
            if count % 50_000 == 0:
                log.info(f"    {count:,} rows written...")

    log.info(f"  IndicCorp: {count:,} rows -> {out.name}")
    return out


# ===========================================================================
# Source 2: OSCAR Hindi
# ===========================================================================

def download_oscar(max_rows: int = 500_000) -> Path:
    out = RAW_DIR / "oscar_hindi.jsonl"
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading OSCAR 2301 (Hindi)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("oscar-corpus/OSCAR-2301", "hi", split="train", streaming=True)
    except Exception as e:
        log.warning(f"  HuggingFace load failed: {e}")
        return out

    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            text = str(row.get("text", "")).strip()
            if len(text) < 50 or not _is_hindi_heavy(text):
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_rows:
                break
            if count % 50_000 == 0:
                log.info(f"    {count:,} rows written...")

    log.info(f"  OSCAR: {count:,} rows -> {out.name}")
    return out


# ===========================================================================
# Source 2b: CC-100 Hindi
# ===========================================================================

def download_cc100(max_rows: int = 500_000) -> Path:
    out = RAW_DIR / "cc100_hindi.jsonl"
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading CC-100 (Hindi)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("cc100", "hi", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        log.warning(f"  HuggingFace load failed: {e}")
        return out

    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            text = str(row.get("text", "")).strip()
            if len(text) < 50 or not _is_hindi_heavy(text):
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_rows:
                break
            if count % 50_000 == 0:
                log.info(f"    {count:,} rows written...")

    log.info(f"  CC-100: {count:,} rows -> {out.name}")
    return out


# ===========================================================================
# Source 3a: IndicInstruct
# ===========================================================================

def download_indicinstruct(max_rows: int = 100_000) -> Path:
    out = RAW_DIR / "indicinstruct_hindi.jsonl"
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading IndicInstruct (Hindi subsets)...")
    try:
        from datasets import load_dataset
        # IndicInstruct has multiple subsets; try the main Hindi one
        for subset in ["indic-instruct-hindi", "hin_Deva", "hindi"]:
            try:
                ds = load_dataset("ai4bharat/IndicInstruct", subset, split="train", streaming=True)
                break
            except Exception:
                continue
        else:
            # Try without subset
            ds = load_dataset("ai4bharat/IndicInstruct", split="train", streaming=True)
    except Exception as e:
        log.warning(f"  HuggingFace load failed: {e}")
        return out

    count = 0
    first_row_keys = None
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            if first_row_keys is None:
                first_row_keys = list(row.keys())
                log.info(f"  IndicInstruct first row keys: {first_row_keys}")
            text = str(row.get("text", row.get("input", row.get("instruction", "")))).strip()
            if not text or len(text) < 20:
                continue
            # Format as conversation
            messages = []
            if "instruction" in row:
                messages.append({"role": "user", "content": str(row["instruction"])})
                if row.get("output"):
                    messages.append({"role": "assistant", "content": str(row["output"])})
            elif "prompt" in row and "response" in row:
                messages.append({"role": "user", "content": str(row["prompt"])})
                messages.append({"role": "assistant", "content": str(row["response"])})
            elif "source" in row and "target" in row:
                messages.append({"role": "user", "content": str(row["source"])})
                messages.append({"role": "assistant", "content": str(row["target"])})
            else:
                messages.append({"role": "user", "content": text})

            f.write(json.dumps({"messages": messages, "domain": "conversation"}, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_rows:
                break
            if count % 10_000 == 0:
                log.info(f"    {count:,} rows written...")

    if count == 0:
        log.warning(f"  WARNING: IndicInstruct produced 0 rows! Field names may not match. First row keys: {first_row_keys}")
    log.info(f"  IndicInstruct: {count:,} rows -> {out.name}")
    return out


# ===========================================================================
# Source 3b: Cohere Aya
# ===========================================================================

def download_aya(max_rows: int = 100_000) -> Path:
    out = RAW_DIR / "aya_hindi.jsonl"
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading Cohere Aya (Hindi)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("CohereForMultilingual/aya_dataset", split="train", streaming=True)
    except Exception as e:
        log.warning(f"  HuggingFace load failed: {e}")
        return out

    count = 0
    first_row_keys = None
    hindi_hit = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            if first_row_keys is None:
                first_row_keys = list(row.keys())
                log.info(f"  Aya first row keys: {first_row_keys}")
            lang = str(row.get("language", row.get("lang", ""))).lower()
            if "hindi" in lang or "hi" in lang:
                hindi_hit += 1
            inputs = str(row.get("inputs", row.get("input", row.get("prompt", "")))).strip()
            targets = str(row.get("targets", row.get("target", row.get("response", "")))).strip()
            if not inputs or not targets:
                continue
            messages = [
                {"role": "user", "content": inputs},
                {"role": "assistant", "content": targets},
            ]
            f.write(json.dumps({"messages": messages, "domain": "conversation"}, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_rows:
                break
            if count % 10_000 == 0:
                log.info(f"    {count:,} rows written...")

    if count == 0:
        log.warning(f"  WARNING: Aya produced 0 rows! Field names may not match. First row keys: {first_row_keys}")
        log.warning(f"  Hindi language matches seen: {hindi_hit} (but may have had empty inputs/targets)")
    log.info(f"  Aya: {count:,} rows -> {out.name}")
    return out


# ===========================================================================
# Source 4: Samanantar (En-Hi translation)
# ===========================================================================

def download_samanantar(max_rows: int = 200_000) -> Path:
    out = RAW_DIR / "samanantar_hindi.jsonl"
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading Samanantar (En-Hi translation)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("ai4bharat/samanantar", split="train", streaming=True)
    except Exception as e:
        log.warning(f"  HuggingFace load failed: {e}")
        return out

    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            src = str(row.get("src", row.get("en", row.get("source", "")))).strip()
            tgt = str(row.get("tgt", row.get("hi", row.get("target", "")))).strip()
            if not src or not tgt or len(src) < 10 or len(tgt) < 10:
                continue
            messages = [
                {"role": "user", "content": f"Translate this to Hindi:\n{src}"},
                {"role": "assistant", "content": tgt},
            ]
            f.write(json.dumps({"messages": messages, "domain": "translation"}, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_rows:
                break
            if count % 10_000 == 0:
                log.info(f"    {count:,} rows written...")

    log.info(f"  Samanantar: {count:,} rows -> {out.name}")
    return out


# ===========================================================================
# Source 5: Hindi Wikipedia
# ===========================================================================

def download_wikipedia(max_rows: int = 100_000) -> Path:
    """Download Hindi Wikipedia via HuggingFace wikipedia dataset."""
    out = RAW_DIR / WIKI_OUTPUT
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading Hindi Wikipedia...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikipedia", "20231101.hi", split="train", streaming=True)
    except Exception:
        try:
            ds = load_dataset("wikipedia", "20220301.hi", split="train", streaming=True)
        except Exception as e:
            log.warning(f"  HuggingFace load failed: {e}")
            return out

    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            title = str(row.get("title", "")).strip()
            text = str(row.get("text", "")).strip()
            if not title or not text or len(text) < 200:
                continue
            # Format as a general knowledge Q&A
            messages = [
                {"role": "user", "content": f"{title} के बारे में बताइए।"},
                {"role": "assistant", "content": text[:4000]},  # cap at 4k chars
            ]
            f.write(json.dumps({"messages": messages, "domain": "general"}, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_rows:
                break
            if count % 10_000 == 0:
                log.info(f"    {count:,} rows written...")

    log.info(f"  Wikipedia: {count:,} rows -> {out.name}")
    return out


# ===========================================================================
# Source 6: CodeAlpaca-Hindi
# ===========================================================================

def download_codealpaca(max_rows: int = 50_000) -> Path:
    """Download CodeAlpaca and convert to Hindi explanations + English code."""
    out = RAW_DIR / CODEALPACA_OUTPUT
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading CodeAlpaca...")
    try:
        from datasets import load_dataset
        ds = load_dataset(CODEALPACA_HF, split="train", streaming=True)
    except Exception as e:
        log.warning(f"  HuggingFace load failed: {e}")
        return out

    # Honest labeling: Hindi prompt wrapper + original English instruction + English code.
    # No fake translation — the English instruction is kept as-is because the code
    # output is English code anyway, and word-by-word Hindi substitution produces
    # gibberish.  The Hindi wrapper text is genuine Hindi that teaches the model
    # to respond to Hindi prompts with code.
    CODE_PROMPT_TEMPLATE = (
        "निम्नलिखित प्रोग्रामिंग समस्या का समाधान Python कोड में लिखें।\n"
        "समस्या: {instruction}\n\n"
        "कोड:"
    )

    def _honest_hindi_prompt(instruction: str) -> str:
        return CODE_PROMPT_TEMPLATE.format(instruction=instruction)

    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            instruction = str(row.get("instruction", "")).strip()
            output = str(row.get("output", "")).strip()
            if not instruction or not output or len(output) < 20:
                continue
            hindi_prompt = _honest_hindi_prompt(instruction)
            messages = [
                {"role": "user", "content": hindi_prompt},
                {"role": "assistant", "content": output},
            ]
            f.write(json.dumps({"messages": messages, "domain": "code"}, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_rows:
                break
            if count % 10_000 == 0:
                log.info(f"    {count:,} rows written...")

    log.info(f"  CodeAlpaca-Hindi: {count:,} rows -> {out.name}")
    return out


# ===========================================================================
# Fallback: direct text download
# ===========================================================================

def _download_text_fallback(url: str, out: Path, max_rows: int) -> Path:
    """Download a text file, extract lines, save as JSONL."""
    import requests
    import gzip
    import io

    log.info(f"  Downloading from {url[:80]}...")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"  Download failed: {e}")
        return out

    count = 0
    content_type = resp.headers.get("content-type", "")

    if "gzip" in url or "gzip" in content_type or url.endswith(".gz"):
        with gzip.open(resp.raw, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                text = line.strip()
                if len(text) < 50 or not _is_hindi_heavy(text):
                    continue
                with open(out, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1
                if count >= max_rows:
                    break
                if count % 50_000 == 0:
                    log.info(f"    {count:,} rows written...")
    else:
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            text = line.strip()
            if len(text) < 50 or not _is_hindi_heavy(text):
                continue
            with open(out, "a", encoding="utf-8") as fout:
                fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_rows:
                break
            if count % 50_000 == 0:
                log.info(f"    {count:,} rows written...")

    log.info(f"  Fallback: {count:,} rows -> {out.name}")
    return out


# ===========================================================================
# Build expert files from raw data
# ===========================================================================

def build_expert_files():
    """Merge raw downloaded data into expert_*.jsonl files."""
    log.info("=" * 60)
    log.info("Building expert files from raw data...")

    # --- expert_general.jsonl ---
    general_rows = []
    wiki_path = RAW_DIR / WIKI_OUTPUT
    if wiki_path.exists():
        with open(wiki_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        general_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        log.info(f"  Wikipedia: {len(general_rows):,} rows for expert_general")

    # Also include any raw text rows as general knowledge
    for raw_file in ["indicorp_hindi.jsonl", "oscar_hindi.jsonl", "cc100_hindi.jsonl"]:
        p = RAW_DIR / raw_file
        if p.exists():
            count = 0
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = row.get("text", "")
                    if text and len(text) > 200:
                        general_rows.append({
                            "messages": [
                                {"role": "user", "content": "इसके बारे में जानकारी दें।"},
                                {"role": "assistant", "content": text[:4000]},
                            ],
                            "domain": "general",
                        })
                        count += 1
                        if count >= 5000:
                            break
            log.info(f"  {raw_file}: contributed {count:,} rows to expert_general")

    # Write expert_general.jsonl
    general_out = DATASETS_DIR / "expert_general.jsonl"
    with open(general_out, "w", encoding="utf-8") as f:
        for row in general_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"  expert_general.jsonl: {len(general_rows):,} rows")

    # --- expert_code.jsonl ---
    code_rows = []
    code_path = RAW_DIR / CODEALPACA_OUTPUT
    if code_path.exists():
        with open(code_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        code_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    code_out = DATASETS_DIR / "expert_code.jsonl"
    with open(code_out, "w", encoding="utf-8") as f:
        for row in code_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"  expert_code.jsonl: {len(code_rows):,} rows")

    # --- expert_conversation.jsonl ---
    conv_rows = []
    for raw_file in ["indicinstruct_hindi.jsonl", "aya_hindi.jsonl", "samanantar_hindi.jsonl"]:
        p = RAW_DIR / raw_file
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            conv_rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    conv_out = DATASETS_DIR / "expert_conversation.jsonl"
    existing_conv = []
    if conv_out.exists():
        with open(conv_out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_conv.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    # Merge, deduplicate by first user message
    seen = set()
    all_conv = []
    for row in existing_conv + conv_rows:
        msgs = row.get("messages", [])
        key = msgs[0].get("content", "")[:100] if msgs else ""
        if key and key not in seen:
            seen.add(key)
            all_conv.append(row)
    with open(conv_out, "w", encoding="utf-8") as f:
        for row in all_conv:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"  expert_conversation.jsonl: {len(all_conv):,} rows (deduplicated)")

    # --- Hindi conversation train/val split ---
    hindi_conv = [r for r in all_conv if _is_hindi_row(r)]
    random.seed(42)
    random.shuffle(hindi_conv)
    split_idx = int(len(hindi_conv) * 0.95)
    hindi_train = hindi_conv[:split_idx]
    hindi_val = hindi_conv[split_idx:]

    hindi_train_out = DATASETS_DIR / "tantra_hindi_conversation_train.jsonl"
    hindi_val_out = DATASETS_DIR / "tantra_hindi_conversation_val.jsonl"
    with open(hindi_train_out, "w", encoding="utf-8") as f:
        for row in hindi_train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(hindi_val_out, "w", encoding="utf-8") as f:
        for row in hindi_val:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"  tantra_hindi_conversation_train.jsonl: {len(hindi_train):,} rows")
    log.info(f"  tantra_hindi_conversation_val.jsonl: {len(hindi_val):,} rows")


def _is_hindi_row(row: dict) -> bool:
    """Check if a row's content is predominantly Hindi."""
    msgs = row.get("messages", [])
    for msg in msgs:
        content = msg.get("content", "")
        if _is_hindi_heavy(content):
            return True
    return False


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Download and prepare Hindi datasets")
    parser.add_argument("--only", type=str, nargs="*",
                        choices=list(HF_SOURCES.keys()) + ["wikipedia", "codealpaca"],
                        help="Only download specific sources")
    parser.add_argument("--tokenizer-only", action="store_true",
                        help="Only download tokenizer sources (indicorp, oscar, cc100)")
    parser.add_argument("--build-only", action="store_true",
                        help="Skip downloads, just build expert files from existing raw data")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Override max rows per source")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Hindi Dataset Download & Preparation")
    log.info("=" * 60)

    if not args.build_only:
        sources = args.only or list(HF_SOURCES.keys()) + ["wikipedia", "codealpaca"]

        if args.tokenizer_only:
            sources = ["indicorp", "oscar", "cc100"]

        download_funcs = {
            "indicorp": lambda: download_indicorp(args.max_rows or 500_000),
            "oscar": lambda: download_oscar(args.max_rows or 500_000),
            "cc100": lambda: download_cc100(args.max_rows or 500_000),
            "indicinstruct": lambda: download_indicinstruct(args.max_rows or 100_000),
            "aya": lambda: download_aya(args.max_rows or 100_000),
            "samanantar": lambda: download_samanantar(args.max_rows or 200_000),
            "wikipedia": lambda: download_wikipedia(args.max_rows or 100_000),
            "codealpaca": lambda: download_codealpaca(args.max_rows or 50_000),
        }

        for src in sources:
            log.info(f"\n--- {src} ---")
            try:
                download_funcs[src]()
            except Exception as e:
                log.error(f"  FAILED: {e}")

    build_expert_files()

    log.info("\n" + "=" * 60)
    log.info("Done! Final file sizes:")
    empty_warnings = []
    for f in sorted(DATASETS_DIR.glob("*.jsonl")):
        if f.name.startswith("."):
            continue
        count = _line_count(f)
        size_kb = f.stat().st_size / 1024
        log.info(f"  {f.name:50s} {count:8,} rows  {size_kb:8.1f} KB")
        if count == 0 and f.name.startswith("expert_"):
            empty_warnings.append(f.name)

    # Also check raw files
    for f in sorted(RAW_DIR.glob("*.jsonl")):
        count = _line_count(f)
        if count == 0:
            empty_warnings.append(f"raw/{f.name}")

    if empty_warnings:
        log.warning("")
        log.warning("WARNING: The following files have 0 rows — something went wrong:")
        for w in empty_warnings:
            log.warning(f"  - {w}")
        log.warning("Check the HF dataset ID, subset name, and field names above.")
        log.warning("Look for 'first row keys' logs to see actual field names.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
