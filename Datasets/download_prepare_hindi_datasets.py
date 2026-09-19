#!/usr/bin/env python3
"""
Download and prepare Hindi datasets from open-source sources.

VERIFIED working sources (all tested 2026-09-20):
  1. AI4Bharat IndicCorp v2 (Hindi) — tokenizer training
  2. KathirKs/fineweb-edu-hindi — tokenizer + base LM (replaces gated OSCAR + deprecated cc100)
  3. AI4Bharat IndicInstruct v0.1 — conversation fine-tune (dolly/flan_v2 configs, hi split)
  4. FreedomIntelligence/evol-instruct-hindi — Hindi instruction following
  5. AI4Bharat Samanantar — En-Hi translation pairs
  6. Hindi Wikipedia dump — general knowledge
  7. CodeAlpaca — code with Hindi prompt wrappers

Usage:
  pip install datasets huggingface_hub requests tqdm
  python Datasets/download_prepare_hindi_datasets.py              # download all
  python Datasets/download_prepare_hindi_datasets.py --only indicorp  # one source
  python Datasets/download_prepare_hindi_datasets.py --tokenizer-only  # just #1 and #2
  python Datasets/download_prepare_hindi_datasets.py --build-only  # rebuild expert files from raw/

Output goes to Datasets/raw/ as .jsonl files, then this script
builds the final expert_*.jsonl files from them.
"""

import argparse
import hashlib
import io
import json
import logging
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# On Windows, IndicCorpV2 contains bytes that fail with cp1252 (the default
# Windows codec).  Setting PYTHONUTF8=1 forces UTF-8 decoding for all I/O.
# This must happen BEFORE any other import touches the codec system, so we
# re-exec ourselves with the env var if it's not already set.
if sys.platform == "win32" and os.environ.get("PYTHONUTF8") != "1":
    os.environ["PYTHONUTF8"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    result = subprocess.run([sys.executable] + sys.argv, env=os.environ)
    sys.exit(result.returncode)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).parent
RAW_DIR = DATASETS_DIR / "raw"
RAW_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Source 1: IndicCorp v2 (Hindi monolingual — tokenizer)
# ===========================================================================

def download_indicorp(max_rows: int = 500_000) -> Path:
    """Download Hindi text from AI4Bharat IndicCorp v2.
    Config: indiccorp_v2, Split: hin_Deva, Field: text"""
    out = RAW_DIR / "indicorp_hindi.jsonl"
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading IndicCorp v2 (Hindi)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="hin_Deva", streaming=True)
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

    log.info(f"  IndicCorp: {count:,} rows -> {out.name}")
    return out


# ===========================================================================
# Source 2: fineweb-edu-hindi (replaces gated OSCAR + deprecated cc100)
# ===========================================================================

def download_fineweb_hindi(max_rows: int = 500_000) -> Path:
    """Download Hindi educational web text. Verified keys: text, uuid, meta_data."""
    out = RAW_DIR / "fineweb_hindi.jsonl"
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading fineweb-edu-hindi...")
    try:
        from datasets import load_dataset
        ds = load_dataset("KathirKs/fineweb-edu-hindi", split="train", streaming=True)
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

    log.info(f"  fineweb-hindi: {count:,} rows -> {out.name}")
    return out


# ===========================================================================
# Source 3a: IndicInstruct v0.1 — dolly/hi (conversation)
# ===========================================================================

def download_indicinstruct(max_rows: int = 100_000) -> Path:
    """Download Hindi instruction data from IndicInstruct v0.1.
    Configs: dolly, flan_v2 — both have 'hi' split.
    dolly keys: id, category, instruction, context, response, backtranslated_*, quality_metrics
    flan_v2 keys: id, inputs, targets, backtranslated_*, quality_metrics, metadata"""
    out = RAW_DIR / "indicinstruct_hindi.jsonl"
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading IndicInstruct v0.1 (Hindi)...")
    from datasets import load_dataset

    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for config in ["dolly", "flan_v2"]:
            try:
                ds = load_dataset("ai4bharat/indic-instruct-data-v0.1", config, split="hi", streaming=True)
            except Exception as e:
                log.warning(f"  IndicInstruct/{config}/hi failed: {e}")
                continue

            config_count = 0
            for row in ds:
                messages = []
                if config == "dolly":
                    instruction = str(row.get("instruction", "")).strip()
                    response = str(row.get("response", "")).strip()
                    context = str(row.get("context", "")).strip()
                    if not instruction or not response:
                        continue
                    user_msg = instruction
                    if context:
                        user_msg = f"संदर्भ: {context}\n\n{instruction}"
                    messages = [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": response},
                    ]
                elif config == "flan_v2":
                    inputs = str(row.get("inputs", "")).strip()
                    targets = str(row.get("targets", "")).strip()
                    if not inputs or not targets:
                        continue
                    messages = [
                        {"role": "user", "content": inputs},
                        {"role": "assistant", "content": targets},
                    ]

                f.write(json.dumps({"messages": messages, "domain": "conversation"}, ensure_ascii=False) + "\n")
                count += 1
                config_count += 1
                if count >= max_rows:
                    break
                if config_count % 10_000 == 0:
                    log.info(f"    {config}: {config_count:,} rows...")

            log.info(f"  IndicInstruct/{config}: {config_count:,} rows")
            if count >= max_rows:
                break

    if count == 0:
        log.warning(f"  WARNING: IndicInstruct produced 0 rows!")
    log.info(f"  IndicInstruct total: {count:,} rows -> {out.name}")
    return out


# ===========================================================================
# Source 3b: FreedomIntelligence/evol-instruct-hindi
# ===========================================================================

def download_evol_hindi(max_rows: int = 100_000) -> Path:
    """Download Hindi instruction data. Verified keys: conversations, id.
    conversations is a list of {from, value} dicts."""
    out = RAW_DIR / "evol_hindi.jsonl"
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading FreedomIntelligence/evol-instruct-hindi...")
    try:
        from datasets import load_dataset
        ds = load_dataset("FreedomIntelligence/evol-instruct-hindi", split="train", streaming=True)
    except Exception as e:
        log.warning(f"  HuggingFace load failed: {e}")
        return out

    count = 0
    first_row_keys = None
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            if first_row_keys is None:
                first_row_keys = list(row.keys())
                log.info(f"  evol-hindi first row keys: {first_row_keys}")

            convs = row.get("conversations", [])
            if not convs or len(convs) < 2:
                continue

            messages = []
            for turn in convs:
                role = str(turn.get("from", "")).lower()
                value = str(turn.get("value", "")).strip()
                if not value:
                    continue
                if role in ("human", "user"):
                    messages.append({"role": "user", "content": value})
                elif role in ("gpt", "assistant", "chatgpt"):
                    messages.append({"role": "assistant", "content": value})

            if len(messages) < 2:
                continue

            f.write(json.dumps({"messages": messages, "domain": "conversation"}, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_rows:
                break
            if count % 10_000 == 0:
                log.info(f"    {count:,} rows written...")

    if count == 0:
        log.warning(f"  WARNING: evol-hindi produced 0 rows! First row keys: {first_row_keys}")
    log.info(f"  evol-hindi: {count:,} rows -> {out.name}")
    return out


# ===========================================================================
# Source 4: Samanantar (En-Hi translation)
# ===========================================================================

def download_samanantar(max_rows: int = 200_000) -> Path:
    """Download Samanantar translation pairs. Verified keys: idx, src, tgt. Config: hi."""
    out = RAW_DIR / "samanantar_hindi.jsonl"
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading Samanantar (En-Hi translation)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("ai4bharat/samanantar", "hi", split="train", streaming=True)
    except Exception as e:
        log.warning(f"  HuggingFace load failed: {e}")
        return out

    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            src = str(row.get("src", "")).strip()
            tgt = str(row.get("tgt", "")).strip()
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
    """Download Hindi Wikipedia. Verified dataset: wikimedia/wikipedia, config: 20231101.hi.
    Verified keys: id, url, title, text"""
    out = RAW_DIR / "wikipedia_hindi.jsonl"
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading Hindi Wikipedia...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
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
            messages = [
                {"role": "user", "content": f"{title} के बारे में बताइए।"},
                {"role": "assistant", "content": text[:4000]},
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
# Source 6: CodeAlpaca (Hindi prompt wrapper + English code)
# ===========================================================================

def download_codealpaca(max_rows: int = 50_000) -> Path:
    """Download CodeAlpaca with honest Hindi prompt wrapper.
    Verified keys: instruction, input, output, text"""
    out = RAW_DIR / "codealpaca_hindi.jsonl"
    if out.exists() and _line_count(out) >= max_rows * 0.9:
        log.info(f"  [SKIP] {out.name} already has {_line_count(out):,} rows")
        return out

    log.info("Downloading CodeAlpaca...")
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    except Exception as e:
        log.warning(f"  HuggingFace load failed: {e}")
        return out

    CODE_PROMPT_TEMPLATE = (
        "निम्नलिखित प्रोग्रामिंग समस्या का समाधान Python कोड में लिखें।\n"
        "समस्या: {instruction}\n\n"
        "कोड:"
    )

    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            instruction = str(row.get("instruction", "")).strip()
            output = str(row.get("output", "")).strip()
            if not instruction or not output or len(output) < 20:
                continue
            # Filter to code-related instructions only
            lower = instruction.lower()
            code_keywords = ["write", "create", "implement", "code", "function", "program",
                             "script", "class", "algorithm", "sort", "find", "calculate",
                             "python", "java", "c++", "sql", "array", "list", "string",
                             "def ", "import ", "return", "loop", "print"]
            if not any(kw in lower for kw in code_keywords):
                continue
            hindi_prompt = CODE_PROMPT_TEMPLATE.format(instruction=instruction)
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
# Helpers
# ===========================================================================

def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _is_hindi_heavy(text: str) -> bool:
    """Return True if >= 30% of non-space characters are Devanagari."""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 10:
        return False
    deva = sum(1 for c in chars if "\u0900" <= c <= "\u097F")
    return deva / len(chars) >= 0.30


def _is_hindi_row(row: dict) -> bool:
    """Check if a row's content is predominantly Hindi."""
    msgs = row.get("messages", [])
    for msg in msgs:
        if _is_hindi_heavy(msg.get("content", "")):
            return True
    return False


# ===========================================================================
# Build expert files from raw data
# ===========================================================================

def build_expert_files():
    """Merge raw downloaded data into expert_*.jsonl files."""
    log.info("=" * 60)
    log.info("Building expert files from raw data...")

    # --- expert_general.jsonl (Wikipedia + fineweb + indicorp) ---
    general_rows = []

    # Wikipedia -> general knowledge Q&A
    wiki_path = RAW_DIR / "wikipedia_hindi.jsonl"
    if wiki_path.exists():
        with open(wiki_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        general_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        log.info(f"  Wikipedia: {len(general_rows):,} rows for expert_general")

    # Raw text -> general knowledge
    for raw_file in ["indicorp_hindi.jsonl", "fineweb_hindi.jsonl"]:
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

    general_out = DATASETS_DIR / "expert_general.jsonl"
    with open(general_out, "w", encoding="utf-8") as f:
        for row in general_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"  expert_general.jsonl: {len(general_rows):,} rows")

    # --- expert_code.jsonl ---
    code_rows = []
    code_path = RAW_DIR / "codealpaca_hindi.jsonl"
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

    # --- expert_conversation.jsonl (IndicInstruct + evol + samanantar) ---
    conv_rows = []
    for raw_file in ["indicinstruct_hindi.jsonl", "evol_hindi.jsonl", "samanantar_hindi.jsonl"]:
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
    # Deduplicate by first user message
    seen = set()
    all_conv = []
    for row in conv_rows:
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


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Download and prepare Hindi datasets")
    parser.add_argument("--only", type=str, nargs="*",
                        choices=["indicorp", "fineweb", "indicinstruct", "evol",
                                 "samanantar", "wikipedia", "codealpaca"],
                        help="Only download specific sources")
    parser.add_argument("--tokenizer-only", action="store_true",
                        help="Only download tokenizer sources (indicorp, fineweb)")
    parser.add_argument("--build-only", action="store_true",
                        help="Skip downloads, just build expert files from existing raw data")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Override max rows per source")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Hindi Dataset Download & Preparation")
    log.info("=" * 60)

    if not args.build_only:
        sources = args.only or ["indicorp", "fineweb", "indicinstruct", "evol",
                                "samanantar", "wikipedia", "codealpaca"]

        if args.tokenizer_only:
            sources = ["indicorp", "fineweb"]

        download_funcs = {
            "indicorp": lambda: download_indicorp(args.max_rows or 500_000),
            "fineweb": lambda: download_fineweb_hindi(args.max_rows or 500_000),
            "indicinstruct": lambda: download_indicinstruct(args.max_rows or 100_000),
            "evol": lambda: download_evol_hindi(args.max_rows or 100_000),
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

    for f in sorted(RAW_DIR.glob("*.jsonl")):
        count = _line_count(f)
        if count == 0:
            empty_warnings.append(f"raw/{f.name}")

    if empty_warnings:
        log.warning("")
        log.warning("WARNING: The following files have 0 rows — something went wrong:")
        for w in empty_warnings:
            log.warning(f"  - {w}")
        log.warning("Check the logs above for 'first row keys' to debug field name mismatches.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
