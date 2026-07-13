"""Clean, deduplicate, and combine all downloaded open-source datasets.

Multi-stage cleaning pipeline:
  1. Parse all JSONL files under Download/
  2. Remove PII (emails, phones, credit cards)
  3. Quality filters (min length, bad markers, language)
  4. Near-deduplicate (fuzzy text matching)
  5. Balance by category & first-token diversity
  6. Write final clean train pack

Usage:
    python tools/clean_and_combine.py
    python tools/clean_and_combine.py --max-rows 500000
    python tools/clean_and_combine.py --output-name mega_clean --dedup-strength 0.85
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATA_DIR = Path("Download")
OUT_DIR = Path("Download/train_pack")

# ── PII patterns ──
PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[CC]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
]

# ── Quality bad markers ──
BAD_MARKERS = (
    "thought:", "action:", "action input:", "available apis:", "relevant apis:",
    "tool_name", "api_name", "<human>:", "<bot>:", "near me", "tonight",
    "current weather", "weather today", "unsupported claims", "connect the answer",
    "\u00e2", "\ufffd", "{{", "{%", "{{",
)

# ── Boilerplate / low-value patterns ──
BOILERPLATE = re.compile(
    r"(?i)(terms?\s+of\s+service|copyright\s+\d{4}|all\s+rights\s+reserved|"
    r"privacy\s+policy|cookie\s+policy|click\s+here|subscribe\s+to|"
    r"follow\s+us\s+on|share\s+this|leave\s+a\s+comment|"
    r"disclaimer|powered\s+by|theme\s+by)"
)

SYSTEM_REPLACEMENTS = [
    "You are Atulya. Answer clearly and accurately.",
    "You are Atulya. Be clear, direct, and helpful.",
    "You are Atulya. Give accurate, practical answers.",
]


def redact_pii(text: str) -> str:
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def first_token(text: str) -> str:
    m = re.search(r"[A-Za-z0-9]+", text.strip())
    return m.group(0) if m else "?"


def is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


# ── Stage 1: Parse & filter ──

def parse_row(line: str) -> dict[str, str] | None:
    try:
        raw = json.loads(line)
    except Exception:
        return None

    user = clean_space(raw.get("user") or raw.get("instruction") or raw.get("prompt") or raw.get("question") or "")
    assistant = clean_space(raw.get("assistant") or raw.get("response") or raw.get("output") or raw.get("answer") or raw.get("completion") or "")
    system = clean_space(raw.get("system") or "")
    text = clean_space(raw.get("text") or "")

    if not user and not assistant and text:
        if len(text) > 50:
            user = "Continue the following text:"
            assistant = text[:4000]
        else:
            return None

    if not user or not assistant:
        return None
    if len(user) < 3 or len(assistant) < 3:
        return None

    user = redact_pii(user)
    assistant = redact_pii(assistant)

    if not system:
        system = SYSTEM_REPLACEMENTS[0]

    category = clean_space(raw.get("category") or raw.get("lang") or "downloaded")
    return {"system": system, "user": user, "assistant": assistant, "category": category}


def pass_quality_filter(row: dict[str, str], min_words: int = 4) -> bool:
    joined = f"{row['user']} {row['assistant']}".lower()
    if len(joined) < 30:
        return False
    if any(m in joined for m in BAD_MARKERS):
        return False
    words = re.findall(r"[A-Za-z]+", row["assistant"])
    if len(words) < min_words:
        return False
    if row["assistant"].count("?") > 4:
        return False
    if len(row["assistant"]) > 8000:
        return False
    if BOILERPLATE.search(row["assistant"]):
        return False
    if not is_ascii(joined):
        return False
    return True


# ── Stage 2: Deduplication ──

def ngram_shingle(text: str, n: int = 5) -> set[str]:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def dedup_similarity(a: str, b: str) -> float:
    sha = ngram_shingle(a)
    shb = ngram_shingle(b)
    if not sha or not shb:
        return 0.0
    intersection = sha & shb
    return len(intersection) / max(len(sha), len(shb))


class DedupFilter:
    def __init__(self, threshold: float = 0.88):
        self.threshold = threshold
        self.seen_hashes: set[str] = set()
        self.fingerprints: list[set[str]] = []
        self.fp_max = 5000  # keep recent for near-dedup

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    @staticmethod
    def _fp(text: str) -> set[str]:
        """5-gram word fingerprint for similarity."""
        words = re.findall(r"[A-Za-z0-9]+", text.lower())
        return set(" ".join(words[i:i + 5]) for i in range(len(words) - 4))

    def is_dup(self, text: str) -> bool:
        text_clean = text.lower().strip()
        h = self._hash(text_clean)
        if h in self.seen_hashes:
            return True

        # Near-dedup: check against a small window of recent fingerprints
        if self.fingerprints:
            fp = self._fp(text_clean[:200])
            if fp:
                sample = self.fingerprints[:min(30, len(self.fingerprints))]
                for existing_fp in sample:
                    if not existing_fp or not fp:
                        continue
                    j = len(fp & existing_fp) / max(len(fp), len(existing_fp))
                    if j > self.threshold:
                        return True
                self.fingerprints.append(fp)
                if len(self.fingerprints) > self.fp_max:
                    self.fingerprints = self.fingerprints[-self.fp_max // 2:]

        self.seen_hashes.add(h)
        return False


# ── Stage 3: Classification ──

def classify(row: dict[str, str]) -> str:
    content = f"{row['user']} {row['assistant']}".lower()
    cat = row.get("category", "").lower()

    code_kw = ["python", "function", "def ", "import ", "class ", "return ", "print(", "code", "javascript", "java"]
    reason_kw = ["why", "how", "explain", "reason", "steps", "compare", "solve", "math", "equation"]
    factual_kw = ["what is", "who is", "when did", "where is", "define", "science", "history", "wikipedia"]
    sentiment_kw = ["sad", "angry", "nervous", "stuck", "overwhelmed", "feeling", "worried", "mistake"]
    chat_kw = ["hi", "hello", "who are you", "your name", "how are you", "thanks"]

    if any(x in content for x in code_kw):
        return "code"
    if any(x in content for x in reason_kw):
        return "reasoning"
    if any(x in cat for x in ["code", "math"]):
        return "code"
    if any(x in cat for x in ["reason"]):
        return "reasoning"
    if any(x in content for x in factual_kw):
        return "factual"
    if any(x in content for x in sentiment_kw):
        return "sentiment"
    if any(x in content for x in chat_kw):
        return "chat"
    if any(x in cat for x in ["general", "web", "fineweb"]):
        return "general"
    return "instruction"


# ── Stage 4: Balancing ──

def entropy(rows: list[dict]) -> float:
    counts = Counter(first_token(r["assistant"]) for r in rows)
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def balance(
    pools: dict[str, list[dict]],
    target: int,
    rng: random.Random,
) -> list[dict]:
    names = [n for n in pools if pools[n]]
    rng.shuffle(names)
    picked = []
    seen = set()
    tok_counts: Counter = Counter()
    max_tok_frac = 0.10

    def try_pick(row: dict) -> bool:
        key = (row["user"][:80], row["assistant"][:80])
        if key in seen:
            return False
        tok = first_token(row["assistant"])
        if tok_counts[tok] > target * max_tok_frac and rng.random() < 0.5:
            return False
        seen.add(key)
        tok_counts[tok] += 1
        picked.append(row)
        return True

    picked_cat: Counter = Counter()
    cursors = {name: 0 for name in names}
    quota = max(1, target // len(names))
    for name in names:
        pool = pools[name]
        rng.shuffle(pool)
        while cursors[name] < len(pool) and picked_cat[name] < quota:
            if len(picked) >= target:
                break
            row = pool[cursors[name]]
            cursors[name] += 1
            if try_pick(row):
                picked_cat[name] += 1

    # Fill any remaining target by rotating categories so large pools cannot
    # monopolize the tail after smaller categories hit their natural limit.
    while len(picked) < target:
        made_progress = False
        for name in names:
            pool = pools[name]
            while cursors[name] < len(pool):
                row = pool[cursors[name]]
                cursors[name] += 1
                if try_pick(row):
                    made_progress = True
                    break
            if len(picked) >= target:
                break
        if not made_progress:
            break

    rng.shuffle(picked)
    return picked[:target]


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Clean & combine open-source datasets")
    parser.add_argument("--max-rows", type=int, default=1_000_000, help="Target rows")
    parser.add_argument("--dedup-strength", type=float, default=0.88, help="Near-dedup threshold (0-1)")
    parser.add_argument("--output-name", default="mega_clean", help="Output name prefix")
    parser.add_argument("--source-file", action="append", default=[],
                        help="Specific JSONL source file to clean. Can be passed multiple times.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", action="store_true", help="Stats only, no write")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find JSONL files. By default avoid Download/train_pack so generated
    # outputs do not recursively become future inputs. Use --source-file to
    # intentionally rebalance a generated pack.
    if args.source_file:
        jsonl_files = [Path(p) for p in args.source_file]
    else:
        jsonl_files = sorted(Path("Download").rglob("*.jsonl"))
        jsonl_files = [p for p in jsonl_files if "train_pack" not in str(p) and "archived" not in str(p)]
    jsonl_files = [p for p in jsonl_files if p.exists()]
    if not jsonl_files:
        print("No datasets found. Run `python tools/download_data.py` first.")
        sys.exit(1)

    print("=" * 60)
    print("STAGE 1: PARSING & QUALITY FILTERING")
    print("=" * 60)

    all_rows = []
    parse_errors = 0
    quality_dropped = 0
    source_counts: Counter = Counter()

    for path in jsonl_files:
        src = str(path.relative_to(Path("Download")))
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                row = parse_row(line)
                if not row:
                    parse_errors += 1
                    continue
                if not pass_quality_filter(row):
                    quality_dropped += 1
                    continue
                all_rows.append(row)
                source_counts[path.parent.name] += 1

    total_raw = len(all_rows) + parse_errors + quality_dropped
    print(f"  Total raw lines: {total_raw:,}")
    print(f"  Parse errors:    {parse_errors:,}")
    print(f"  Quality drops:   {quality_dropped:,}")
    print(f"  Passed quality:  {len(all_rows):,}")
    print(f"  Sources:         {dict(source_counts.most_common(10))}")

    if not all_rows:
        print("No rows passed quality filter!")
        sys.exit(1)

    # ── Stage 2: PII redaction ──
    print("\n" + "=" * 60)
    print("STAGE 2: PII REDACTION & NORMALIZATION")
    print("=" * 60)
    pii_found = 0
    for row in all_rows:
        before_u = row["user"]
        before_a = row["assistant"]
        row["user"] = redact_pii(row["user"])
        row["assistant"] = redact_pii(row["assistant"])
        if row["user"] != before_u or row["assistant"] != before_a:
            pii_found += 1
        row["user"] = clean_space(row["user"])
        row["assistant"] = clean_space(row["assistant"])
    print(f"  Rows with PII redacted: {pii_found:,}")

    # ── Stage 3: Deduplication ──
    print("\n" + "=" * 60)
    print(f"STAGE 3: DEDUPLICATION (threshold={args.dedup_strength})")
    print("=" * 60)
    deduper = DedupFilter(threshold=args.dedup_strength)
    deduped = []
    exact_dup = 0
    near_dup = 0
    for row in all_rows:
        text = f"{row['user']} {row['assistant']}"
        if deduper.is_dup(text):
            near_dup += 1
            continue
        deduped.append(row)
    print(f"  After dedup:  {len(deduped):,} (removed {len(all_rows)-len(deduped):,})")

    # ── Stage 4: Classification ──
    print("\n" + "=" * 60)
    print("STAGE 4: CLASSIFICATION")
    print("=" * 60)
    pools: dict[str, list] = defaultdict(list)
    for row in deduped:
        cat = classify(row)
        row["category"] = cat
        pools[cat].append(row)
    for cat, rows in sorted(pools.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:15s}: {len(rows):,}  (entropy: {entropy(rows):.2f})")

    if args.report:
        return

    # ── Stage 5: Balance & write ──
    print("\n" + "=" * 60)
    print(f"STAGE 5: BALANCING (target: {args.max_rows:,})")
    print("=" * 60)
    target = min(args.max_rows, len(deduped))
    balanced = balance(pools, target, rng)
    print(f"  Final rows: {len(balanced):,}")
    print(f"  Entropy:    {entropy(balanced):.2f}")

    name = f"{args.output_name}_{len(balanced)//1000}k.jsonl"
    out_path = out_dir / name
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in balanced:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cat_counts = Counter(r["category"] for r in balanced)
    first_toks = Counter(first_token(r["assistant"]) for r in balanced)
    manifest = {
        "output": str(out_path),
        "rows": len(balanced),
        "target": args.max_rows,
        "dedup_threshold": args.dedup_strength,
        "pii_redacted": pii_found,
        "quality_pipeline": ["parse", "pii_redact", "quality_filter", "dedup", "classify", "balance"],
        "categories": dict(cat_counts.most_common()),
        "entropy": round(entropy(balanced), 2),
        "top_first_tokens": dict(first_toks.most_common(20)),
    }
    manifest_path = out_dir / f"{args.output_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n  Written: {out_path}")
    print(f"  Size:    {out_path.stat().st_size/1_000_000:.1f} MB")
    print(f"  Manifest: {manifest_path}")
    print(f"\n  Categories:")
    for c, n in cat_counts.most_common():
        print(f"    {c:15s}: {n:,}")
    print(f"\n  Top first tokens (diversity check):")
    for t, n in first_toks.most_common(10):
        print(f"    '{t}': {n:,} ({n/len(balanced)*100:.1f}%)")


if __name__ == "__main__":
    main()
