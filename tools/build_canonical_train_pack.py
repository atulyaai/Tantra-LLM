"""Build one canonical deduped training pack from Download/train_pack.

This is for the current local workflow where the old source folders have been
cleaned and the useful material lives in train_pack JSONL files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUTS = [
    Path("Download/train_pack/quality_pack_81k.jsonl"),
    Path("Download/train_pack/train_pack_all_expanded_1040k.jsonl"),
    Path("Download/train_pack/train_pack_core_20k.jsonl"),
]

BAD_STARTS = (
    "the key idea",
    "a practical answer",
    "a practical way",
    "clear answer",
    "in short",
    "simply put",
    "practically,",
)

BAD_MARKERS = (
    "\ufffd",
    "\u00e2",
    "{{",
    "{%",
    "available apis:",
    "action input:",
    "tool_name",
    "api_name",
    "near me",
    "weather today",
)


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


def parse_row(line: str) -> dict[str, str] | None:
    try:
        raw = json.loads(line)
    except Exception:
        return None

    user = clean_space(raw.get("user") or raw.get("instruction") or raw.get("prompt") or raw.get("question") or "")
    assistant = clean_space(raw.get("assistant") or raw.get("response") or raw.get("output") or raw.get("answer") or "")
    system = clean_space(raw.get("system") or "")
    category = clean_space(raw.get("category") or raw.get("source") or "general")

    if not user or not assistant:
        return None
    if not system:
        system = "You are Atulya. Be clear, direct, and helpful."
    return {"system": system, "user": user, "assistant": assistant, "category": category}


def classify(row: dict[str, str]) -> str:
    text = f"{row['user']} {row['assistant']} {row.get('category', '')}".lower()
    if any(x in text for x in ("python", "function", "def ", "return ", "class ", "javascript", "code")):
        return "code"
    if any(x in text for x in ("why", "how", "explain", "reason", "steps", "compare", "solve")):
        return "reasoning"
    if any(x in text for x in ("sad", "angry", "nervous", "worried", "stuck", "feeling", "overwhelmed")):
        return "sentiment"
    if any(x in text for x in ("hello", "hi", "how are you", "who are you", "thanks")):
        return "chat"
    if any(x in text for x in ("what is", "who was", "who is", "define", "gravity", "machine learning", "science")):
        return "factual"
    return "instruction"


def repeated_word_run(text: str, max_run: int = 5) -> bool:
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return True
    run = 1
    prev = words[0]
    for word in words[1:]:
        if word == prev:
            run += 1
            if run >= max_run:
                return True
        else:
            prev = word
            run = 1
    return False


def quality_ok(row: dict[str, str]) -> bool:
    joined = f"{row['user']} {row['assistant']}".lower()
    answer = row["assistant"].strip()
    answer_l = answer.lower()

    if len(row["user"]) < 3 or len(answer) < 12:
        return False
    if len(answer) > 2500:
        return False
    if any(answer_l.startswith(prefix) for prefix in BAD_STARTS):
        return False
    if any(marker in joined for marker in BAD_MARKERS):
        return False
    if answer.count("?") > 4:
        return False
    if repeated_word_run(answer):
        return False
    if not is_ascii(joined):
        return False
    words = re.findall(r"[A-Za-z]+", answer)
    if len(words) < 4:
        return False
    return True


def row_hash(row: dict[str, str]) -> str:
    text = f"{row['user'].lower()} \n {row['assistant'].lower()}"
    return hashlib.sha1(clean_space(text).encode("utf-8")).hexdigest()


def entropy(rows: list[dict[str, str]]) -> float:
    counts = Counter(first_token(r["assistant"]) for r in rows)
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=120_000)
    parser.add_argument("--output", type=Path, default=Path("Download/train_pack/train_pack_canonical_120k.jsonl"))
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--first-token-frac", type=float, default=0.045)
    parser.add_argument("--inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen: set[str] = set()
    pools: dict[str, list[dict[str, str]]] = defaultdict(list)
    stats = Counter()

    for path in args.inputs:
        if not path.exists():
            continue
        print(f"Reading {path}...")
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stats["lines"] += 1
                row = parse_row(line)
                if not row:
                    stats["parse_drop"] += 1
                    continue
                if not quality_ok(row):
                    stats["quality_drop"] += 1
                    continue
                h = row_hash(row)
                if h in seen:
                    stats["exact_dup"] += 1
                    continue
                seen.add(h)
                row["category"] = classify(row)
                pools[row["category"]].append(row)
                stats["kept_pool"] += 1

    if not pools:
        raise SystemExit("No rows survived filtering.")

    for rows in pools.values():
        rng.shuffle(rows)

    target = min(args.target, sum(len(v) for v in pools.values()))
    categories = [cat for cat, rows in pools.items() if rows]
    # Slightly favor underrepresented chat/factual/sentiment so the model does
    # not become only instruction/code shaped.
    weights = {
        "chat": 1.3,
        "factual": 1.3,
        "sentiment": 1.2,
        "code": 1.0,
        "reasoning": 1.0,
        "instruction": 0.9,
    }
    weight_sum = sum(weights.get(cat, 1.0) for cat in categories)
    quotas = {cat: int(target * weights.get(cat, 1.0) / weight_sum) for cat in categories}

    selected: list[dict[str, str]] = []
    selected_hashes: set[str] = set()
    first_counts: Counter = Counter()
    first_cap = max(200, int(target * args.first_token_frac))

    def try_take(row: dict[str, str]) -> bool:
        h = row_hash(row)
        if h in selected_hashes:
            return False
        tok = first_token(row["assistant"])
        if first_counts[tok] >= first_cap:
            return False
        selected_hashes.add(h)
        first_counts[tok] += 1
        selected.append(row)
        return True

    for cat in categories:
        for row in pools[cat]:
            if sum(1 for r in selected if r["category"] == cat) >= quotas[cat]:
                break
            try_take(row)

    # Fill any remaining slots from all pools while respecting first-token caps.
    leftovers = [row for rows in pools.values() for row in rows]
    rng.shuffle(leftovers)
    for row in leftovers:
        if len(selected) >= target:
            break
        try_take(row)

    rng.shuffle(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    cats = Counter(r["category"] for r in selected)
    firsts = Counter(first_token(r["assistant"]) for r in selected)
    manifest = {
        "output": str(args.output),
        "rows": len(selected),
        "target": args.target,
        "inputs": [str(p) for p in args.inputs if p.exists()],
        "stats": dict(stats),
        "categories": dict(cats.most_common()),
        "entropy": round(entropy(selected), 2),
        "first_token_cap": first_cap,
        "top_first_tokens": dict(firsts.most_common(30)),
        "removed_bad_starts": BAD_STARTS,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {args.output} ({args.output.stat().st_size / 1_000_000:.1f} MB)")
    print(f"Rows: {len(selected):,}  Entropy: {entropy(selected):.2f}")
    print("Categories:", cats.most_common())
    print("Top first tokens:", firsts.most_common(15))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
