"""Build a balanced 100k seed/chat QA dataset.

The output is intended for early prompt-masked training, not broad raw LM
continuation. It prefers short, clean records and avoids tool traces.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("Download/qa")
DEFAULT_OUTPUT = Path("Download/qa/seed_chat_balanced_100k.jsonl")
SYSTEM = "You are Atulya. Answer clearly and briefly."
RANDOM_SEED = 42

TARGETS = {
    "balanced_seed_chat": 20_000,
    "short_factual": 30_000,
    "reasoning_simple": 20_000,
    "code_basics": 10_000,
    "conversation_help": 10_000,
    "style_paraphrases": 10_000,
}

TOOLISH = (
    "thought:",
    "action:",
    "action input:",
    "available apis:",
    "relevant apis:",
    "api_name",
    "tool_name",
    "<human>:",
    "<bot>:",
)

GENERIC_USERS = {
    "what is it?",
    "what is it",
    "what is it,?",
    "what is it, mom/dad?",
}

CODE_HINTS = (
    "python",
    "function",
    "code",
    "list",
    "string",
    "loop",
    "dictionary",
    "array",
    "sort",
    "merge",
    "prime",
    "lambda",
    "def ",
)

REASONING_HINTS = (
    "why",
    "how",
    "explain",
    "difference",
    "compare",
    "solve",
    "steps",
    "reason",
    "deal with",
    "improve",
)

CONVERSATION_HINTS = (
    "hi",
    "hello",
    "thanks",
    "thank you",
    "how are you",
    "who are you",
    "your name",
    "can you",
    "help",
    "i feel",
    "i am",
    "advice",
)


def load_rows(input_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fp in sorted(input_dir.glob("*.jsonl")):
        if fp.name == DEFAULT_OUTPUT.name:
            continue
        with fp.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                user = (d.get("user") or d.get("instruction") or d.get("prompt") or "").strip()
                assistant = (d.get("assistant") or d.get("response") or d.get("output") or "").strip()
                system = (d.get("system") or "").strip()
                if user and assistant:
                    rows.append({"system": system, "user": user, "assistant": assistant})
    return rows


def is_clean(row: dict[str, str]) -> bool:
    user = row["user"].strip()
    assistant = row["assistant"].strip()
    joined = f"{user} {assistant}".lower()
    if any(marker in joined for marker in TOOLISH):
        return False
    if user.lower() in GENERIC_USERS:
        return False
    if len(user) < 3 or len(assistant) < 2:
        return False
    if len(user) > 260 or len(assistant) > 520:
        return False
    if assistant.count("\n") > 6:
        return False
    if len(re.findall(r"[A-Za-z]", assistant)) < 2:
        return False
    return True


def first_token(text: str) -> str:
    match = re.search(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]", text.strip())
    return match.group(0) if match else ""


def lower_first(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].lower() + text[1:]


def clean_row(row: dict[str, str], category: str) -> dict[str, str]:
    return {
        "system": row.get("system") or SYSTEM,
        "user": row["user"].strip(),
        "assistant": row["assistant"].strip(),
        "category": category,
    }


def classify(row: dict[str, str]) -> set[str]:
    user = row["user"].lower()
    assistant = row["assistant"]
    joined = f"{user} {assistant.lower()}"
    cats: set[str] = set()
    if any(h in joined for h in CODE_HINTS) or "return " in assistant or "def " in assistant:
        cats.add("code_basics")
    if any(h in user for h in REASONING_HINTS) or len(assistant) > 120:
        cats.add("reasoning_simple")
    if any(h in user for h in CONVERSATION_HINTS):
        cats.add("conversation_help")
    if re.match(r"^(what|who|where|when|which|define)\b", user) and len(assistant) <= 220:
        cats.add("short_factual")
    cats.add("balanced_seed_chat")
    return cats


def rewrite_answer(answer: str, mode: int) -> str:
    a = answer.strip()
    if not a:
        return a
    bare = a
    if mode == 0:
        return a
    if mode == 1:
        return "In short, " + lower_first(bare)
    if mode == 2:
        return "Simply put, " + lower_first(bare)
    if mode == 3:
        return "It means " + lower_first(bare).rstrip(".") + "."
    if mode == 4 and bare.startswith("A "):
        return "It is a " + lower_first(bare[2:])
    if mode == 5 and bare.startswith("An "):
        return "It is an " + lower_first(bare[3:])
    if mode == 6 and bare.startswith("The "):
        return "This is the " + lower_first(bare[4:])
    if mode == 7:
        return "You can think of it as " + lower_first(bare).rstrip(".") + "."
    if mode == 8:
        return "Briefly, " + lower_first(bare)
    if mode == 9:
        return "Practically, " + lower_first(bare)
    if mode == 10:
        return "For most cases, " + lower_first(bare)
    if mode == 11:
        return "One clear answer is: " + bare
    if mode == 12:
        return "A direct answer is: " + lower_first(bare)
    return a


def make_variants(row: dict[str, str], category: str) -> list[dict[str, str]]:
    variants = [clean_row(row, category)]
    for mode in range(1, 13):
        rewritten = rewrite_answer(row["assistant"], mode)
        if rewritten and rewritten != row["assistant"] and len(rewritten) <= 560:
            variants.append({
                "system": row.get("system") or SYSTEM,
                "user": row["user"].strip(),
                "assistant": rewritten,
                "category": category,
            })
    return variants


USER_PREFIXES = (
    "",
    "Answer briefly: ",
    "Give a short answer: ",
    "In simple terms, ",
    "Can you answer this? ",
    "Quick question: ",
)


def synthesize_fill(
    base_rows: list[dict[str, str]],
    category: str,
    target: int,
    seen: set[tuple[str, str]],
    rng: random.Random,
) -> list[dict[str, str]]:
    """Create extra variants when clean source rows do not fill a target."""
    if not base_rows:
        return []
    out: list[dict[str, str]] = []
    shuffled = list(base_rows)
    rng.shuffle(shuffled)
    attempts = 0
    while len(out) < target and attempts < target * 50:
        base = shuffled[attempts % len(shuffled)]
        prefix = USER_PREFIXES[(attempts // len(shuffled)) % len(USER_PREFIXES)]
        mode = 1 + (attempts % 12)
        user = base["user"].strip()
        if prefix and not user.lower().startswith(prefix.lower()):
            user = prefix + user[0].lower() + user[1:]
        assistant = rewrite_answer(base["assistant"], mode)
        row = {
            "system": base.get("system") or SYSTEM,
            "user": user,
            "assistant": assistant,
            "category": category,
        }
        key = dedupe_key(row)
        if key not in seen and is_clean(row):
            seen.add(key)
            out.append(row)
        attempts += 1
    return out


def synthetic_code_rows() -> list[dict[str, str]]:
    tasks = [
        ("Write a Python function to add two numbers.", "def add(a, b):\n    return a + b"),
        ("Write a Python function to multiply two numbers.", "def multiply(a, b):\n    return a * b"),
        ("Check if a number is even in Python.", "Use n % 2 == 0 to check whether n is even."),
        ("Filter even numbers from a list.", "evens = [x for x in nums if x % 2 == 0]"),
        ("Reverse a string in Python.", "Use s[::-1] to reverse a string."),
        ("Sort a list in Python.", "Use sorted(items) for a new sorted list."),
        ("Find the largest number in a list.", "Use max(nums) to find the largest number."),
        ("Count words in a string.", "Use len(text.split()) to count whitespace-separated words."),
        ("Create a dictionary in Python.", "Use braces, for example: data = {'name': 'Atulya'}."),
        ("Write a simple for loop in Python.", "for item in items:\n    print(item)"),
    ]
    rows = []
    for user, assistant in tasks:
        rows.append({"system": SYSTEM, "user": user, "assistant": assistant})
    return rows


def dedupe_key(row: dict[str, str]) -> tuple[str, str]:
    return (re.sub(r"\s+", " ", row["user"].lower()).strip(), re.sub(r"\s+", " ", row["assistant"].lower()).strip())


def pick_balanced(candidates: list[dict[str, str]], target: int, rng: random.Random) -> list[dict[str, str]]:
    if not candidates:
        return []
    by_tok: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen = set()
    for row in candidates:
        key = dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        by_tok[first_token(row["assistant"])].append(row)
    for bucket in by_tok.values():
        rng.shuffle(bucket)

    picked: list[dict[str, str]] = []
    token_order = list(by_tok)
    rng.shuffle(token_order)
    cursor = Counter()

    while len(picked) < target and token_order:
        made_progress = False
        for tok in list(token_order):
            bucket = by_tok[tok]
            idx = cursor[tok]
            if idx >= len(bucket):
                token_order.remove(tok)
                continue
            picked.append(bucket[idx])
            cursor[tok] += 1
            made_progress = True
            if len(picked) >= target:
                break
        if not made_progress:
            break
    return picked


def entropy(rows: list[dict[str, str]]) -> float:
    counts = Counter(first_token(r["assistant"]) for r in rows)
    total = sum(counts.values())
    if not total:
        return 0.0
    import math
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [row for row in load_rows(args.input) if is_clean(row)]
    rows.extend(synthetic_code_rows())

    pools: dict[str, list[dict[str, str]]] = {name: [] for name in TARGETS}
    for row in rows:
        cats = classify(row)
        for cat in cats:
            if cat in pools:
                pools[cat].extend(make_variants(row, cat))
        pools["style_paraphrases"].extend(make_variants(row, "style_paraphrases")[1:])

    output_rows: list[dict[str, str]] = []
    global_seen = set()
    for cat, target in TARGETS.items():
        picked = pick_balanced(pools[cat], target, rng)
        if len(picked) < target:
            raise RuntimeError(f"{cat}: only produced {len(picked)} of {target} rows")
        kept = []
        for row in picked:
            key = dedupe_key(row)
            if key in global_seen:
                continue
            global_seen.add(key)
            kept.append(row)
        if len(kept) < target:
            needed = target - len(kept)
            if cat == "style_paraphrases":
                fill_source = [clean_row(r, cat) for r in rows]
            else:
                fill_source = [clean_row(r, cat) for r in rows if cat in classify(r)]
            kept.extend(synthesize_fill(fill_source, cat, needed, global_seen, rng))
        output_rows.extend(kept[:target])
        print(f"{cat}: {len(kept[:target])} rows, entropy={entropy(kept[:target]):.2f}")

    rng.shuffle(output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(row["category"] for row in output_rows)
    firsts = Counter(first_token(row["assistant"]) for row in output_rows)
    print(f"wrote {args.output}")
    print(f"rows={len(output_rows)} entropy={entropy(output_rows):.2f}")
    print("categories", dict(counts))
    print("first_tokens", firsts.most_common(25))


if __name__ == "__main__":
    main()
