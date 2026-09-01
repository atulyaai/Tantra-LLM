"""Build a small, deterministic, local-only SFT pilot dataset from JSONL."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from pathlib import Path


def is_usable(item: object, benchmark_prompts: set[str]) -> bool:
    if not isinstance(item, dict):
        return False
    user, assistant = item.get("user"), item.get("assistant")
    if not isinstance(user, str) or not isinstance(assistant, str):
        return False
    user, assistant = user.strip(), assistant.strip()
    if not (4 <= len(user) <= 700 and 40 <= len(assistant) <= 1800):
        return False
    if user.lower() in benchmark_prompts or "<|" in user or "<|" in assistant:
        return False
    words = re.findall(r"\b\w+\b", assistant.lower())
    if len(words) < 8:
        return False
    # Reject obvious repetition/garbled samples while leaving maths and code intact.
    if len(set(words)) / len(words) < 0.22 or re.search(r"(.{12,}?)\1{3,}", assistant):
        return False
    return True


def add_candidate(heap: list[tuple[int, str]], limit: int, score: int, line: str) -> None:
    entry = (-score, line)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def build(source: Path, benchmark: Path, train_out: Path, val_out: Path, train_size: int, val_size: int) -> None:
    prompts = {
        json.loads(line).get("prompt", "").strip().lower()
        for line in benchmark.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    train_heap: list[tuple[int, str]] = []
    val_heap: list[tuple[int, str]] = []
    seen_users: set[str] = set()
    scanned = accepted = 0
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            scanned += 1
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not is_usable(item, prompts):
                continue
            user_key = item["user"].strip().lower()
            if user_key in seen_users:
                continue
            seen_users.add(user_key)
            accepted += 1
            line = json.dumps(
                {key: item[key].strip() for key in ("system", "user", "assistant") if isinstance(item.get(key), str) and item[key].strip()},
                ensure_ascii=False,
            )
            digest = int.from_bytes(hashlib.sha256(line.encode("utf-8")).digest()[:8], "big")
            # Independent deterministic buckets prevent train/validation overlap.
            if digest % 10 == 0:
                add_candidate(val_heap, val_size, digest, line)
            else:
                add_candidate(train_heap, train_size, digest, line)

    if len(train_heap) < train_size or len(val_heap) < val_size:
        raise RuntimeError(f"Not enough clean examples: train={len(train_heap)}/{train_size}, val={len(val_heap)}/{val_size}")
    for path, heap in ((train_out, train_heap), (val_out, val_heap)):
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [line for _, line in sorted(heap, reverse=True)]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Scanned {scanned:,}; accepted {accepted:,}; wrote {train_size:,} train and {val_size:,} validation examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Datasets/tantra_train.jsonl")
    parser.add_argument("--benchmark", default="Datasets/eval_60_benchmark.jsonl")
    parser.add_argument("--train-out", default="Datasets/clean_pilot_train.jsonl")
    parser.add_argument("--val-out", default="Datasets/clean_pilot_val.jsonl")
    parser.add_argument("--train-size", type=int, default=1800)
    parser.add_argument("--val-size", type=int, default=200)
    args = parser.parse_args()
    build(Path(args.source), Path(args.benchmark), Path(args.train_out), Path(args.val_out), args.train_size, args.val_size)
