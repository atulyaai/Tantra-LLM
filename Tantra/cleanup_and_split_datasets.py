"""
Tantra/cleanup_and_split_datasets.py — Deduplicate, quality-check, and split JSONL datasets.

Scans every *.jsonl under Datasets/, globally deduplicates by content hash
(keeping the first-seen occurrence), reports per-file quality statistics,
optionally splits large files into smaller shards, and moves redundant files
into a `.duplicates/` subfolder for review instead of hard-deleting them.

Usage:
    python Tantra/cleanup_and_split_datasets.py                  # dry-run report only
    python Tantra/cleanup_and_split_datasets.py --apply         # actually move/split
    python Tantra/cleanup_and_split_datasets.py --shard-lines 50000
    python Tantra/cleanup_and_split_datasets.py --min-split 500000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Datasets")
DUPE_DIR = "_duplicates"  # inside Datasets, but dirs starting with "_" are skipped as topics by main.py

# Fields used for quality heuristics. Any of these -> structured chat record.
CHAT_FIELDS = ("messages", "system", "user", "assistant", "prompt", "response", "instruction", "input", "output")

MIN_CONTENT_LEN = 2  # tokens-ish; a "useful" item must have at least this many chars of text


def md5(line: str) -> str:
    return hashlib.md5(line.encode("utf-8", errors="ignore")).hexdigest()


def sniff_record(line: str) -> tuple[str, bool, int, int, bool]:
    """Return (kind, has_chat, text_len, assistant_len, has_placeholder).

    kind is one of: 'chat', 'stack', 'raw', 'unrecognized', 'invalid'.
    """
    try:
        item = json.loads(line)
    except Exception:
        return "invalid", False, len(line), 0, False
    if not isinstance(item, dict):
        return "invalid", False, len(line), 0, False

    # ChatML-style
    msgs = item.get("messages")
    if isinstance(msgs, list) and msgs:
        text_len = 0
        has_chat = False
        asst_len = 0
        for m in msgs:
            if isinstance(m, dict):
                content = m.get("content", "")
                if isinstance(content, str):
                    text_len += len(content)
                if m.get("role") == "assistant":
                    has_chat = True
                    asst_len += len(content) if isinstance(content, str) else 0
        if has_chat:
            return "chat", True, text_len, asst_len, _has_placeholder(text_len, asst_len)

    # Flat schema
    has_chat = any(item.get(k) for k in ("assistant", "response", "output"))
    text_len = sum(len(str(item.get(k, ""))) for k in ("system", "user", "assistant", "prompt", "response", "instruction", "input", "output") if k in item)
    asst_len = len(str(item.get("assistant", item.get("response", item.get("output", "")))))
    if has_chat:
        return "chat", True, text_len, asst_len, _has_placeholder(text_len, asst_len)

    if any(k in item for k in CHAT_FIELDS):
        return "unrecognized", False, text_len, 0, False

    return "raw", False, len(line), 0, False


def _has_placeholder(text_len: int, asst_len: int) -> bool:
    if text_len > 0 and asst_len / text_len < 0.05:
        return True
    return False


def scan_files() -> list[str]:
    jsonls = []
    for root, _dirs, files in os.walk(DATASETS_DIR):
        for fn in sorted(files):
            if fn.endswith(".jsonl"):
                jsonls.append(os.path.join(root, fn))
    return sorted(jsonls)


def quality_report(files: list[str]) -> None:
    print("\n=== QUALITY & OVERLAP REPORT ===")
    total_lines = 0
    for p in files:
        stats = {"chat": 0, "stack": 0, "raw": 0, "unrecognized": 0, "invalid": 0, "short": 0, "placeholder": 0}
        lines = 0
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lines += 1
                    kind, has_chat, text_len, asst_len, has_placeholder = sniff_record(line)
                    stats[kind] += 1
                    if text_len < MIN_CONTENT_LEN and has_chat:
                        stats["short"] += 1
                    if has_placeholder:
                        stats["placeholder"] += 1
        except Exception as e:
            print(f"  {os.path.relpath(p, DATASETS_DIR)}: ERROR {e}")
            continue
        total_lines += lines
        print(f"  {os.path.relpath(p, DATASETS_DIR)}")
        print(f"      lines={lines:,} chat={stats['chat']:,} raw={stats['raw']:,} "
              f"unrec={stats['unrecognized']:,} invalid={stats['invalid']:,} "
              f"short={stats['short']:,} placeholder={stats['placeholder']:,}")
    print(f"  TOTAL lines: {total_lines:,}\n")


def dedupe_and_split(files: list[str], apply: bool, shard_lines: int, min_split: int) -> None:
    print("=== GLOBAL DEDUPLICATION PASS ===")
    seen: dict[str, tuple[str, int]] = {}  # hash -> (path, line_no_original)
    dup_moves: list[tuple[str, str]] = []  # (src, dst) file-level moves
    keep_lines: dict[str, list[str]] = {}  # path -> deduped lines (kept order)
    kept_per_file: dict[str, int] = {}

    for p in files:
        with open(p, encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                h = md5(line)
                if h in seen:
                    # duplicate: record which file it lives in
                    orig = seen[h]
                    if orig[0] != p:
                        dup_moves.append((p, orig[0]))
                    continue
                seen[h] = (p, len(keep_lines.get(p, [])) + 1)
                keep_lines.setdefault(p, []).append(raw_line)

    for p in files:
        kept_per_file[p] = len(keep_lines.get(p, []))

    # File-level duplicates: if EVERY line of a file appeared earlier in another
    # file, the whole file is redundant.
    file_dupes: list[str] = []
    for p in files:
        if os.path.getsize(p) == 0:
            file_dupes.append(p)
            continue
        if kept_per_file[p] == 0:
            file_dupes.append(p)

    # Cross-file redundancy estimate: for each pair, how much of the smaller is dup.
    print("\nPer-file dedup results:")
    for p in sorted(kept_per_file):
        orig_lines = sum(1 for _ in open(p, encoding="utf-8", errors="ignore"))
        print(f"  {os.path.relpath(p, DATASETS_DIR)}: kept {kept_per_file[p]:,} / {orig_lines:,} lines")

    if not apply:
        print("\n[DRY-RUN] Nothing was moved or split. Re-run with --apply to execute.")
        return

    # 1. Move redundant files
    print("\n=== FILE-LEVEL CLEANUP ===")
    for p in file_dupes:
        rel = os.path.relpath(p, DATASETS_DIR)
        dst = os.path.join(DATASETS_DIR, DUPE_DIR, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"  [duplicate] {rel} -> {DUPE_DIR}/")
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(p, dst)
        _move_sibling_bin(p)

    # 2. Write deduped content back, splitting large files into shards
    print("\n=== SPLITTING / REWRITING ===")
    for p, lines in keep_lines.items():
        if not lines:
            continue
        if os.path.getsize(p) == 0:
            continue
        # Skip files already in a shard/clean layout (only touch if not too small)
        total_tokens = sum(len(l) for l in lines)
        if total_tokens < min_split:
            _rewrite(p, lines)
            continue
        base, ext = os.path.splitext(p)
        shard_idx = 0
        batch: list[str] = []
        n_shards = 0
        for line in lines:
            batch.append(line)
            if len(batch) >= shard_lines:
                shard_path = f"{base}.shard{shard_idx:04d}{ext}"
                _rewrite(shard_path, batch)
                shard_idx += 1
                n_shards += 1
                if n_shards == 1:
                    # Original file will be replaced by shards; if it's not the first shard path, remove it
                    pass
                batch = []
        if batch:
            shard_path = f"{base}.shard{shard_idx:04d}{ext}"
            _rewrite(shard_path, batch)
            n_shards += 1
        print(f"  {os.path.relpath(p, DATASETS_DIR)} -> {n_shards} shard(s)")

    # 3. Remove original oversized files that were sharded (replaced by shard* names)
    print("\n=== REMOVING PRE-SHARD ORIGINALS ===")
    for p in files:
        if p in keep_lines and keep_lines[p]:
            total_tokens = sum(len(l) for l in keep_lines[p])
            if total_tokens >= min_split and os.path.exists(p):
                rel = os.path.relpath(p, DATASETS_DIR)
                # keep original only if it wasn't rewritten into itself (it has shards now)
                if not p.endswith(".shard0000.jsonl"):
                    print(f"  archived {rel} -> {DUPE_DIR}/")
                    dst = os.path.join(DATASETS_DIR, DUPE_DIR, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if os.path.exists(dst):
                        os.remove(dst)
                    shutil.move(p, dst)
                    _move_sibling_bin(p)


def _move_sibling_bin(jsonl_path: str) -> None:
    """Move the stale pre-tokenized .bin cache next to a jsonl that is being
    archived (its token stream no longer matches the remaining jsonl file)."""
    bin_path = os.path.splitext(jsonl_path)[0] + ".bin"
    if os.path.exists(bin_path):
        rel = os.path.relpath(bin_path, DATASETS_DIR)
        dst = os.path.join(DATASETS_DIR, DUPE_DIR, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(bin_path, dst)
        print(f"  [stale-bin] {rel} -> {DUPE_DIR}/")


def _rewrite(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", errors="ignore") as f:
        f.writelines(lines)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually move duplicates & split; default is dry-run report")
    ap.add_argument("--shard-lines", type=int, default=50000, help="Max lines per shard")
    ap.add_argument("--min-split", type=int, default=1000000, help="Only split files with at least this many total chars")
    args = ap.parse_args()

    files = scan_files()
    if not files:
        print("No JSONL files found.")
        return
    quality_report(files)
    dedupe_and_split(files, args.apply, args.shard_lines, args.min_split)


if __name__ == "__main__":
    main()
