"""
Expert datasets
===============
Each expert has its own jsonl file:  <data_dir>/<domain>/<expert>.jsonl

Accepted row formats:
  {"messages": [{"role": "user", "content": ...}, {"role": "assistant", ...}]}
  {"text": "<|user|>\\n...\\n<|assistant|> ..."}            (legacy)
  {"text": "plain text"}                                    (pretraining)

Junk rows (template filler like "Sample 12 of 10000", "[SYNTHETIC ...]")
and exact duplicates are dropped at load time.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import torch

from Tantra.tokenizer import TantraTokenizer

JUNK = re.compile(r"Sample \d+ of \d+|\[SYNTHETIC|GENERATION NEEDED", re.I)
LEGACY = re.compile(r"<\|(user|assistant|system)\|>")


def _legacy_to_messages(text: str) -> Optional[List[Dict]]:
    parts = LEGACY.split(text)
    if len(parts) < 3:
        return None
    msgs = []
    for role, content in zip(parts[1::2], parts[2::2]):
        content = content.strip()
        if content:
            msgs.append({"role": role, "content": content})
    return msgs or None


def read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict]:
    """Load rows as {"messages": [...]} or {"text": ...}; filters junk + dups."""
    rows, seen = [], set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "messages" in item:
                row = {"messages": item["messages"]}
                if "experts" in item:  # mixed-intent row (used by eval / router)
                    row["experts"] = item["experts"]
            elif "text" in item:
                msgs = _legacy_to_messages(item["text"])
                row = {"messages": msgs} if msgs else {"text": item["text"]}
            else:
                continue
            key = json.dumps(row, ensure_ascii=False, sort_keys=True)
            if JUNK.search(key) or key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def find_expert_files(data_dirs: Sequence[str | Path]) -> Dict[str, Path]:
    """expert name -> jsonl path. Earlier dirs win (put curated data first)."""
    found: Dict[str, Path] = {}
    for d in data_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for f in sorted(d.glob("*/*.jsonl")):
            found.setdefault(f.stem, f)
    return found


def row_texts(row: Dict) -> Iterator[str]:
    if "messages" in row:
        for m in row["messages"]:
            yield m["content"]
    else:
        yield row["text"]


def encode_row(tok: TantraTokenizer, row: Dict, max_len: int, assistant_only: bool = True):
    if "messages" in row:
        ids, mask = tok.encode_chat(row["messages"])
        if not assistant_only:
            mask = [1] * len(ids)
    else:
        ids = [tok.bos_id] + tok.encode(row["text"]) + [tok.end_id]
        mask = [1] * len(ids)
    ids, mask = ids[:max_len], mask[:max_len]
    labels = [i if m else -100 for i, m in zip(ids, mask)]
    return ids, labels


class EncodedDataset:
    """Pre-tokenized samples (small data fits in RAM easily)."""

    def __init__(self, tok: TantraTokenizer, rows: List[Dict], max_len: int = 256,
                 assistant_only: bool = True, expert_id: int = -1):
        self.pad_id = tok.pad_id
        self.items = [encode_row(tok, r, max_len, assistant_only) for r in rows]
        self.items = [it for it in self.items if any(l != -100 for l in it[1][1:])]
        self.expert_id = expert_id

    def __len__(self) -> int:
        return len(self.items)

    def batches(self, batch_size: int, shuffle: bool = True, rng: Optional[random.Random] = None):
        idx = list(range(len(self.items)))
        if shuffle:
            (rng or random).shuffle(idx)
        for s in range(0, len(idx), batch_size):
            chunk = [self.items[i] for i in idx[s:s + batch_size]]
            yield collate(chunk, self.pad_id)


def collate(chunk, pad_id: int):
    L = max(len(ids) for ids, _ in chunk)
    ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
    labels = torch.full((len(chunk), L), -100, dtype=torch.long)
    mask = torch.zeros((len(chunk), L), dtype=torch.long)
    for i, (a, b) in enumerate(chunk):
        ids[i, :len(a)] = torch.tensor(a)
        labels[i, :len(b)] = torch.tensor(b)
        mask[i, :len(a)] = 1
    return ids, labels, mask


def first_user_text(row: Dict) -> str:
    """The text the router sees: the last user message (what the request is about)."""
    if "messages" in row:
        users = [m["content"] for m in row["messages"] if m["role"] == "user"]
        return users[-1] if users else ""
    return row["text"][:200]


def validate_expert_datasets(data_dirs: Sequence[str | Path], experts: Sequence[str]) -> Dict:
    files = find_expert_files(data_dirs)
    report = {"ok": {}, "missing": [], "low": {}}
    for e in experts:
        if e not in files:
            report["missing"].append(e)
            continue
        n = len(read_jsonl(files[e]))
        (report["ok"] if n >= 100 else report["low"])[e] = n
    return report
