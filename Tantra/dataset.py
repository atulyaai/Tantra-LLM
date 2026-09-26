"""
Tantra/dataset.py — Turns JSONL files into training batches.

Accepted row formats (one JSON object per line):
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  {"system": "...", "user": "...", "assistant": "..."}
  {"instruction": "...", "input": "...", "output": "..."}          (Alpaca)
  {"text": "..."}                                                 (plain text)

Chat layout (must be identical in training, chat, WebUI and the probe):
  <|system|>\\n{system}\\n\\n<|user|>\\n{question}\\n\\n<|assistant|>\\n{answer}</s>

stage="sft":      only assistant tokens (and the </s> after them) are learned.
stage="pretrain": every token is learned.

Rows whose question is a generic filler like "इसके बारे में विस्तृत जानकारी दें:"
followed by a long article are treated as plain text: the article is learned,
the fake question is dropped. (Otherwise the model learns to answer a vague
question with a random article = hallucination.)

Everything is packed into full seq_len windows (no padding waste) and streamed,
so a 4 GB file never has to fit in RAM.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import IterableDataset

from Tantra.config import EOS_ID
from Tantra.utils import get_logger

log = get_logger(__name__)

IGNORE_INDEX = -100
Segment = Tuple[str, bool]  # (text, is_learned)

# Filler questions found in the Hindi corpus that carry no information.
GENERIC_PROMPTS = {
    "ज्ञानवर्धन हेतु इसका विवरण प्रस्तुत करें:",
    "इसके मुख्य बिंदु क्या हैं? स्पष्ट कीजिए:",
    "इस विषय पर प्रकाश डालें और मुख्य तथ्य बताएं:",
    "कृपया निम्नलिखित विषय को विस्तार से समझाएं:",
    "इसके बारे में विस्तृत जानकारी दें:",
}


# ── Chat formatting ──────────────────────────────────────────────────────────

def chat_prompt(user: str, system: Optional[str] = None,
                history: Optional[Sequence[Tuple[str, str]]] = None) -> str:
    """Prompt text for generation, ending right where the answer starts."""
    parts = []
    if system:
        parts.append(f"<|system|>\n{system.strip()}\n\n")
    for q, a in history or []:
        parts.append(f"<|user|>\n{q.strip()}\n\n<|assistant|>\n{a.strip()}</s>\n\n")
    parts.append(f"<|user|>\n{user.strip()}\n\n<|assistant|>\n")
    return "".join(parts)


def _norm_role(role: str) -> str:
    role = (role or "").strip().lower()
    if role in ("assistant", "gpt", "bot", "model"):
        return "assistant"
    if role in ("user", "human"):
        return "user"
    return "system"


def _is_generic(question: str, answer: str) -> bool:
    q = question.strip()
    return q in GENERIC_PROMPTS or (q.endswith(":") and len(q) < 60 and len(answer) > 400 and "\n" not in q)


def item_segments(item: Dict[str, Any], stage: str = "sft") -> Optional[List[Segment]]:
    """Split one JSONL row into (text, learned?) pieces. None = unusable row."""
    turns: List[Tuple[str, str]] = []
    msgs = item.get("messages") or item.get("conversations")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                content = str(m.get("content") or m.get("value") or "").strip()
                if content:
                    turns.append((_norm_role(m.get("role") or m.get("from")), content))
    else:
        system = str(item.get("system") or "").strip()
        user = str(item.get("user") or item.get("prompt") or item.get("instruction") or "").strip()
        if item.get("input"):
            user = f"{user}\n\n{str(item['input']).strip()}".strip()
        answer = str(item.get("assistant") or item.get("response") or item.get("output")
                     or item.get("completion") or "").strip()
        for role, text in (("system", system), ("user", user), ("assistant", answer)):
            if text:
                turns.append((role, text))
        if not turns:
            text = item.get("text") or item.get("content")
            if isinstance(text, str) and text.strip():
                return [(text.strip(), True)]
            return None

    if not turns:
        return None

    # Generic filler question + article -> plain text.
    users = [t for r, t in turns if r == "user"]
    answers = [t for r, t in turns if r == "assistant"]
    if len(users) == 1 and len(answers) == 1 and _is_generic(users[0], answers[0]):
        return [(answers[0], True)]

    segs: List[Segment] = []
    learn_all = stage == "pretrain"
    for role, text in turns:
        is_answer = role == "assistant"
        segs.append((f"<|{role}|>\n", learn_all))
        segs.append((text, is_answer or learn_all))
        if is_answer:
            segs.append(("</s>", True))   # the model must learn to stop
        segs.append(("\n\n", learn_all))
    if segs and segs[-1][0] == "\n\n":
        segs.pop()                           # row ends at </s>
    return segs if any(learned for _, learned in segs) else None


def encode_segments(tokenizer: Any, segs: List[Segment]) -> Tuple[List[int], List[bool]]:
    ids: List[int] = []
    mask: List[bool] = []
    for text, learned in segs:
        piece = [EOS_ID] if text == "</s>" else tokenizer.encode(text)
        ids.extend(piece)
        mask.extend([learned] * len(piece))
    return ids, mask


# ── Streaming, packed dataset ────────────────────────────────────────────────

class JSONLDataset(IterableDataset):
    """Streams JSONL rows -> packed (x, y) windows of length seq_len.

    y holds IGNORE_INDEX wherever the token is not learned. Documents are
    separated by </s>. Loops over the files forever (loop=False: one pass).
    """

    def __init__(self, paths: Any, tokenizer: Any, seq_len: int = 512, stage: str = "sft",
                 shuffle_buffer: int = 10000, seed: int = 42, loop: bool = True,
                 max_samples: Optional[int] = None, **_ignored: Any):
        super().__init__()
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.stage = stage
        self.shuffle_buffer = max(1, int(shuffle_buffer))
        self.seed = seed
        self.loop = loop
        self.max_samples = max_samples
        self.rows_used = 0
        self.rows_skipped = 0
        for p in self.paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Dataset file not found: {p}")

    def _lines(self, shard: int, n_shards: int, epoch: int) -> Iterator[str]:
        rng = random.Random(self.seed + epoch)
        buf: List[str] = []
        idx = -1
        for path in self.paths:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip():
                        continue
                    idx += 1
                    if idx % n_shards != shard:
                        continue
                    if self.shuffle_buffer <= 1:
                        yield line
                        continue
                    buf.append(line)
                    if len(buf) >= self.shuffle_buffer:
                        yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf)
        yield from buf

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        info = torch.utils.data.get_worker_info()
        shard, n_shards = (info.id, info.num_workers) if info else (0, 1)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
            shard, n_shards = rank * n_shards + shard, n_shards * world

        ids_buf: List[int] = []
        mask_buf: List[bool] = []
        emitted, epoch = 0, 0
        window = self.seq_len + 1
        while True:
            produced_this_epoch = False
            for line in self._lines(shard, n_shards, epoch):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    item = {"text": line.strip()}
                segs = item_segments(item, self.stage) if isinstance(item, dict) else None
                if not segs:
                    self.rows_skipped += 1
                    continue
                ids, mask = encode_segments(self.tokenizer, segs)
                if not ids:
                    continue
                if ids[-1] != EOS_ID:              # document boundary
                    ids.append(EOS_ID)
                    mask.append(True)
                self.rows_used += 1
                ids_buf.extend(ids)
                mask_buf.extend(mask)
                while len(ids_buf) >= window:
                    chunk, cmask = ids_buf[:window], mask_buf[:window]
                    ids_buf, mask_buf = ids_buf[self.seq_len:], mask_buf[self.seq_len:]
                    if not any(cmask[1:]):
                        continue
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    y[~torch.tensor(cmask[1:], dtype=torch.bool)] = IGNORE_INDEX
                    yield x, y
                    produced_this_epoch = True
                    emitted += 1
                    if self.max_samples and emitted >= self.max_samples:
                        return
            if not self.loop or not produced_this_epoch:
                if not produced_this_epoch:
                    log.warning("No trainable rows found in %s", self.paths)
                return
            epoch += 1


# ── Preference pairs (DPO) ───────────────────────────────────────────────────

class DPODataset(IterableDataset):
    """Rows: {"prompt"|"user": "...", "chosen": "...", "rejected": "...", "system"?: "..."}"""

    def __init__(self, path: str, tokenizer: Any, max_len: int = 512, max_samples: Optional[int] = None,
                 seed: int = 42):
        super().__init__()
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Preference file not found: {path}. DPO needs real chosen/rejected pairs.")
        self.path, self.tokenizer, self.max_len, self.max_samples = path, tokenizer, max_len, max_samples

    def _encode(self, prompt: str, answer: str) -> Tuple[List[int], List[int]]:
        p = self.tokenizer.encode(prompt)
        a = self.tokenizer.encode(answer.strip()) + [EOS_ID]
        ids = (p + a)[: self.max_len]
        labels = ([IGNORE_INDEX] * len(p) + a)[: self.max_len]
        pad = self.max_len - len(ids)
        return ids + [0] * pad, labels + [IGNORE_INDEX] * pad

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        count = 0
        while True:
            produced = False
            with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    q = d.get("prompt") or d.get("user") or d.get("instruction")
                    chosen, rejected = d.get("chosen"), d.get("rejected")
                    if not (q and chosen and rejected):
                        continue
                    prompt = chat_prompt(q, d.get("system"))
                    c_ids, c_lab = self._encode(prompt, chosen)
                    r_ids, r_lab = self._encode(prompt, rejected)
                    produced = True
                    yield {"chosen_input_ids": torch.tensor(c_ids), "chosen_labels": torch.tensor(c_lab),
                           "rejected_input_ids": torch.tensor(r_ids), "rejected_labels": torch.tensor(r_lab)}
                    count += 1
                    if self.max_samples and count >= self.max_samples:
                        return
            if not produced:
                return
