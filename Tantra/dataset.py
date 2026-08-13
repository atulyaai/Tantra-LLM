"""
tantra/dataset.py — High-performance JSONL & raw text dataset loader for Tantra-LLM.

Loss masking: for chat-formatted items (flat system/user/assistant or ChatML
`messages`), only the assistant's own reply tokens are supervised — everything
else (role tags, system prompt, user turn) is masked out with IGNORE_INDEX so
gradient isn't wasted "learning" to reproduce input the model already has as
context. Raw / unstructured text lines are supervised in full, as before
(normal next-token pretraining).

Document boundaries: an <eos> token is inserted between items before they're
packed into the sliding token buffer, so a training window can no longer
silently straddle two unrelated documents/examples.
"""
from __future__ import annotations

import json
import os
import random
from typing import Iterator, List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import IterableDataset, DataLoader

from Tantra.utils import get_logger

log = get_logger(__name__)

IGNORE_INDEX = -100
EOS_ID = 2  # must match VocabConfig.special_tokens["<eos>"]


def format_jsonl_prompt(item: Dict[str, Any]) -> str:
    """Format a JSONL entry into a structured conversation prompt.

    Supports both flat format ({system, user, assistant}) and ChatML format
    ({messages: [{role, content}, ...]}).
    """
    messages = item.get("messages")
    if messages and isinstance(messages, list):
        parts = []
        for msg in messages:
            role = msg.get("role", "").strip()
            content = msg.get("content", "").strip()
            if role and content:
                parts.append(f"<|{'system' if role == 'system' else role}|>\n{content}")
        if parts:
            return "\n\n".join(parts)

    system = item.get("system", "").strip()
    user = item.get("user", "").strip()
    assistant = item.get("assistant", "").strip()

    parts = []
    if system:
        parts.append(f"<|system|>\n{system}")
    if user:
        parts.append(f"<|user|>\n{user}")
    if assistant:
        parts.append(f"<|assistant|>\n{assistant}")

    return "\n\n".join(parts)


def build_prompt_segments(item: Dict[str, Any]) -> Optional[List[Tuple[str, bool]]]:
    """Split a chat-formatted item into (text, is_target) segments.

    Returns None if `item` doesn't look like a structured chat record (no
    `messages` list and no system/user/assistant fields) — callers should
    fall back to treating the whole line as fully-supervised raw text.
    """
    segments: List[Tuple[str, bool]] = []

    messages = item.get("messages")
    if messages and isinstance(messages, list):
        for msg in messages:
            role = msg.get("role", "").strip()
            content = msg.get("content", "").strip()
            if not (role and content):
                continue
            tag = f"<|{'system' if role == 'system' else role}|>\n"
            segments.append((tag, False))
            segments.append((content, role == "assistant"))
            segments.append(("\n\n", False))
        return segments if segments else None

    system = item.get("system", "").strip() if isinstance(item.get("system", ""), str) else ""
    user = item.get("user", "").strip() if isinstance(item.get("user", ""), str) else ""
    assistant = item.get("assistant", "").strip() if isinstance(item.get("assistant", ""), str) else ""

    if not (system or user or assistant):
        return None

    if system:
        segments.append(("<|system|>\n", False))
        segments.append((system, False))
        segments.append(("\n\n", False))
    if user:
        segments.append(("<|user|>\n", False))
        segments.append((user, False))
        segments.append(("\n\n", False))
    if assistant:
        segments.append(("<|assistant|>\n", False))
        segments.append((assistant, True))
        segments.append(("\n\n", False))
    return segments


def _encode(tokenizer: Any, text: str) -> List[int]:
    try:
        return tokenizer.encode(text, modality="text")
    except TypeError:
        return tokenizer.encode(text)


class PretokenizedBinDataset(IterableDataset):
    """
    Streaming IterableDataset over a pre-tokenized .bin cache produced by
    Tantra/prepare_dataset.py --dump-bin (a {"tokens": IntTensor, "masks":
    BoolTensor} file). Skips BPE encode() entirely at train time — the
    whole point of building the cache — by slicing directly out of the
    already-tokenized tensor instead of re-tokenizing JSONL text.

    Chunking/masking semantics match JSONLDataset exactly: sliding window
    of seq_len+1 tokens, target at position i is IGNORE_INDEX unless
    masks[i+1] is True (assistant-reply token or a doc-boundary <eos>).
    """

    def __init__(self, bin_path: str, seq_len: int = 128,
                 max_samples: Optional[int] = None,
                 mask_non_assistant: bool = True):
        super().__init__()
        self.bin_path = bin_path
        self.seq_len = max(1, seq_len)
        self.max_samples = max_samples
        self.mask_non_assistant = mask_non_assistant

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if not os.path.exists(self.bin_path):
            log.warning(f"Bin cache path does not exist: {self.bin_path}. Yielding nothing.")
            return

        cache = torch.load(self.bin_path, map_location="cpu", weights_only=True)
        tokens = cache["tokens"]
        masks = cache["masks"]
        if tokens.numel() != masks.numel():
            log.error(f"Corrupt bin cache ({self.bin_path}): tokens/masks length mismatch "
                      f"({tokens.numel()} vs {masks.numel()}). Yielding nothing.")
            return

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        window = self.seq_len + 1
        total = tokens.numel()
        n_windows = max(0, (total - window) // self.seq_len + 1)

        effective_max = self.max_samples
        if effective_max and num_workers > 1:
            effective_max = max(1, effective_max // num_workers)

        count = 0
        for w in range(n_windows):
            if num_workers > 1 and (w % num_workers) != worker_id:
                continue
            start = w * self.seq_len
            chunk_ids = tokens[start:start + window].long()
            chunk_mask = masks[start:start + window]

            x = chunk_ids[:-1]
            y = chunk_ids[1:]
            y_is_target = chunk_mask[1:] if self.mask_non_assistant else torch.ones_like(chunk_mask[1:], dtype=torch.bool)
            y = torch.where(y_is_target, y, torch.full_like(y, IGNORE_INDEX))

            yield x, y
            count += 1
            if effective_max and count >= effective_max:
                return


def find_bin_cache(dataset_path: str) -> Optional[str]:
    """Return the sibling .bin cache path for a JSONL dataset if it exists
    (Tantra/prepare_dataset.py --dump-bin writes <name>.bin next to
    <name>.jsonl), else None."""
    candidate = os.path.splitext(dataset_path)[0] + ".bin"
    return candidate if os.path.exists(candidate) else None


class JSONLDataset(IterableDataset):
    """
    Streaming IterableDataset for JSONL files.
    Reads large dataset files line-by-line without loading entire files into RAM.
    """

    def __init__(self, jsonl_path: str, tokenizer: Any, seq_len: int = 128,
                 max_samples: Optional[int] = None, mask_non_assistant: bool = True,
                 insert_doc_boundaries: bool = True):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.seq_len = max(1, seq_len)
        self.max_samples = max_samples
        self.vocab_size = getattr(tokenizer, "vocab_size", 32000)
        self.mask_non_assistant = mask_non_assistant
        self.insert_doc_boundaries = insert_doc_boundaries
        self._unrecognized_json = 0

    def _tokenize_item(self, raw_line: str) -> Tuple[List[int], List[bool]]:
        """Return (token_ids, is_target) for one JSONL line.

        is_target[i] tells the caller whether token i should be supervised
        (True) or masked out of the loss (False). Raw / non-chat lines are
        fully supervised, matching the original (pre-masking) behavior.
        """
        try:
            item = json.loads(raw_line)
            parsed_ok = True
        except Exception:
            item = None
            parsed_ok = False

        segments = build_prompt_segments(item) if isinstance(item, dict) else None

        if segments is None:
            if parsed_ok:
                # Valid JSON, but not a recognized chat schema (e.g. {} or an
                # unexpected field layout). Do NOT fall back to the raw JSON
                # text — that would train the model on literal JSON syntax.
                # Skip the line and let the caller's schema-mismatch counter
                # catch it if this happens a lot.
                self._unrecognized_json += 1
                return [], []
            # Not JSON at all -> a genuine raw/plain-text corpus line.
            text = raw_line
            ids = _encode(self.tokenizer, text)
            return ids, [True] * len(ids)

        ids: List[int] = []
        is_target: List[bool] = []
        for text, target in segments:
            if not text:
                continue
            seg_ids = _encode(self.tokenizer, text)
            ids.extend(seg_ids)
            is_target.extend([target] * len(seg_ids))
        return ids, is_target

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if not os.path.exists(self.jsonl_path):
            log.warning(f"Dataset path does not exist: {self.jsonl_path}. Returning synthetic stream.")
            return

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        count = 0
        lines_seen = 0
        line_idx = -1
        token_buffer: List[int] = []
        mask_buffer: List[bool] = []

        with open(self.jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line_idx += 1
                # Shard lines across DataLoader workers so num_workers > 1
                # actually covers more data per wall-clock second instead of
                # every worker re-reading the same file from the start.
                if num_workers > 1 and (line_idx % num_workers) != worker_id:
                    continue
                lines_seen += 1

                ids, is_target = self._tokenize_item(line)

                # Warn loudly (once) if most lines are valid JSON we can't
                # recognize a schema for -- this usually means the dataset
                # uses field names (e.g. "prompt"/"response") that
                # build_prompt_segments()/format_jsonl_prompt() don't handle
                # yet, and the run would otherwise silently train on very
                # little real data.
                if lines_seen == 500 and self._unrecognized_json / lines_seen > 0.3:
                    log.warning(
                        f"{self._unrecognized_json}/{lines_seen} lines so far are valid JSON with an "
                        f"unrecognized schema (not 'messages' or 'system'/'user'/'assistant') and are "
                        f"being SKIPPED, not trained on. Check {self.jsonl_path}'s actual field names "
                        f"against Tantra/dataset.py:build_prompt_segments()."
                    )

                if not ids:
                    continue

                clamped = torch.tensor(ids, dtype=torch.long).clamp_(0, self.vocab_size - 1).tolist()
                token_buffer.extend(clamped)
                if self.mask_non_assistant:
                    mask_buffer.extend(is_target)
                else:
                    mask_buffer.extend([True] * len(clamped))

                if self.insert_doc_boundaries:
                    token_buffer.append(EOS_ID)
                    mask_buffer.append(True)

                # Compute effective_max once per outer line (not per inner chunk) to avoid
                # redundant division on every sample when workers > 1.
                effective_max = self.max_samples
                if effective_max and num_workers > 1:
                    effective_max = max(1, effective_max // num_workers)

                while len(token_buffer) >= self.seq_len + 1:

                    chunk_ids = token_buffer[: self.seq_len + 1]
                    chunk_mask = mask_buffer[: self.seq_len + 1]
                    token_buffer = token_buffer[self.seq_len:]
                    mask_buffer = mask_buffer[self.seq_len:]

                    x = torch.tensor(chunk_ids[:-1], dtype=torch.long)
                    y = torch.tensor(chunk_ids[1:], dtype=torch.long)
                    # Target at position i predicts token i+1; only supervise
                    # it if that *next* token belongs to an assistant turn
                    # (or masking is disabled / this is raw text).
                    y_is_target = torch.tensor(chunk_mask[1:], dtype=torch.bool)
                    y = torch.where(y_is_target, y, torch.full_like(y, IGNORE_INDEX))

                    yield x, y
                    count += 1
                    if effective_max and count >= effective_max:
                        return


def extract_corpus_sample(jsonl_path: str, output_txt_path: str, max_lines: int = 2000) -> str:
    """Extract raw text lines from JSONL to train BPE tokenizer."""
    log.info(f"Extracting sample text from {jsonl_path} for BPE vocabulary training...")
    os.makedirs(os.path.dirname(output_txt_path) or ".", exist_ok=True)

    count = 0
    with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(output_txt_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                text = format_jsonl_prompt(item)
            except Exception:
                text = line

            f_out.write(text + "\n")
            count += 1
            if count >= max_lines:
                break

    log.info(f"Extracted {count} text samples -> {output_txt_path}")
    return output_txt_path


class TopicMixedDataset(IterableDataset):
    """
    Mixes multiple dataset streams based on topic weights.
    Expects topic_paths: { "topic_name": [path1, path2, ...] }
    and weights: { "topic_name": float }

    Each topic may contain multiple JSONL files (e.g. shards). File selection
    is weighted by file size so bigger files contribute proportionally more.
    """
    def __init__(self, topic_paths: Dict[str, List[str]], weights: Dict[str, float], tokenizer: Any, seq_len: int = 128, max_samples: Optional[int] = None, seed: int = 0, mask_non_assistant: bool = True):
        super().__init__()
        self.topic_paths = {t: [p for p in ps if p] for t, ps in topic_paths.items()}
        self.weights = weights
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.seed = seed
        self.mask_non_assistant = mask_non_assistant

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("Total topic weight must be > 0")
        self.topics = [t for t in weights.keys() if self.topic_paths.get(t)]
        self.norm_weights = [weights[t] / total_weight for t in self.topics]
        if not self.topics:
            raise ValueError("No topics have any valid paths")

    def _file_iter(self, path: str) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterator over a single JSONL file, yielding chunks."""
        dataset = JSONLDataset(
            path, self.tokenizer, self.seq_len, max_samples=None,
            mask_non_assistant=self.mask_non_assistant,
        )
        yield from dataset

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Build a per-topic list of (path, weight) where file weight = file size,
        # so bigger files/shards are selected proportionally more often.
        topic_files: Dict[str, List[tuple[str, float]]] = {}
        for topic, paths in self.topic_paths.items():
            if not paths:
                continue
            entries = []
            for p in paths:
                if not os.path.exists(p):
                    continue
                try:
                    size = os.path.getsize(p)
                except OSError:
                    size = 1.0
                entries.append((p, max(size, 1.0)))
            if entries:
                topic_files[topic] = entries

        active_topics = [t for t in self.topics if t in topic_files]
        if not active_topics:
            log.warning("No valid datasets found for any topic. Yielding nothing.")
            return

        total_w = sum(self.weights[t] for t in active_topics)
        active_weights = [self.weights[t] / total_w for t in active_topics]

        # Persistent iterators per file so each file advances exactly once
        # per selection (rather than re-reading its first chunk every time).
        file_iters: Dict[str, Iterator[Tuple[torch.Tensor, torch.Tensor]]] = {}

        # Fresh RNG per iteration (also reset by DataLoader on worker restart).
        rng = random.Random(self.seed)
        count = 0
        while True:
            if not active_topics:
                break

            # Pick a topic by its normalized weight.
            topic = rng.choices(active_topics, weights=active_weights, k=1)[0]

            # Pick a file within the topic, weighted by size.
            files = topic_files[topic]
            path, _ = files[rng.choices(range(len(files)), weights=[f[1] for f in files], k=1)[0]]

            # Lazily create the file iterator on first use, then advance it once.
            if path not in file_iters:
                file_iters[path] = self._file_iter(path)
            try:
                x, y = next(file_iters[path])
            except StopIteration:
                # This file is exhausted; drop it and re-normalize weights.
                file_iters.pop(path, None)
                files.remove((path, max(os.path.getsize(path) if os.path.exists(path) else 1, 1.0)))
                if not files:
                    idx = active_topics.index(topic)
                    active_topics.pop(idx)
                    active_weights.pop(idx)
                    if not active_topics:
                        break
                    total_w = sum(active_weights)
                    active_weights = [w / total_w for w in active_weights]
                continue

            yield x, y
            count += 1
            if self.max_samples and count >= self.max_samples:
                return
