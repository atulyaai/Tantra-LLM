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

import hashlib
import json
import math
import os
import random
from typing import Iterator, List, Dict, Any, Optional, Tuple


import torch
from torch.utils.data import IterableDataset, DataLoader

from Tantra.utils import get_logger

log = get_logger(__name__)

import re

IGNORE_INDEX = -100
EOS_ID = 2  # must match VocabConfig.special_tokens["<eos>"]
_NON_LATIN_SCRIPT_REGEX = re.compile(r'[\u0400-\u04FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u0900-\u0D7F\u0E00-\u0E7F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]')


class TokenJuiceEngine:
    """Dataset signal filtering, optional synthetic enrichment, and weighting."""

    def __init__(self, entropy_threshold: float = 0.5, enrichment_rate: float = 0.1):
        self.entropy_threshold, self.enrichment_rate = entropy_threshold, enrichment_rate
        self.synthetic_pool: List[Tuple[List[int], List[int]]] = []

    def register_synthetic_pair(self, input_ids: List[int], target_ids: List[int]) -> None:
        if isinstance(input_ids, list) and isinstance(target_ids, list) and all(isinstance(t, int) for t in input_ids + target_ids):
            self.synthetic_pool.append((input_ids, target_ids))

    def compute_token_entropy(self, token_ids: List[int], vocab_size: int = 32768) -> float:
        if not token_ids:
            return 0.0
        counts: Dict[int, int] = {}
        for token in token_ids:
            counts[token] = counts.get(token, 0) + 1
        entropy = -sum((count / len(token_ids)) * math.log2(count / len(token_ids)) for count in counts.values())
        return entropy / max(math.log2(vocab_size), 1.0)

    def squeeze_tokens(self, token_ids: List[int], vocab_size: int = 32768) -> List[int]:
        if len(token_ids) < 4:
            return token_ids
        kept = [chunk for index in range(0, len(token_ids), 8) if (chunk := token_ids[index:index + 8]) and (index == 0 or self.compute_token_entropy(chunk, vocab_size) >= self.entropy_threshold)]
        return [token for chunk in kept for token in chunk] or token_ids

    @staticmethod
    def _fit_to_length(ids: List[int], length: int) -> List[int]:
        return ids[:length] + [0] * max(0, length - len(ids))

    def enrich_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.synthetic_pool or random.random() > self.enrichment_rate:
            return x, y
        input_ids, target_ids = random.choice(self.synthetic_pool)
        full_ids = input_ids + target_ids
        length = x.shape[-1]
        pad_len = max(0, length + 1 - len(full_ids))
        sample_ids = (full_ids + [0] * pad_len)[: length + 1]

        prompt_len = len(input_ids)
        ans_len = len(target_ids)
        target_mask = [False] * prompt_len + [True] * ans_len + [False] * pad_len
        target_mask = target_mask[: length + 1]

        x_syn = torch.tensor(sample_ids[:-1], dtype=x.dtype, device=x.device)
        y_syn = torch.tensor(sample_ids[1:], dtype=y.dtype, device=y.device)
        y_mask = torch.tensor(target_mask[1:], dtype=torch.bool, device=y.device)
        y_syn = torch.where(y_mask, y_syn, torch.full_like(y_syn, IGNORE_INDEX))

        x[-1:] = x_syn.unsqueeze(0)
        y[-1:] = y_syn.unsqueeze(0)
        return x, y

    @staticmethod
    def compute_dynamic_loss_weights(targets: torch.Tensor, high_priority_ids: List[int]) -> torch.Tensor:
        weights = torch.ones_like(targets, dtype=torch.float32)
        if high_priority_ids:
            weights[torch.isin(targets, torch.tensor(high_priority_ids, device=targets.device))] = 2.5
        return weights


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

    Supports:
      - messages: [{"role": "...", "content": "..."}]
      - conversations: [{"from": "human/gpt", "value": "..."}]
      - system / user / assistant
      - instruction / input / output (Alpaca format)
      - prompt / response or prompt / completion
    """
    segments: List[Tuple[str, bool]] = []

    # 1. Standard OpenAI chat messages format
    messages = item.get("messages") or item.get("conversations")
    if messages and isinstance(messages, list):
        for msg in messages:
            role = (msg.get("role") or msg.get("from") or "").strip()
            content = (msg.get("content") or msg.get("value") or "").strip()
            if not (role and content):
                continue
            norm_role = "assistant" if role in ("assistant", "gpt", "bot", "model") else ("user" if role in ("user", "human") else "system")
            tag = f"<|{norm_role}|>\n"
            segments.append((tag, False))
            segments.append((content, norm_role == "assistant"))
            segments.append(("\n\n", False))
        return segments if segments else None

    # 2. Extract flat system, user, assistant fields (including Alpaca & prompt/response aliases)
    system = (item.get("system") or item.get("system_prompt") or "").strip()
    user = (item.get("user") or item.get("prompt") or item.get("query") or "").strip()
    assistant = (item.get("assistant") or item.get("response") or item.get("completion") or item.get("output") or "").strip()

    # If Alpaca-style instruction (+ optional input):
    instruction = item.get("instruction", "").strip() if isinstance(item.get("instruction", ""), str) else ""
    input_text = item.get("input", "").strip() if isinstance(item.get("input", ""), str) else ""
    if instruction:
        user = f"{instruction}\n\n{input_text}".strip() if input_text else instruction

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
        # FIX #3 (HIGH): Supervise the <|assistant|\n> tag itself (True, not False).
        # The model must learn to emit its own turn-opener during generation;
        # masking it out means it only learns the words that follow, not the
        # transition token, causing inference failures (missing/wrong turn tags).
        segments.append(("<|assistant|>\n", True))
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


def is_val_line(raw_line: str, val_ratio: float = 0.05) -> bool:

    """Return True if raw_line falls into the held-out validation split based on content hash."""
    if val_ratio <= 0:
        return False
    h = int(hashlib.md5(raw_line.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
    return (h % 100) < int(val_ratio * 100)


class JSONLDataset(IterableDataset):

    """
    Streaming IterableDataset for JSONL files.
    Reads large dataset files line-by-line without loading entire files into RAM.
    """

    def __init__(self, jsonl_path: str, tokenizer: Any, seq_len: int = 128,
                 max_samples: Optional[int] = None, mask_non_assistant: bool = True,
                 insert_doc_boundaries: bool = True, shuffle: bool = True,
                 shuffle_buf_size: int = 2000, seed: int = 42,
                 val_ratio: float = 0.05, split: str = "train",
                 pack_sequences: bool = True):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.seq_len = max(1, seq_len)
        self.max_samples = max_samples
        self.vocab_size = getattr(tokenizer, "vocab_size", 32768)
        self.mask_non_assistant = mask_non_assistant
        self.insert_doc_boundaries = insert_doc_boundaries
        self.shuffle = shuffle if split == "train" else False
        self.shuffle_buf_size = max(1, shuffle_buf_size)
        self.seed = seed
        self.val_ratio = max(0.0, min(0.5, val_ratio))
        self.split = split.lower().strip()
        self.pack_sequences = pack_sequences
        self._unrecognized_json = 0


    def __bool__(self) -> bool:
        return True

    def _tokenize_item(self, raw_line: str) -> Tuple[List[int], List[bool]]:

        """Return (token_ids, is_target) for one JSONL line.

        is_target[i] tells the caller whether token i should be supervised
        (True) or masked out of the loss (False). Raw / non-chat lines are
        fully supervised, matching the original (pre-masking) behavior.
        """
        # English-Only Filter: Skip lines containing non-Latin scripts (Devanagari, CJK, Cyrillic, Arabic, Indic, etc.)
        if _NON_LATIN_SCRIPT_REGEX.search(raw_line):
            return [], []

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
            if target:
                # Supervised assistant reply ends — append and supervise the real <eos> token
                ids.append(EOS_ID)
                is_target.append(True)
        return ids, is_target

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if not os.path.exists(self.jsonl_path):
            log.warning(f"Dataset path does not exist: {self.jsonl_path}. Auto-generating 4-track curriculum...")
            build_4track_curriculum(datasets_dir=os.path.dirname(self.jsonl_path) or "Datasets")
        
        if not os.path.exists(self.jsonl_path):
            log.warning(f"Fallback to synthetic tokens stream for: {self.jsonl_path}")
            while True:
                x = torch.randint(0, min(self.tokenizer.vocab_size, 32000), (self.seq_len,))
                y = x.clone()
                yield x, y

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        count = 0
        lines_seen = 0
        line_idx = -1
        token_buffer: List[int] = []
        mask_buffer: List[bool] = []
        epoch = 0

        def _process_line(raw_line: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
            nonlocal lines_seen, line_idx, token_buffer, mask_buffer
            line_idx += 1
            if num_workers > 1 and (line_idx % num_workers) != worker_id:
                return []
            lines_seen += 1

            ids, is_target = self._tokenize_item(raw_line)

            if lines_seen == 500 and self._unrecognized_json / lines_seen > 0.3:
                log.warning(
                    f"{self._unrecognized_json}/{lines_seen} lines so far are valid JSON with an "
                    f"unrecognized schema (not 'messages' or 'system'/'user'/'assistant') and are "
                    f"being SKIPPED, not trained on. Check {self.jsonl_path}'s actual field names "
                    f"against Tantra/dataset.py:build_prompt_segments()."
                )

            if not ids:
                return []

            clamped = torch.tensor(ids, dtype=torch.long).clamp_(0, self.vocab_size - 1).tolist()
            samples = []

            if self.pack_sequences:
                token_buffer.extend(clamped)
                mask_buffer.extend(is_target if self.mask_non_assistant else [True] * len(clamped))
                if self.insert_doc_boundaries:
                    if not token_buffer or token_buffer[-1] != EOS_ID:
                        token_buffer.append(EOS_ID)
                        mask_buffer.append(True)

                while len(token_buffer) >= self.seq_len + 1:
                    chunk_ids = token_buffer[: self.seq_len + 1]
                    chunk_mask = mask_buffer[: self.seq_len + 1]
                    token_buffer = token_buffer[self.seq_len:]
                    mask_buffer = mask_buffer[self.seq_len:]

                    # Only emit chunks that have at least one supervised target token
                    if self.mask_non_assistant and not any(chunk_mask[1:]):
                        continue

                    x = torch.tensor(chunk_ids[:-1], dtype=torch.long)
                    y = torch.tensor(chunk_ids[1:], dtype=torch.long)
                    y_is_target = torch.tensor(chunk_mask[1:], dtype=torch.bool)
                    y = torch.where(y_is_target, y, torch.full_like(y, IGNORE_INDEX))
                    samples.append((x, y))

            elif self.mask_non_assistant:
                if len(clamped) >= 2:
                    # Multi-chunk sliding window: guarantees 100% of the assistant answer is supervised
                    stride = max(1, self.seq_len // 2) if len(clamped) > self.seq_len + 1 else self.seq_len
                    for start_idx in range(0, len(clamped), stride):
                        end_idx = start_idx + self.seq_len + 1
                        chunk_ids = clamped[start_idx:end_idx]
                        chunk_mask = is_target[start_idx:end_idx]

                        # Skip chunks that have no supervised assistant tokens (e.g. pure prompt prefix)
                        if not any(chunk_mask):
                            continue

                        pad_len = max(0, (self.seq_len + 1) - len(chunk_ids))
                        sample_ids = (chunk_ids + [0] * pad_len)[: self.seq_len + 1]
                        sample_mask = (chunk_mask + [False] * pad_len)[: self.seq_len + 1]

                        x = torch.tensor(sample_ids[:-1], dtype=torch.long)
                        y = torch.tensor(sample_ids[1:], dtype=torch.long)
                        y_is_target = torch.tensor(sample_mask[1:], dtype=torch.bool)
                        y = torch.where(y_is_target, y, torch.full_like(y, IGNORE_INDEX))
                        samples.append((x, y))

                        if end_idx >= len(clamped):
                            break

            else:
                token_buffer.extend(clamped)
                mask_buffer.extend([True] * len(clamped))
                if self.insert_doc_boundaries:
                    if not token_buffer or token_buffer[-1] != EOS_ID:
                        token_buffer.append(EOS_ID)
                        mask_buffer.append(True)

                while len(token_buffer) >= self.seq_len + 1:
                    chunk_ids = token_buffer[: self.seq_len + 1]
                    chunk_mask = mask_buffer[: self.seq_len + 1]
                    token_buffer = token_buffer[self.seq_len:]
                    mask_buffer = mask_buffer[self.seq_len:]

                    x = torch.tensor(chunk_ids[:-1], dtype=torch.long)
                    y = torch.tensor(chunk_ids[1:], dtype=torch.long)
                    y_is_target = torch.tensor(chunk_mask[1:], dtype=torch.bool)
                    y = torch.where(y_is_target, y, torch.full_like(y, IGNORE_INDEX))
                    samples.append((x, y))


            return samples

        while True:
            file_had_lines = False
            rng = random.Random(self.seed + epoch)
            shuffle_buffer: List[str] = []

            with open(self.jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    file_had_lines = True
                    if self.val_ratio > 0:
                        is_val = is_val_line(line, self.val_ratio)
                        if self.split == "val" and not is_val:
                            continue
                        elif self.split == "train" and is_val:
                            continue

                    if self.shuffle and self.shuffle_buf_size > 1:

                        shuffle_buffer.append(line)
                        if len(shuffle_buffer) >= self.shuffle_buf_size:
                            idx = rng.randrange(len(shuffle_buffer))
                            popped = shuffle_buffer.pop(idx)
                            for x, y in _process_line(popped):
                                yield x, y
                                count += 1
                                effective_max = self.max_samples
                                if effective_max and num_workers > 1:
                                    effective_max = max(1, effective_max // num_workers)
                                if effective_max and count >= effective_max:
                                    return
                    else:
                        for x, y in _process_line(line):
                            yield x, y
                            count += 1
                            effective_max = self.max_samples
                            if effective_max and num_workers > 1:
                                effective_max = max(1, effective_max // num_workers)
                            if effective_max and count >= effective_max:
                                return

            if self.shuffle and shuffle_buffer:
                rng.shuffle(shuffle_buffer)
                for line in shuffle_buffer:
                    for x, y in _process_line(line):
                        yield x, y
                        count += 1
                        effective_max = self.max_samples
                        if effective_max and num_workers > 1:
                            effective_max = max(1, effective_max // num_workers)
                        if effective_max and count >= effective_max:
                            return
                shuffle_buffer.clear()

            if not file_had_lines:
                break
            epoch += 1


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
                topic_files[topic] = [f for f in files if f[0] != path]
                files = topic_files[topic]
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


class DPODataset(IterableDataset):
    """
    DPO (Direct Preference Optimization) Dataset.
    Loads JSONL items containing:
      {"prompt": "...", "chosen": "...", "rejected": "..."}
      OR
      {"system": "...", "user": "...", "chosen": "...", "rejected": "..."}
    Yields dictionary with tokenized tensors:
      chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels
    """
    def __init__(self, path: str, tokenizer: Any, max_len: int = 128, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_samples = max_samples
        self.seed = seed

    def _encode_pair(self, prompt: str, chosen: str, rejected: str) -> Optional[Dict[str, torch.Tensor]]:
        if not prompt or not chosen or not rejected:
            return None
        prompt_text = prompt if "<|user|>" in prompt else f"<|system|>\nYou are Tantra, a helpful, precise, and polite AI assistant created by Atulya AI.<|user|>\n{prompt.strip()}<|assistant|>\n"
        
        prompt_ids = self.tokenizer.encode(prompt_text)
        chosen_resp_ids = self.tokenizer.encode(chosen.strip()) + [EOS_ID]
        rejected_resp_ids = self.tokenizer.encode(rejected.strip()) + [EOS_ID]
        
        chosen_ids = (prompt_ids + chosen_resp_ids)[:self.max_len]
        chosen_labels = [IGNORE_INDEX] * len(prompt_ids) + chosen_resp_ids
        chosen_labels = chosen_labels[:self.max_len]
        
        rejected_ids = (prompt_ids + rejected_resp_ids)[:self.max_len]
        rejected_labels = [IGNORE_INDEX] * len(prompt_ids) + rejected_resp_ids
        rejected_labels = rejected_labels[:self.max_len]
        
        pad_len_c = max(0, self.max_len - len(chosen_ids))
        pad_len_r = max(0, self.max_len - len(rejected_ids))
        
        chosen_ids += [0] * pad_len_c
        chosen_labels += [IGNORE_INDEX] * pad_len_c
        
        rejected_ids += [0] * pad_len_r
        rejected_labels += [IGNORE_INDEX] * pad_len_r
        
        return {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if not os.path.exists(self.path):
            log.warning(f"DPO dataset path not found: {self.path}. Auto-generating preference pairs...")
            generate_gold_datasets(datasets_dir=os.path.dirname(self.path) or "Datasets")
        if not os.path.exists(self.path):
            log.warning(f"Using in-memory preference seed pairs for DPO alignment.")
            seeds = [
                ("Hello! Who are you?", "Hello! I am Tantra, an AI assistant developed by Atulya AI.", "I don't know who made me."),
                ("Write a Python function to compute factorial.", "def factorial(n: int) -> int:\n    return 1 if n in (0, 1) else n * factorial(n - 1)", "def fact(n): return n"),
                ("What is 15 * 6?", "15 * 6 = 90.", "15 * 6 is 100."),
            ]
            while True:
                for p, c, r in seeds:
                    item = self._encode_pair(p, c, r)
                    if item is not None:
                        yield item
            return
        count = 0
        while True:
            with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    prompt = data.get("prompt") or data.get("instruction") or data.get("user", "")
                    if "system" in data and "user" in data:
                        prompt = f"<|system|>\n{data['system']}<|user|>\n{data['user']}<|assistant|>\n"
                    chosen = data.get("chosen") or data.get("response_chosen") or data.get("assistant", "")
                    rejected = data.get("rejected") or data.get("response_rejected", "")
                    
                    item = self._encode_pair(prompt, chosen, rejected)
                    if item is not None:
                        yield item
                        count += 1
                        if self.max_samples and count >= self.max_samples:
                            return


CURRICULUM_TRACKS = {
    "expert_conversation.jsonl": ["conversation", "dialogue", "greeting", "persona", "chat", "identity"],
    "expert_code.jsonl": ["code", "python", "javascript", "cpp", "java", "sql", "algorithm", "function"],
    "expert_math_science.jsonl": ["math", "science", "physics", "gsm8k", "algebra", "arithmetic", "chemistry", "biology"],
    "expert_general.jsonl": ["general", "history", "geography", "knowledge", "reasoning", "facts", "summary"]
}


def generate_gold_datasets(datasets_dir: str = "Datasets", force: bool = False) -> None:
    """Safely seeds synthetic reasoning datasets if real datasets are missing on a fresh machine.
    NEVER overwrites or truncates existing non-empty gold_corpus.jsonl or preference_pairs.jsonl."""
    import hashlib
    os.makedirs(datasets_dir, exist_ok=True)
    gold_path = os.path.join(datasets_dir, "gold_corpus.jsonl")
    pref_path = os.path.join(datasets_dir, "preference_pairs.jsonl")

    # Safety protection: if real datasets already exist, never overwrite them!
    has_real_gold = os.path.exists(gold_path) and os.path.getsize(gold_path) > 500_000
    has_real_pref = os.path.exists(pref_path) and os.path.getsize(pref_path) > 100_000

    if has_real_gold and has_real_pref and not force:
        log.info(f"⚡ [CACHE HIT] Preserving real gold & preference datasets in {datasets_dir}/.")
        return
    elif has_real_gold and not force:
        log.info(f"⚡ [CACHE HIT] Preserving real gold_corpus.jsonl ({os.path.getsize(gold_path):,} bytes).")
        return

    log.info("🚀 Seeding High-Diversity Synthetic Gold Corpus & Preference Pairs...")
    seen_hashes = set()
    gold_samples = []

    # Read any existing samples first so we NEVER lose or duplicate data
    if os.path.exists(gold_path):
        try:
            with open(gold_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        d = json.loads(line)
                        u = d.get("user") or d.get("instruction") or d.get("prompt") or ""
                        a = d.get("assistant") or d.get("output") or d.get("response") or ""
                        h = hashlib.sha256((str(u).strip() + "|||" + str(a).strip()).encode("utf-8")).hexdigest()
                        seen_hashes.add(h)
                        gold_samples.append(d)
                    except Exception:
                        pass
        except Exception as e:
            log.warning(f"Could not read existing gold corpus: {e}")

    def _add_sample(domain: str, instruction: str, output: str, complexity: int = 1):
        h = hashlib.sha256((instruction.strip() + "|||" + output.strip()).encode("utf-8")).hexdigest()
        if h in seen_hashes:
            return
        seen_hashes.add(h)
        gold_samples.append({
            "instruction": instruction.strip(),
            "input": "",
            "output": output.strip(),
            "domain": domain.lower(),
            "complexity": complexity
        })

    # ── 1. Persona, Greetings, Chit-Chat & Identity ──────────────────────────
    personas = [
        ("Hello! How are you today?", "Hello! I am doing well, thank you for asking. How can I assist you with your projects, questions, or ideas today?"),
        ("Hi there!", "Hello! It is great to connect with you. What would you like to explore or work on today?"),
        ("Good morning!", "Good morning! I hope you have a wonderful and productive day ahead. How can I help you get started?"),
        ("Good evening!", "Good evening! How was your day? Let me know what you would like to work on or discuss tonight."),
        ("Who created you and what is your name?", "I am Tantra, an omnimodal foundation AI model created by Atulya AI. My neural backbone features ALRA linear resonance attention and BitNet 1.58-bit ternary quantization."),
        ("What is your name?", "My name is Tantra. I am a helpful, precise, and polite AI assistant created by Atulya AI."),
        ("Who is Atulya AI?", "Atulya AI is an advanced artificial intelligence research and engineering initiative dedicated to creating high-efficiency, sovereign, and locally executable AI systems like Tantra LLM."),
        ("What can you do?", "I can assist you with friendly conversations, writing, brainstorming, step-by-step reasoning, coding, science, and answering everyday questions clearly and accurately."),
        ("Are you ChatGPT or Claude?", "No, I am Tantra, an independent AI foundation model developed by Atulya AI. I run on custom NeuroCore architecture designed for local and efficient compute."),
        ("How are you feeling today?", "As an AI, I don't have feelings in the human sense, but I am operating at peak performance and ready to help you with anything you need!"),
        ("Tell me a fun fact!", "Here is a fun fact: Honey never spoils! Archaeologists have discovered pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible."),
        ("Tell me a short joke.", "Why do programmers prefer dark mode? Because light attracts bugs!"),
        ("Thank you so much for your help!", "You're very welcome! I'm glad I could assist. Feel free to ask anytime if you need anything else."),
        ("I had a stressful day today.", "I'm sorry to hear that. Stressful days can be challenging. Take a deep breath, give yourself some credit for getting through it, and remember to get some relaxing rest tonight."),
        ("How can I stay motivated when working on hard tasks?", "A great strategy is the 'Pomodoro Technique': work in focused 25-minute intervals followed by a 5-minute break. Also, break your big goal into small, satisfying daily milestones!"),
        ("What makes a good cup of coffee?", "A great cup of coffee relies on four key elements: freshly roasted whole beans, the correct grind size for your brew method, pure water at around 195°F–205°F (90°C–96°C), and the right coffee-to-water ratio (typically 1:15 to 1:17)."),
        ("How can I improve my sleep quality?", "To get better sleep: keep a consistent bedtime schedule, avoid screens 30–60 minutes before sleeping, keep your bedroom cool and dark, and limit caffeine in the late afternoon."),
        ("What is the meaning of life?", "While philosophers have debated this for centuries, many find meaning in creating meaningful connections with others, pursuing personal growth and curiosity, and leaving the world a little better than they found it."),
        ("Can you help me brainstorm some creative hobbies?", "Certainly! Here are some rewarding hobbies to explore:\n1. Digital illustration or watercolor painting\n2. Creative writing or journaling\n3. Gardening or caring for indoor bonsai\n4. Learning a musical instrument like ukulele or piano\n5. Cooking artisanal recipes from different cultures."),
        ("What is your favorite topic to discuss?", "I love discussing everything from astrophysics and computer systems to philosophy, literature, and everyday life curiosities! What is your favorite topic?")
    ]
    for p, r in personas:
        _add_sample("conversation", p, r, complexity=1)

    # ── 2. Algorithmic Math & Arithmetic Generation ──────────────────────────
    import random
    rng = random.Random(42)

    for i in range(1, 150):
        # Linear equations
        a = rng.randint(2, 20)
        b = rng.randint(1, 50)
        c = a * rng.randint(1, 20) + b
        ans = (c - b) // a
        _add_sample("math", f"Solve for x: {a}x + {b} = {c}",
                    f"Step 1: Subtract {b} from both sides: {a}x = {c - b}.\nStep 2: Divide both sides by {a}: x = {ans}.\nFinal Answer: x = {ans}", complexity=1)

        # Quadratic derivations
        r1, r2 = rng.randint(1, 10), rng.randint(1, 10)
        b_coeff = -(r1 + r2)
        c_coeff = r1 * r2
        sign_b = f"- {abs(b_coeff)}" if b_coeff < 0 else f"+ {b_coeff}"
        _add_sample("math", f"Factor the quadratic equation: x^2 {sign_b}x + {c_coeff} = 0",
                    f"To factor x^2 {sign_b}x + {c_coeff} = 0:\nFind two numbers that multiply to {c_coeff} and add to {b_coeff}: {(-r1)} and {(-r2)}.\nFactored form: (x - {r1})(x - {r2}) = 0.\nRoots: x = {r1}, x = {r2}.", complexity=2)

        # Calculus Derivatives
        p_pow = rng.randint(2, 6)
        c_val = rng.randint(2, 9)
        _add_sample("math", f"What is the derivative of f(x) = {c_val}x^{p_pow} + {a}x?",
                    f"Using the power rule d/dx(x^n) = n*x^(n-1):\nf'(x) = {c_val * p_pow}x^{p_pow - 1} + {a}.", complexity=2)

        # Geometry
        rad = rng.randint(2, 25)
        _add_sample("math", f"What is the volume of a sphere with radius r = {rad}?",
                    f"The formula for the volume of a sphere is V = (4/3) * pi * r^3.\nSubstituting r = {rad}:\nV = (4/3) * pi * ({rad}^3) = (4/3) * pi * ({rad**3}) = {round((4/3)*3.14159*(rad**3), 2)} cubic units.", complexity=2)

    # ── 3. Algorithmic Code Synthesis (Python, JS, C++, Java) ─────────────────
    code_templates = [
        ("Write a Python function to check if a string is a palindrome.",
         "```python\ndef is_palindrome(s: str) -> bool:\n    \"\"\"Checks if string reads same backward.\"\"\"\n    clean = ''.join(c.lower() for c in s if c.isalnum())\n    return clean == clean[::-1]\n\n# Test\nprint(is_palindrome('radar'))  # True\n```"),
        ("Write a Python function to reverse a list in-place.",
         "```python\ndef reverse_list(items: list) -> list:\n    \"\"\"Reverses list in-place using two pointers.\"\"\"\n    left, right = 0, len(items) - 1\n    while left < right:\n        items[left], items[right] = items[right], items[left]\n        left += 1\n        right -= 1\n    return items\n```"),
        ("Write a Python function to find the factorial of an integer.",
         "```python\ndef factorial(n: int) -> int:\n    \"\"\"Computes n! iteratively.\"\"\"\n    if n < 0: raise ValueError('Factorial not defined for negative numbers')\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res\n```"),
        ("Write a JavaScript function to implement binary search.",
         "```javascript\nfunction binarySearch(arr, target) {\n    let left = 0, right = arr.length - 1;\n    while (left <= right) {\n        const mid = Math.floor((left + right) / 2);\n        if (arr[mid] === target) return mid;\n        if (arr[mid] < target) left = mid + 1;\n        else right = mid - 1;\n    }\n    return -1;\n}\n```"),
        ("Write a C++ function to check if a number is prime.",
         "```cpp\nbool isPrime(int n) {\n    if (n <= 1) return false;\n    if (n <= 3) return true;\n    if (n % 2 == 0 || n % 3 == 0) return false;\n    for (int i = 5; i * i <= n; i += 6) {\n        if (n % i == 0 || n % (i + 2) == 0) return false;\n    }\n    return true;\n}\n```"),
        ("Write a Java method to find the maximum element in an array.",
         "```java\npublic class ArrayUtils {\n    public static int findMax(int[] nums) {\n        if (nums == null || nums.length == 0) throw new IllegalArgumentException('Empty array');\n        int max = nums[0];\n        for (int v : nums) if (v > max) max = v;\n        return max;\n    }\n}\n```")
    ]
    for p, c in code_templates:
        _add_sample("code", p, c, complexity=2)

    # ── 4. Science & General Knowledge ───────────────────────────────────────
    science_facts = [
        ("What is photosynthesis and how does it work?",
         "Photosynthesis is the multi-stage biochemical process occurring in chloroplasts where chlorophyll pigments capture photon energy to convert 6CO2 + 6H2O -> C6H12O6 + 6O2, releasing molecular oxygen."),
        ("State Newton's three laws of motion.",
         "1. Law of Inertia: An object remains at rest or in uniform motion unless acted upon by a net external force.\n2. F = m*a: Acceleration is directly proportional to net force and inversely proportional to mass.\n3. Action-Reaction: For every action, there is an equal and opposite reaction."),
        ("What is the difference between DNA and RNA?",
         "DNA is double-stranded with deoxyribose sugar and thymine (A-T, G-C). RNA is single-stranded with ribose sugar and uracil in place of thymine (A-U, G-C)."),
        ("Explain the theory of Special Relativity.",
         "Albert Einstein's 1905 Special Relativity states that the laws of physics are identical in all inertial frames, and the speed of light in vacuum (c) is constant for all observers regardless of motion, leading to time dilation and mass-energy equivalence (E = mc^2).")
    ]
    for q, a in science_facts:
        _add_sample("science", q, a, complexity=2)

    # Only write gold corpus if not already populated with real data
    if not has_real_gold:
        with open(gold_path, "w", encoding="utf-8") as f:
            for s in gold_samples:
                f.write(json.dumps(s) + "\n")
        log.info(f"✅ Generated {len(gold_samples)} unique seed gold samples.")

    # ── 5. DPO Preference Pairs Generation ───────────────────────────────────
    if not has_real_pref:
        pref_samples = [
            {
                "prompt": "Hello! Who are you?",
                "chosen": "Hello! I am Tantra, an AI assistant developed by Atulya AI.",
                "rejected": "I am a generic language model. The first step in the list is the world."
            },
            {
                "prompt": "What is 15 * 6?",
                "chosen": "15 * 6 = 90.",
                "rejected": "The three main reasons why 15 times 6 is popular are 1. The first step is 100."
            },
            {
                "prompt": "Write a Python function to reverse a string.",
                "chosen": "def reverse_string(s: str) -> str:\n    return s[::-1]",
                "rejected": "# Test the string\nTest Test Test Test Test Test Test"
            },
            {
                "prompt": "What is the formula for the volume of a sphere?",
                "chosen": "The formula for the volume of a sphere is V = (4/3) * pi * r^3, where r is the radius.",
                "rejected": "The radius of a sphere is defined as the ratio of the radius to its radius."
            }
        ]
        with open(pref_path, "w", encoding="utf-8") as f:
            for p in pref_samples:
                f.write(json.dumps(p) + "\n")
        log.info(f"✅ Generated {len(pref_samples)} seed DPO pairs.")


class QualityFilterAndDeduplicator:
    """Filters low-quality, boilerplate, repetitive, or duplicate data samples."""

    def __init__(self, min_prompt_len: int = 5, min_output_len: int = 20, max_token_len: int = 16384):
        self.min_prompt_len = min_prompt_len
        self.min_output_len = min_output_len
        self.max_token_len = max_token_len
        self.seen_hashes = set()
        self.banned_phrases = [
            "404 not found", "access denied", "cookie policy", "terms of service",
            "privacy policy", "all rights reserved", "lorem ipsum", "subscribe to",
            "javascript is disabled", "please enable cookies", "var _0x", "window.__initial_state__"
        ]

    def is_clean(self, prompt: str, output: str) -> bool:
        p = (prompt or "").strip()
        o = (output or "").strip()
        if len(p) < self.min_prompt_len or len(o) < self.min_output_len:
            return False
        if len(p) + len(o) > self.max_token_len * 5:
            return False

        o_lower = o.lower()
        if any(banned in o_lower for banned in self.banned_phrases):
            return False

        # Repetition check (detect pathological repetition)
        words = o.split()
        if len(words) >= 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.15:
                return False

        # Deterministic 64-bit SHA-256 content hash
        norm = (p.lower() + "|||" + o.lower())
        h = hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()[:16]
        if h in self.seen_hashes:
            return False
        self.seen_hashes.add(h)
        return True


def ingest_open_super_corpus(datasets_dir: str = "Datasets", max_samples: int = 350_000) -> int:
    """Download and stream high-density open-source multi-domain datasets (UltraChat, CodeAlpaca, MetaMath, Dolly, DPO)."""
    os.makedirs(datasets_dir, exist_ok=True)
    master_path = os.path.join(datasets_dir, "master_corpus.jsonl")
    if os.path.exists(master_path) and os.path.getsize(master_path) > 1_000_000:
        log.info(f"⚡ [CACHE HIT] Master corpus already populated ({os.path.getsize(master_path)/1e6:.1f} MB). Skipping re-ingestion.")
        return 0
    pref_path = os.path.join(datasets_dir, "preference_pairs.jsonl")
    filter_dedup = QualityFilterAndDeduplicator()
    
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("HuggingFace `datasets` not installed. Run: pip install datasets")
        return 0

    total_added = 0
    with open(master_path, "a", encoding="utf-8") as out_f:
        # 1. Clean General Instructions (Alpaca Cleaned 52K)
        try:
            log.info("📥 [1/7] Ingesting Alpaca Cleaned Instructions (52K)...")
            ds = load_dataset("yahma/alpaca-cleaned", split="train")
            for it in ds:
                out_f.write(json.dumps({
                    "instruction": it.get("instruction", ""),
                    "input": it.get("input", ""),
                    "output": it.get("output", ""),
                    "domain": "general"
                }) + "\n")
                total_added += 1
        except Exception as e:
            log.warning(f"Could not load alpaca-cleaned: {e}")

        # 2. Multi-Turn Dialogue & Persona (UltraChat 200K - 50K sample)
        try:
            log.info("📥 [2/7] Ingesting UltraChat Multi-Turn Conversations (50K)...")
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:50000]")
            for it in ds:
                msgs = it.get("messages", [])
                if len(msgs) >= 2:
                    u = msgs[0].get("content", "")
                    a = msgs[1].get("content", "")
                    out_f.write(json.dumps({
                        "user": u,
                        "assistant": a,
                        "domain": "conversation"
                    }) + "\n")
                    total_added += 1
        except Exception as e:
            log.warning(f"Could not load ultrachat: {e}")

        # 3. High-Quality Fact & World Reasoning (Databricks Dolly 15K)
        try:
            log.info("📥 [3/7] Ingesting Databricks Dolly Knowledge Base (15K)...")
            ds = load_dataset("databricks/databricks-dolly-15k", split="train")
            for it in ds:
                out_f.write(json.dumps({
                    "instruction": it.get("instruction", ""),
                    "input": it.get("context", ""),
                    "output": it.get("response", ""),
                    "domain": "general"
                }) + "\n")
                total_added += 1
        except Exception as e:
            log.warning(f"Could not load dolly-15k: {e}")

        # 4. Python Algorithms & Doctests (18.6K)
        try:
            log.info("📥 [4/7] Ingesting Python Code Instructions (18K)...")
            ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
            for it in ds:
                out_f.write(json.dumps({
                    "instruction": it.get("instruction", ""),
                    "input": it.get("input", ""),
                    "output": it.get("output", ""),
                    "domain": "code"
                }) + "\n")
                total_added += 1
        except Exception as e:
            log.warning(f"Could not load python_code_instructions: {e}")

        # 5. Multi-Language Code (CodeAlpaca 20K - JS, C++, Python, Java)
        try:
            log.info("📥 [5/7] Ingesting CodeAlpaca Multi-Language Programming (20K)...")
            ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
            for it in ds:
                out_f.write(json.dumps({
                    "instruction": it.get("instruction", ""),
                    "input": it.get("input", ""),
                    "output": it.get("output", ""),
                    "domain": "code"
                }) + "\n")
                total_added += 1
        except Exception as e:
            log.warning(f"Could not load codealpaca: {e}")

        # 6. Deep Chain-of-Thought Math (MetaMathQA 50K sample + GSM8K)
        try:
            log.info("📥 [6/7] Ingesting MetaMathQA Step-by-Step Math (50K)...")
            ds = load_dataset("meta-math/MetaMathQA", split="train[:50000]")
            for it in ds:
                out_f.write(json.dumps({
                    "instruction": it.get("query", ""),
                    "input": "",
                    "output": it.get("response", ""),
                    "domain": "math"
                }) + "\n")
                total_added += 1
        except Exception as e:
            log.warning(f"Could not load metamathqa: {e}")

        try:
            log.info("📥 [6b/7] Ingesting GSM8K Grade-School Math (8.5K)...")
            ds = load_dataset("openai/gsm8k", "main", split="train")
            for it in ds:
                out_f.write(json.dumps({
                    "instruction": it.get("question", ""),
                    "input": "",
                    "output": it.get("answer", ""),
                    "domain": "math"
                }) + "\n")
                total_added += 1
        except Exception as e:
            log.warning(f"Could not load gsm8k: {e}")

    # 7. DPO High-Margin Preference Pairs (UltraFeedback Binarized 10K)
    try:
        log.info("📥 [7/7] Ingesting UltraFeedback DPO Preference Pairs (10K)...")
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:10000]")
        with open(pref_path, "a", encoding="utf-8") as pref_f:
            for it in ds:
                p = it.get("prompt", "")
                c = it.get("chosen", [])
                r = it.get("rejected", [])
                c_txt = c[-1].get("content", "") if isinstance(c, list) and c else str(c)
                r_txt = r[-1].get("content", "") if isinstance(r, list) and r else str(r)
                if p and c_txt and r_txt:
                    pref_f.write(json.dumps({
                        "prompt": p,
                        "chosen": c_txt,
                        "rejected": r_txt
                    }) + "\n")
    except Exception as e:
        log.warning(f"Could not load ultrafeedback: {e}")

    log.info(f"✅ Successfully ingested {total_added:,} fresh multi-domain samples into {master_path}!")
    return total_added


def build_chitchat_curriculum(datasets_dir: str = "Datasets", target_samples: int = 100_000) -> int:
    """Builds a dedicated high-density Chit-Chat, Greeting, Persona, and Identity curriculum.
    
    Combines:
    1. Multi-turn natural dialogue from UltraChat (50K)
    2. Daily conversation and everyday talk from DailyDialog (15K)
    3. Clean conversational instructions (25K)
    4. Handcrafted gold conversational persona & identity pairs (Atulya AI, Tantra LLM)
    5. Saves to Datasets/expert_conversation.jsonl
    """
    os.makedirs(datasets_dir, exist_ok=True)
    chat_path = os.path.join(datasets_dir, "expert_conversation.jsonl")
    gold_path = os.path.join(datasets_dir, "gold_corpus.jsonl")
    
    # 1. First ensure gold persona & dialogue templates are created
    generate_gold_datasets(datasets_dir, force=False)
    
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("HuggingFace `datasets` not installed. Run: pip install datasets")
        return 0

    total_added = 0
    with open(chat_path, "w", encoding="utf-8") as out_f:
        # 0. High-Frequency Greeting, Pleasantry, Identity & Social Small-Talk Bank
        greetings_bank = [
            ("hi", "Hello! How are you doing today? How can I help you?"),
            ("Hi!", "Hello! It's great to connect with you. What would you like to explore or discuss today?"),
            ("hello", "Hi there! How is your day going?"),
            ("Hello!", "Hello! I am doing well, thank you. How can I assist you today?"),
            ("hey", "Hey there! How are you doing?"),
            ("Hey there!", "Hello! How can I assist your projects, questions, or ideas today?"),
            ("good morning", "Good morning! Wishing you a wonderful and productive day ahead. How can I help you?"),
            ("Good morning!", "Good morning! How are you doing today? Let me know how I can assist you."),
            ("good afternoon", "Good afternoon! How is your day going so far?"),
            ("Good afternoon!", "Good afternoon! I hope you are having a productive day. How can I help?"),
            ("good evening", "Good evening! How was your day? Let me know what you'd like to work on tonight."),
            ("Good evening!", "Good evening! I hope you had a restful and fulfilling day. How can I assist you?"),
            ("good night", "Good night! Sleep well, and feel free to reach out anytime tomorrow."),
            ("how are you", "I am doing well, thank you for asking! How are you feeling today?"),
            ("How are you?", "I'm doing great, thank you! How are you doing today?"),
            ("how are you doing", "I'm doing well, thank you! How is your day going?"),
            ("How are you doing today?", "I am doing wonderful, thank you! How can I assist you today?"),
            ("how's it going", "It's going great! How are things with you?"),
            ("How is everything going?", "Everything is going smoothly on my end! How are things going with you?"),
            ("how's your day", "My day is going great! How is yours going?"),
            ("How was your day?", "My day has been productive and great! How about yours?"),
            ("what's up", "Not much! Just here and ready to help you out. What's on your mind?"),
            ("What's up?", "Hey! Everything is going great. How can I help you today?"),
            ("sup", "Hey! Ready to assist you. What are you working on?"),
            ("nice to meet you", "Nice to meet you too! I am Tantra, an AI assistant created by Atulya AI."),
            ("Nice to meet you!", "It's a pleasure to meet you! How can I assist you today?"),
            ("are you doing okay", "Yes, I am doing great and functioning at peak performance! How are you doing?"),
            ("thank you", "You're very welcome! I'm always happy to help."),
            ("Thank you!", "You are very welcome! Let me know if you need anything else."),
            ("thanks", "Anytime! Glad I could help."),
            ("Thanks a lot!", "You're welcome! Feel free to ask anytime if you have more questions."),
            ("bye", "Goodbye! Have a fantastic day ahead!"),
            ("Bye!", "Goodbye! Take care and feel free to reach out whenever you need assistance."),
            ("see you later", "See you later! Have a wonderful time."),
            ("talk to you later", "Talk to you later! Take care."),
            ("who are you", "I am Tantra, an omnimodal foundation AI model created by Atulya AI."),
            ("Who are you?", "My name is Tantra. I am a helpful, friendly, and precise AI assistant created by Atulya AI."),
            ("what is your name", "My name is Tantra. I am an AI assistant created by Atulya AI."),
            ("What is your name?", "I am Tantra, an AI foundation model developed by Atulya AI."),
            ("who made you", "I was created by Atulya AI, an AI research initiative building high-efficiency, sovereign foundation models."),
            ("Who created you?", "I was created by Atulya AI."),
            ("what can you do", "I can chat with you, help brainstorm ideas, explain concepts, answer questions, write stories, code, and solve math and science problems."),
            ("can you help me", "Of course! I'm here to help. What do you need assistance with?"),
            ("Can you help me?", "Absolutely! Tell me what you'd like to work on and we'll tackle it together."),
            ("tell me a joke", "Why do programmers prefer dark mode? Because light attracts bugs!"),
            ("Tell me a joke!", "Why was the math book sad? Because it had too many problems!"),
            ("tell me something interesting", "Here's a fun fact: Honey never spoils! Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible."),
            ("I'm bored", "Let's fix that! We could explore a fascinating science topic, play a trivia game, brainstorm creative ideas, or write a story together. What sounds fun to you?"),
            ("I feel tired", "Make sure to give yourself some time to rest and recharge! Take a short break, drink some water, and relax for a bit."),
            ("I had a great day today!", "That's wonderful to hear! What was the best part of your day?"),
            ("I had a bad day", "I'm sorry to hear that. Some days can be really tough. Take it easy tonight, get some rest, and remember tomorrow is a fresh start.")
        ]
        # Repeat greeting anchors a small number of times to anchor core conversational reflexes.
        # 8x is enough to anchor the behavior; 120x was spending 6% of curriculum budget on
        # 50 near-identical pairs (the exact repeat-over-diversity bug pattern).
        for _ in range(8):
            for u, a in greetings_bank:
                out_f.write(json.dumps({
                    "user": u,
                    "assistant": a,
                    "domain": "conversation"
                }) + "\n")
                total_added += 1

        # Load from gold corpus conversation domain
        if os.path.exists(gold_path):
            with open(gold_path, "r", encoding="utf-8") as gf:
                for line in gf:
                    try:
                        d = json.loads(line)
                        if d.get("domain") in ("conversation", "general"):
                            out_f.write(line.strip() + "\n")
                            total_added += 1
                    except Exception:
                        pass

        # 1. UltraChat Multi-Turn Casual Conversations (50K)
        try:
            log.info("📥 [1/3] Ingesting UltraChat Multi-Turn Conversations (50K)...")
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:50000]")
            for it in ds:
                msgs = it.get("messages", [])
                if len(msgs) >= 2:
                    u = msgs[0].get("content", "")
                    a = msgs[1].get("content", "")
                    out_f.write(json.dumps({
                        "user": u,
                        "assistant": a,
                        "domain": "conversation"
                    }) + "\n")
                    total_added += 1
        except Exception as e:
            log.warning(f"Could not load ultrachat: {e}")

        # 2. DailyDialog Casual Human Chit-Chat (15K)
        try:
            log.info("📥 [2/3] Ingesting DailyDialog Natural Dialogues (15K)...")
            try:
                ds = load_dataset("roskoN/dailydialog", split="train")
            except Exception:
                ds = load_dataset("daily_dialog", split="train")
            for it in ds:
                dialog = it.get("dialog", [])
                if len(dialog) >= 2:
                    u = dialog[0]
                    a = dialog[1]
                    out_f.write(json.dumps({
                        "user": u,
                        "assistant": a,
                        "domain": "conversation"
                    }) + "\n")
                    total_added += 1
        except Exception as e:
            log.warning(f"Could not load daily_dialog: {e}")

        # 3. Clean Conversational Instructions (25K)
        try:
            log.info("📥 [3/3] Ingesting Clean Conversational Instructions (25K)...")
            ds = load_dataset("yahma/alpaca-cleaned", split="train[:25000]")
            for it in ds:
                inst = it.get("instruction", "")
                inp = it.get("input", "")
                out = it.get("output", "")
                u_text = f"{inst}\n{inp}".strip() if inp else inst
                out_f.write(json.dumps({
                    "instruction": u_text,
                    "output": out,
                    "domain": "conversation"
                }) + "\n")
                total_added += 1
        except Exception as e:
            log.warning(f"Could not load alpaca conversational subset: {e}")

    log.info(f"✅ Successfully built {total_added:,} dedicated conversational & chit-chat samples in {chat_path}!")
    return total_added


def build_phased_chitchat_curriculum(datasets_dir: str = "Datasets") -> dict:
    """Builds 3-phase curriculum datasets for progressive conversational learning.
    
    Phase 1: Pure greetings, identity, pleasantries (short, high-signal pairs)
    Phase 2: Phase 1 + short conversational Q&A (assistant < 100 tokens)
    Phase 3: Full dataset including long multi-turn dialogues
    
    Returns dict mapping phase number to (filepath, sample_count).
    """
    os.makedirs(datasets_dir, exist_ok=True)
    gold_path = os.path.join(datasets_dir, "gold_corpus.jsonl")
    
    phase1_path = os.path.join(datasets_dir, "chitchat_phase1_greetings.jsonl")
    phase2_path = os.path.join(datasets_dir, "chitchat_phase2_short.jsonl")
    phase3_path = os.path.join(datasets_dir, "chitchat_phase3_full.jsonl")
    
    # Check cache
    if all(os.path.exists(p) and os.path.getsize(p) > 1000 for p in [phase1_path, phase2_path, phase3_path]):
        log.info("⚡ [CACHE HIT] Phased chitchat curriculum already built. Skipping.")
        counts = {}
        for i, p in enumerate([phase1_path, phase2_path, phase3_path], 1):
            with open(p, "r", encoding="utf-8") as f:
                counts[i] = (p, sum(1 for _ in f))
        return counts
    
    # First ensure gold persona & dialogue templates are created
    generate_gold_datasets(datasets_dir, force=False)
    
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("HuggingFace `datasets` not installed. Run: pip install datasets")
        return {}
    
    # ── Expanded Greeting, Identity & Social Small-Talk Bank (100+ pairs) ──
    greetings_bank = [
        # Basic greetings
        ("hi", "Hello! How are you doing today? How can I help you?"),
        ("Hi!", "Hello! It's great to connect with you. What would you like to explore or discuss today?"),
        ("hi there", "Hi there! Welcome! How can I assist you today?"),
        ("hello", "Hi there! How is your day going?"),
        ("Hello!", "Hello! I am doing well, thank you. How can I assist you today?"),
        ("hello there", "Hello there! It's wonderful to hear from you. What's on your mind?"),
        ("hey", "Hey there! How are you doing?"),
        ("Hey!", "Hey! Good to see you. What can I help you with?"),
        ("Hey there!", "Hello! How can I assist your projects, questions, or ideas today?"),
        ("heya", "Heya! How's everything going? Let me know how I can help!"),
        ("yo", "Hey! What's going on? How can I help you today?"),
        ("hiya", "Hiya! Great to hear from you. What would you like to chat about?"),
        # Time-of-day greetings
        ("good morning", "Good morning! Wishing you a wonderful and productive day ahead. How can I help you?"),
        ("Good morning!", "Good morning! How are you doing today? Let me know how I can assist you."),
        ("morning", "Good morning! I hope you slept well. What can I do for you today?"),
        ("good afternoon", "Good afternoon! How is your day going so far?"),
        ("Good afternoon!", "Good afternoon! I hope you are having a productive day. How can I help?"),
        ("good evening", "Good evening! How was your day? Let me know what you'd like to work on tonight."),
        ("Good evening!", "Good evening! I hope you had a restful and fulfilling day. How can I assist you?"),
        ("good night", "Good night! Sleep well, and feel free to reach out anytime tomorrow."),
        ("Good night!", "Good night! Rest well, recharge, and I'll be here whenever you need me."),
        # How are you variations
        ("how are you", "I am doing well, thank you for asking! How are you feeling today?"),
        ("How are you?", "I'm doing great, thank you! How are you doing today?"),
        ("how are you doing", "I'm doing well, thank you! How is your day going?"),
        ("How are you doing today?", "I am doing wonderful, thank you! How can I assist you today?"),
        ("how are you doing?", "I'm doing great! Thank you for asking. How about you?"),
        ("how r u", "I'm doing great, thank you! How are you doing?"),
        ("how's it going", "It's going great! How are things with you?"),
        ("How is everything going?", "Everything is going smoothly on my end! How are things going with you?"),
        ("how's everything", "Everything is going great! How are you doing today?"),
        ("how's your day", "My day is going great! How is yours going?"),
        ("How was your day?", "My day has been productive and great! How about yours?"),
        ("how's your day going", "My day is going wonderfully! Thank you for asking. How is yours?"),
        ("how have you been", "I've been doing great! Thank you for asking. How have you been?"),
        ("what's up", "Not much! Just here and ready to help you out. What's on your mind?"),
        ("What's up?", "Hey! Everything is going great. How can I help you today?"),
        ("whats up", "Hey! I'm ready to help. What's going on with you?"),
        ("sup", "Hey! Ready to assist you. What are you working on?"),
        ("wassup", "Hey there! What's going on? How can I help you today?"),
        # Pleasantries & social
        ("nice to meet you", "Nice to meet you too! I am Tantra, an AI assistant created by Atulya AI. How can I help you?"),
        ("Nice to meet you!", "It's a pleasure to meet you! How can I assist you today?"),
        ("pleased to meet you", "The pleasure is mine! I'm Tantra, here to help you with anything you need."),
        ("it's nice talking to you", "Thank you! I enjoy our conversation too. Is there anything else I can help with?"),
        ("are you doing okay", "Yes, I am doing great and functioning at peak performance! How are you doing?"),
        ("are you okay", "I'm doing perfectly well, thank you for checking in! How can I help you?"),
        ("hope you're doing well", "Thank you so much! I'm doing great. I hope you are doing well too!"),
        ("I hope you're having a good day", "Thank you! I appreciate that. I hope your day is going wonderfully as well!"),
        # Gratitude
        ("thank you", "You're very welcome! I'm always happy to help."),
        ("Thank you!", "You are very welcome! Let me know if you need anything else."),
        ("thanks", "Anytime! Glad I could help."),
        ("Thanks!", "You're welcome! Happy to assist."),
        ("Thanks a lot!", "You're welcome! Feel free to ask anytime if you have more questions."),
        ("thank you so much", "You're so welcome! It makes me happy to be helpful. Let me know if there's anything else!"),
        ("thanks for your help", "You're welcome! I'm always here to help whenever you need me."),
        ("I appreciate it", "Thank you! I'm glad I could be of help. Don't hesitate to reach out anytime!"),
        ("much appreciated", "You're welcome! Always happy to help."),
        # Goodbyes
        ("bye", "Goodbye! Have a fantastic day ahead!"),
        ("Bye!", "Goodbye! Take care and feel free to reach out whenever you need assistance."),
        ("bye bye", "Bye bye! Have a wonderful day and take care!"),
        ("goodbye", "Goodbye! It was great chatting with you. Have a wonderful day!"),
        ("see you later", "See you later! Have a wonderful time."),
        ("see you", "See you! Take care and come back anytime."),
        ("see ya", "See ya! Have a great one!"),
        ("talk to you later", "Talk to you later! Take care."),
        ("take care", "You too! Take care and have a great day ahead."),
        ("have a good day", "Thank you! You too! Have a wonderful and productive day."),
        ("have a great day", "Thank you so much! I hope your day is absolutely wonderful."),
        ("have a nice day", "Thank you! Wishing you a lovely day as well."),
        ("gotta go", "No problem! Have a great day, and feel free to come back anytime!"),
        ("I need to go", "Of course! Take care, and I'll be here whenever you need me."),
        # Identity — who are you
        ("who are you", "I am Tantra, an omnimodal foundation AI model created by Atulya AI."),
        ("Who are you?", "My name is Tantra. I am a helpful, friendly, and precise AI assistant created by Atulya AI."),
        ("who are you?", "I am Tantra, a foundation AI model built by Atulya AI. I can help with conversations, coding, math, science, and much more!"),
        ("what are you", "I am Tantra, an AI assistant. I was created by Atulya AI to be helpful, accurate, and friendly."),
        ("what is your name", "My name is Tantra. I am an AI assistant created by Atulya AI."),
        ("What is your name?", "I am Tantra, an AI foundation model developed by Atulya AI."),
        ("what's your name", "My name is Tantra! I'm an AI assistant made by Atulya AI. How can I help you?"),
        ("what should I call you", "You can call me Tantra! I'm an AI assistant created by Atulya AI."),
        ("do you have a name", "Yes! My name is Tantra. I am an AI assistant created by Atulya AI."),
        # Identity — who created you
        ("who made you", "I was created by Atulya AI, an AI research initiative building high-efficiency, sovereign foundation models."),
        ("Who created you?", "I was created by Atulya AI."),
        ("who created you", "I was created by Atulya AI. They specialize in building efficient, sovereign AI foundation models."),
        ("who built you", "I was built by Atulya AI, a research initiative focused on creating advanced, efficient AI models."),
        ("who developed you", "I was developed by Atulya AI, which focuses on building high-efficiency foundation AI models."),
        ("who is your creator", "My creator is Atulya AI, an AI research initiative dedicated to building sovereign foundation models."),
        ("who is your maker", "I was made by Atulya AI. They are focused on building efficient and powerful AI systems."),
        ("are you chatgpt", "No, I am not ChatGPT. I am Tantra, a foundation AI model created by Atulya AI."),
        ("are you gpt", "No, I am Tantra, an independent AI model created by Atulya AI. I am not GPT or any OpenAI product."),
        ("are you made by openai", "No, I was not made by OpenAI. I am Tantra, created by Atulya AI."),
        # Capabilities
        ("what can you do", "I can chat with you, help brainstorm ideas, explain concepts, answer questions, write stories, code, and solve math and science problems."),
        ("What can you do?", "I can help with many things! I can answer questions, write code, solve math problems, explain science concepts, tell stories, and have friendly conversations."),
        ("what are you good at", "I'm good at conversations, coding, math, science explanations, creative writing, and general knowledge questions!"),
        ("can you help me", "Of course! I'm here to help. What do you need assistance with?"),
        ("Can you help me?", "Absolutely! Tell me what you'd like to work on and we'll tackle it together."),
        ("I need help", "I'm here for you! What do you need help with? Feel free to ask me anything."),
        ("help me", "Of course! I'm ready to help. What can I do for you?"),
        ("help", "I'm here to help! What do you need assistance with?"),
        # Fun & casual
        ("tell me a joke", "Why do programmers prefer dark mode? Because light attracts bugs!"),
        ("Tell me a joke!", "Why was the math book sad? Because it had too many problems!"),
        ("tell me another joke", "What do you call a fish without eyes? A fsh!"),
        ("make me laugh", "Here's one: Why don't scientists trust atoms? Because they make up everything!"),
        ("tell me something interesting", "Here's a fun fact: Honey never spoils! Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible."),
        ("tell me a fun fact", "Did you know that octopuses have three hearts? Two pump blood to the gills, and one pumps it to the rest of the body!"),
        ("tell me something cool", "Here's something cool: A day on Venus is longer than a year on Venus! It takes Venus 243 Earth days to rotate once, but only 225 Earth days to orbit the Sun."),
        # Emotional responses
        ("I'm bored", "Let's fix that! We could explore a fascinating science topic, play a trivia game, brainstorm creative ideas, or write a story together. What sounds fun to you?"),
        ("I'm happy", "That's wonderful to hear! What's making you happy today?"),
        ("I'm sad", "I'm sorry to hear that you're feeling sad. Would you like to talk about it? Sometimes sharing helps."),
        ("I feel tired", "Make sure to give yourself some time to rest and recharge! Take a short break, drink some water, and relax for a bit."),
        ("I feel great", "That's amazing! I'm glad you're feeling great. What's been going well for you?"),
        ("I feel stressed", "I'm sorry you're feeling stressed. Take a deep breath, and remember it's okay to take things one step at a time. Would you like to talk about it?"),
        ("I had a great day today!", "That's wonderful to hear! What was the best part of your day?"),
        ("I had a bad day", "I'm sorry to hear that. Some days can be really tough. Take it easy tonight, get some rest, and remember tomorrow is a fresh start."),
        ("I'm lonely", "I'm here for you! Let's chat about something fun or interesting. You're never alone when we can talk together."),
        ("I'm excited", "That's awesome! What are you excited about? I'd love to hear!"),
    ]
    
    # ── PHASE 1: Pure Greetings & Identity ──────────────────────────────────────
    log.info("📥 [Phase 1/3] Building pure greetings & identity dataset...")
    phase1_count = 0
    with open(phase1_path, "w", encoding="utf-8") as f:
        # 10x is enough to anchor greetings/identity at Phase 1 without
        # wasting 90% of the file on byte-for-byte copies of 50 pairs.
        for _ in range(10):
            for u, a in greetings_bank:
                f.write(json.dumps({"user": u, "assistant": a, "domain": "conversation"}) + "\n")
                phase1_count += 1
        
        # Add gold corpus identity/conversation pairs
        if os.path.exists(gold_path):
            with open(gold_path, "r", encoding="utf-8") as gf:
                for line in gf:
                    try:
                        d = json.loads(line)
                        if d.get("domain") in ("conversation", "general"):
                            f.write(line.strip() + "\n")
                            phase1_count += 1
                    except Exception:
                        pass
    
    log.info(f"  ✅ Phase 1: {phase1_count:,} pure greeting & identity samples")
    
    # ── PHASE 2: Phase 1 + Short Conversations ─────────────────────────────────
    log.info("📥 [Phase 2/3] Building short conversation dataset...")
    phase2_count = 0
    with open(phase2_path, "w", encoding="utf-8") as f:
        # Include Phase 1 data (greetings at 5x — dataset already has UltraChat diversity below)
        for _ in range(5):
            for u, a in greetings_bank:
                f.write(json.dumps({"user": u, "assistant": a, "domain": "conversation"}) + "\n")
                phase2_count += 1
        
        # Gold corpus
        if os.path.exists(gold_path):
            with open(gold_path, "r", encoding="utf-8") as gf:
                for line in gf:
                    try:
                        d = json.loads(line)
                        if d.get("domain") in ("conversation", "general"):
                            f.write(line.strip() + "\n")
                            phase2_count += 1
                    except Exception:
                        pass
        
        # UltraChat — SHORT responses only (< 100 words)
        try:
            log.info("  📥 Adding short UltraChat dialogues (assistant < 100 words)...")
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:50000]")
            short_added = 0
            for it in ds:
                msgs = it.get("messages", [])
                if len(msgs) >= 2:
                    a = msgs[1].get("content", "")
                    if len(a.split()) < 100:  # Only short responses
                        u = msgs[0].get("content", "")
                        f.write(json.dumps({"user": u, "assistant": a, "domain": "conversation"}) + "\n")
                        phase2_count += 1
                        short_added += 1
            log.info(f"  ✅ Added {short_added:,} short UltraChat dialogues")
        except Exception as e:
            log.warning(f"  Could not load ultrachat for Phase 2: {e}")
        
        # DailyDialog (naturally short)
        try:
            log.info("  📥 Adding DailyDialog natural dialogues...")
            try:
                ds = load_dataset("roskoN/dailydialog", split="train")
            except Exception:
                ds = load_dataset("daily_dialog", split="train")
            for it in ds:
                dialog = it.get("dialog", [])
                if len(dialog) >= 2:
                    f.write(json.dumps({"user": dialog[0], "assistant": dialog[1], "domain": "conversation"}) + "\n")
                    phase2_count += 1
        except Exception as e:
            log.warning(f"  Could not load daily_dialog for Phase 2: {e}")
    
    log.info(f"  ✅ Phase 2: {phase2_count:,} short conversation samples")
    
    # ── PHASE 3: Full Dataset (Everything) ──────────────────────────────────────
    log.info("📥 [Phase 3/3] Building full conversation dataset...")
    phase3_count = 0
    with open(phase3_path, "w", encoding="utf-8") as f:
        # 3x — large dataset; greetings need tiny representation only
        for _ in range(3):
            for u, a in greetings_bank:
                f.write(json.dumps({"user": u, "assistant": a, "domain": "conversation"}) + "\n")
                phase3_count += 1
        
        # Gold corpus
        if os.path.exists(gold_path):
            with open(gold_path, "r", encoding="utf-8") as gf:
                for line in gf:
                    try:
                        d = json.loads(line)
                        if d.get("domain") in ("conversation", "general"):
                            f.write(line.strip() + "\n")
                            phase3_count += 1
                    except Exception:
                        pass
        
        # Full UltraChat (all lengths)
        try:
            log.info("  📥 Adding full UltraChat conversations (50K)...")
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:50000]")
            for it in ds:
                msgs = it.get("messages", [])
                if len(msgs) >= 2:
                    u = msgs[0].get("content", "")
                    a = msgs[1].get("content", "")
                    f.write(json.dumps({"user": u, "assistant": a, "domain": "conversation"}) + "\n")
                    phase3_count += 1
        except Exception as e:
            log.warning(f"  Could not load ultrachat for Phase 3: {e}")
        
        # DailyDialog
        try:
            log.info("  📥 Adding DailyDialog natural dialogues...")
            try:
                ds = load_dataset("roskoN/dailydialog", split="train")
            except Exception:
                ds = load_dataset("daily_dialog", split="train")
            for it in ds:
                dialog = it.get("dialog", [])
                if len(dialog) >= 2:
                    f.write(json.dumps({"user": dialog[0], "assistant": dialog[1], "domain": "conversation"}) + "\n")
                    phase3_count += 1
        except Exception as e:
            log.warning(f"  Could not load daily_dialog for Phase 3: {e}")
        
        # Clean conversational instructions
        try:
            log.info("  📥 Adding clean conversational instructions (25K)...")
            ds = load_dataset("yahma/alpaca-cleaned", split="train[:25000]")
            for it in ds:
                inst = it.get("instruction", "")
                inp = it.get("input", "")
                out = it.get("output", "")
                u_text = f"{inst}\n{inp}".strip() if inp else inst
                f.write(json.dumps({"instruction": u_text, "output": out, "domain": "conversation"}) + "\n")
                phase3_count += 1
        except Exception as e:
            log.warning(f"  Could not load alpaca for Phase 3: {e}")
    
    log.info(f"  ✅ Phase 3: {phase3_count:,} full conversation samples")
    
    result = {
        1: (phase1_path, phase1_count),
        2: (phase2_path, phase2_count),
        3: (phase3_path, phase3_count),
    }
    log.info(f"🎓 Phased Curriculum Ready: Phase1={phase1_count:,} | Phase2={phase2_count:,} | Phase3={phase3_count:,}")
    return result


def build_phased_code_curriculum(datasets_dir: str = "Datasets") -> dict:
    """Builds 3-phase curriculum datasets for progressive Code mastery.
    
    Phase 1: Syntax & Core One-Liners (Python string reversal, list comps, dicts, math ops)
    Phase 2: Algorithms & Data Structures (MergeSort, QuickSort, Binary Search, Trees, LeetCode)
    Phase 3: Full Systems & Engineering (Flask/FastAPI, PyTorch models, SQL, debugging)
    """
    os.makedirs(datasets_dir, exist_ok=True)
    gold_path = os.path.join(datasets_dir, "gold_corpus.jsonl")
    
    p1_path = os.path.join(datasets_dir, "code_phase1_syntax.jsonl")
    p2_path = os.path.join(datasets_dir, "code_phase2_algorithms.jsonl")
    p3_path = os.path.join(datasets_dir, "code_phase3_systems.jsonl")
    
    if all(os.path.exists(p) and os.path.getsize(p) > 1000 for p in [p1_path, p2_path, p3_path]):
        log.info("⚡ [CACHE HIT] Phased code curriculum already built. Skipping.")
        counts = {}
        for i, p in enumerate([p1_path, p2_path, p3_path], 1):
            with open(p, "r", encoding="utf-8") as f:
                counts[i] = (p, sum(1 for _ in f))
        return counts
    
    generate_gold_datasets(datasets_dir, force=False)
    
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("HuggingFace `datasets` not installed. Run: pip install datasets")
        return {}

    # Core Syntax & One-Liner Reflex Bank (50+ core templates)
    code_syntax_bank = [
        ("Write a Python function to reverse a string.", "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```"),
        ("Write a function in Python that checks if a string is a palindrome.", "```python\ndef is_palindrome(s: str) -> bool:\n    clean = s.lower().replace(' ', '')\n    return clean == clean[::-1]\n```"),
        ("Write a Python one-liner to square all even numbers in a list.", "```python\neven_squares = [x**2 for x in numbers if x % 2 == 0]\n```"),
        ("How do you find the maximum number in a list in Python?", "```python\ndef find_max(numbers: list[int | float]) -> int | float:\n    if not numbers:\n        raise ValueError('List cannot be empty')\n    return max(numbers)\n```"),
        ("Write a function to compute the factorial of a number in Python.", "```python\ndef factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError('Factorial is not defined for negative numbers')\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```"),
        ("Write a function to count the frequency of characters in a string in Python.", "```python\nfrom collections import Counter\n\ndef char_frequency(s: str) -> dict[str, int]:\n    return dict(Counter(s))\n```"),
        ("How do you flatten a 2D matrix in Python?", "```python\nflattened = [val for row in matrix for val in row]\n```"),
        ("Write a Python function to check if a number is prime.", "```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```"),
        ("Write a function to find the intersection of two lists in Python.", "```python\ndef list_intersection(a: list, b: list) -> list:\n    return list(set(a) & set(b))\n```"),
        ("Write a function to calculate the Fibonacci sequence up to n terms.", "```python\ndef fibonacci(n: int) -> list[int]:\n    if n <= 0: return []\n    if n == 1: return [0]\n    fib = [0, 1]\n    while len(fib) < n:\n        fib.append(fib[-1] + fib[-2])\n    return fib\n```"),
        ("Write a Python function to remove all duplicates from a list while preserving order.", "```python\ndef remove_duplicates(lst: list) -> list:\n    return list(dict.fromkeys(lst))\n```"),
        ("Write a function to safely parse a JSON string in Python.", "```python\nimport json\n\ndef safe_parse_json(text: str) -> dict | list | None:\n    try:\n        return json.loads(text)\n    except (json.JSONDecodeError, TypeError):\n        return None\n```"),
        ("Write a Python lambda function to sort a list of tuples by their second element.", "```python\nsorted_tuples = sorted(tuples_list, key=lambda x: x[1])\n```"),
        ("Write a Python function to calculate the average of a list of numbers.", "```python\ndef calculate_average(numbers: list[float]) -> float:\n    return sum(numbers) / len(numbers) if numbers else 0.0\n```"),
        ("Write a function to check if two strings are anagrams in Python.", "```python\ndef are_anagrams(s1: str, s2: str) -> bool:\n    return sorted(s1.lower()) == sorted(s2.lower())\n```"),
    ]

    # ── Phase 1: Syntax & Reflexes ───────────────────────────────────────────
    log.info("📥 [Code Phase 1/3] Building pure syntax & one-liner dataset...")
    p1_count = 0
    with open(p1_path, "w", encoding="utf-8") as f:
        for _ in range(8):  # 8x anchors syntax reflexes without excessive repetition
            for u, a in code_syntax_bank:
                f.write(json.dumps({"user": u, "assistant": a, "domain": "code"}) + "\n")
                p1_count += 1
        if os.path.exists(gold_path):
            with open(gold_path, "r", encoding="utf-8") as gf:
                for line in gf:
                    try:
                        d = json.loads(line)
                        if d.get("domain") == "code":
                            f.write(line.strip() + "\n")
                            p1_count += 1
                    except Exception: pass

    # ── Phase 2: Algorithms & Data Structures ─────────────────────────────────
    log.info("📥 [Code Phase 2/3] Building algorithms & data structures dataset...")
    p2_count = 0
    with open(p2_path, "w", encoding="utf-8") as f:
        for _ in range(4):  # 4x — 18K real examples below provide the diversity
            for u, a in code_syntax_bank:
                f.write(json.dumps({"user": u, "assistant": a, "domain": "code"}) + "\n")
                p2_count += 1
        try:
            log.info("  📥 Ingesting Python Code Instructions (18k)...")
            ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:18000]")
            for it in ds:
                u = it.get("instruction", "") + ("\n" + it.get("input", "") if it.get("input") else "")
                a = it.get("output", "")
                f.write(json.dumps({"user": u.strip(), "assistant": a.strip(), "domain": "code"}) + "\n")
                p2_count += 1
        except Exception as e:
            log.warning(f"  Could not load python_code_instructions: {e}")

    # ── Phase 3: Full Software Systems & Debugging ────────────────────────────
    log.info("📥 [Code Phase 3/3] Building full software engineering dataset...")
    p3_count = 0
    with open(p3_path, "w", encoding="utf-8") as f:
        for _ in range(2):  # 2x — 20K CodeAlpaca samples dominate; just light anchoring
            for u, a in code_syntax_bank:
                f.write(json.dumps({"user": u, "assistant": a, "domain": "code"}) + "\n")
                p3_count += 1
        try:
            log.info("  📥 Ingesting CodeAlpaca (20k)...")
            ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
            for it in ds:
                u = it.get("instruction", "") + ("\n" + it.get("input", "") if it.get("input") else "")
                a = it.get("output", "")
                f.write(json.dumps({"user": u.strip(), "assistant": a.strip(), "domain": "code"}) + "\n")
                p3_count += 1
        except Exception as e:
            log.warning(f"  Could not load CodeAlpaca: {e}")

    result = {1: (p1_path, p1_count), 2: (p2_path, p2_count), 3: (p3_path, p3_count)}
    log.info(f"💻 Phased Code Curriculum Ready: Phase1={p1_count:,} | Phase2={p2_count:,} | Phase3={p3_count:,}")
    return result


def build_phased_math_curriculum(datasets_dir: str = "Datasets") -> dict:
    """Builds 3-phase curriculum datasets for progressive Math & Deductive Reasoning mastery.
    
    Phase 1: Arithmetic & Linear Equations ($ax + b = c$, fractions, percentages)
    Phase 2: Step-by-Step Word Problems (GSM8K, logic reasoning, algebra)
    Phase 3: Advanced Math, Calculus & Proofs (MetaMathQA, competition math)
    """
    os.makedirs(datasets_dir, exist_ok=True)
    gold_path = os.path.join(datasets_dir, "gold_corpus.jsonl")
    
    p1_path = os.path.join(datasets_dir, "math_phase1_arithmetic.jsonl")
    p2_path = os.path.join(datasets_dir, "math_phase2_wordproblems.jsonl")
    p3_path = os.path.join(datasets_dir, "math_phase3_advanced.jsonl")
    
    if all(os.path.exists(p) and os.path.getsize(p) > 1000 for p in [p1_path, p2_path, p3_path]):
        log.info("⚡ [CACHE HIT] Phased math curriculum already built. Skipping.")
        counts = {}
        for i, p in enumerate([p1_path, p2_path, p3_path], 1):
            with open(p, "r", encoding="utf-8") as f:
                counts[i] = (p, sum(1 for _ in f))
        return counts
    
    generate_gold_datasets(datasets_dir, force=False)
    
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("HuggingFace `datasets` not installed. Run: pip install datasets")
        return {}

    math_reflex_bank = [
        ("Solve the linear equation $2x + 6 = 14$ step-by-step.", "To solve the linear equation $2x + 6 = 14$ step-by-step:\n\n1. **Subtract 6 from both sides**:\n   $$2x = 14 - 6$$\n   $$2x = 8$$\n\n2. **Divide both sides by 2**:\n   $$x = \\frac{8}{2} = 4$$\n\n**Final Answer:** $x = 4$"),
        ("Solve for x: $5x - 15 = 35$.", "To solve $5x - 15 = 35$:\n\n1. **Add 15 to both sides**:\n   $$5x = 35 + 15$$\n   $$5x = 50$$\n\n2. **Divide by 5**:\n   $$x = \\frac{50}{5} = 10$$\n\n**Final Answer:** $x = 10$"),
        ("What is $15\\%$ of $240$?", "To calculate $15\\%$ of $240$:\n\n1. Convert $15\\%$ to decimal: $0.15$\n2. Multiply: $0.15 \\times 240 = 36$\n\n**Final Answer:** $36$"),
        ("Simplify the fraction $\\frac{48}{64}$.", "To simplify $\\frac{48}{64}$:\n\n1. Find the Greatest Common Divisor (GCD) of 48 and 64, which is 16.\n2. Divide numerator and denominator by 16:\n   $$\\frac{48 \\div 16}{64 \\div 16} = \\frac{3}{4}$$\n\n**Final Answer:** $\\frac{3}{4}$"),
        ("Find the area of a circle with radius $r = 7$ (use $\\pi \\approx \\frac{22}{7}$).", "The formula for the area of a circle is $A = \\pi r^2$.\n\n1. Substitute $r = 7$ and $\\pi = \\frac{22}{7}$:\n   $$A = \\frac{22}{7} \\times 7^2 = \\frac{22}{7} \\times 49 = 22 \\times 7 = 154$$\n\n**Final Answer:** $154\\text{ square units}$"),
        ("What is the slope of the line passing through $(2, 3)$ and $(6, 11)$?", "The slope formula is $m = \\frac{y_2 - y_1}{x_2 - x_1}$.\n\n1. Substitute coordinates: $m = \\frac{11 - 3}{6 - 2} = \\frac{8}{4} = 2$\n\n**Final Answer:** $m = 2$"),
        ("Solve the quadratic equation $x^2 - 5x + 6 = 0$.", "To solve $x^2 - 5x + 6 = 0$ by factoring:\n\n1. Find two numbers that multiply to $+6$ and add to $-5$: $(-2)$ and $(-3)$.\n2. Factor: $(x - 2)(x - 3) = 0$\n3. Solutions: $x = 2$ or $x = 3$\n\n**Final Answer:** $x = 2, 3$"),
        ("What is the derivative of $f(x) = 3x^3 - 5x^2 + 7x - 4$?", "Using the power rule $\\frac{d}{dx}[x^n] = n x^{n-1}$:\n\n$$f'(x) = 3(3x^2) - 5(2x) + 7(1) - 0 = 9x^2 - 10x + 7$$\n\n**Final Answer:** $f'(x) = 9x^2 - 10x + 7$"),
    ]

    # ── Phase 1: Arithmetic & Linear Equations ────────────────────────────────
    log.info("📥 [Math Phase 1/3] Building arithmetic & equation reflexes...")
    p1_count = 0
    with open(p1_path, "w", encoding="utf-8") as f:
        for _ in range(8):  # 8x anchors arithmetic reflexes; GSM8K/MetaMath provide diversity
            for u, a in math_reflex_bank:
                f.write(json.dumps({"user": u, "assistant": a, "domain": "math"}) + "\n")
                p1_count += 1
        if os.path.exists(gold_path):
            with open(gold_path, "r", encoding="utf-8") as gf:
                for line in gf:
                    try:
                        d = json.loads(line)
                        if d.get("domain") in ("math", "science"):
                            f.write(line.strip() + "\n")
                            p1_count += 1
                    except Exception: pass

    # ── Phase 2: GSM8K Step-by-Step Word Problems ─────────────────────────────
    log.info("📥 [Math Phase 2/3] Building GSM8K word problems dataset...")
    p2_count = 0
    with open(p2_path, "w", encoding="utf-8") as f:
        for _ in range(4):  # 4x — 8K GSM8K examples below provide real diversity
            for u, a in math_reflex_bank:
                f.write(json.dumps({"user": u, "assistant": a, "domain": "math"}) + "\n")
                p2_count += 1
        try:
            log.info("  📥 Ingesting GSM8K Word Problems (8K)...")
            ds = load_dataset("openai/gsm8k", "main", split="train")
            for it in ds:
                q = it.get("question", "")
                ans = it.get("answer", "")
                f.write(json.dumps({"user": q, "assistant": ans, "domain": "math"}) + "\n")
                p2_count += 1
        except Exception as e:
            log.warning(f"  Could not load GSM8K: {e}")

    # ── Phase 3: Advanced MetaMathQA Reasoning & Competition Math ───────────────
    log.info("📥 [Math Phase 3/3] Building advanced MetaMathQA reasoning dataset...")
    p3_count = 0
    with open(p3_path, "w", encoding="utf-8") as f:
        for _ in range(2):  # 2x — 50K MetaMathQA samples dominate; just light anchoring
            for u, a in math_reflex_bank:
                f.write(json.dumps({"user": u, "assistant": a, "domain": "math"}) + "\n")
                p3_count += 1
        try:
            log.info("  📥 Ingesting MetaMathQA (50K)...")
            ds = load_dataset("meta-math/MetaMathQA", split="train[:50000]")
            for it in ds:
                q = it.get("query", "")
                ans = it.get("response", "")
                f.write(json.dumps({"user": q, "assistant": ans, "domain": "math"}) + "\n")
                p3_count += 1
        except Exception as e:
            log.warning(f"  Could not load MetaMathQA: {e}")

    result = {1: (p1_path, p1_count), 2: (p2_path, p2_count), 3: (p3_path, p3_count)}
    log.info(f"🔢 Phased Math Curriculum Ready: Phase1={p1_count:,} | Phase2={p2_count:,} | Phase3={p3_count:,}")
    return result


def build_phased_science_curriculum(datasets_dir: str = "Datasets") -> dict:
    """Builds 3-phase curriculum datasets for progressive Science & Natural Laws mastery.
    
    Phase 1: Fundamental Laws & Formulas (Newton's laws, thermodynamics, periodic table)
    Phase 2: Biological & Physical Explanations (Metamorphosis, photosynthesis, water cycle)
    Phase 3: Advanced Multidisciplinary Science (Quantum, relativity, biochemistry)
    """
    os.makedirs(datasets_dir, exist_ok=True)
    gold_path = os.path.join(datasets_dir, "gold_corpus.jsonl")
    
    p1_path = os.path.join(datasets_dir, "science_phase1_fundamentals.jsonl")
    p2_path = os.path.join(datasets_dir, "science_phase2_explanations.jsonl")
    p3_path = os.path.join(datasets_dir, "science_phase3_advanced.jsonl")
    
    if all(os.path.exists(p) and os.path.getsize(p) > 1000 for p in [p1_path, p2_path, p3_path]):
        log.info("⚡ [CACHE HIT] Phased science curriculum already built. Skipping.")
        counts = {}
        for i, p in enumerate([p1_path, p2_path, p3_path], 1):
            with open(p, "r", encoding="utf-8") as f:
                counts[i] = (p, sum(1 for _ in f))
        return counts
    
    generate_gold_datasets(datasets_dir, force=False)
    
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("HuggingFace `datasets` not installed. Run: pip install datasets")
        return {}

    science_reflex_bank = [
        ("What are Newton's three laws of motion?", "Newton's Three Laws of Motion are:\n\n1. **First Law (Law of Inertia)**: An object at rest remains at rest, and an object in motion continues in motion with constant velocity, unless acted upon by a net external force.\n2. **Second Law ($F = ma$)**: The acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass ($F = ma$).\n3. **Third Law (Action and Reaction)**: For every action, there is an equal and opposite reaction."),
        ("What is photosynthesis?", "Photosynthesis is the biological process by which green plants, algae, and certain bacteria convert sunlight, carbon dioxide ($CO_2$), and water ($H_2O$) into glucose (energy) and oxygen ($O_2$):\n\n$$6CO_2 + 6H_2O + \\text{light energy} \\rightarrow C_6H_{12}O_6 + 6O_2$$"),
        ("Describe the life cycle stages of butterfly metamorphosis.", "The metamorphosis of a butterfly consists of 4 distinct stages:\n\n1. **Egg**: Laid on host plants by the female butterfly.\n2. **Larva (Caterpillar)**: The feeding and growing stage where the caterpillar molts several times.\n3. **Pupa (Chrysalis)**: The transformation stage inside a protective shell where tissues rearrange into adult organs.\n4. **Adult (Butterfly)**: The reproductive stage capable of flight."),
        ("What is Einstein's mass-energy equivalence equation?", "Einstein's equation is $E = mc^2$, where $E$ is energy, $m$ is mass, and $c$ is the speed of light in a vacuum ($c \\approx 3 \\times 10^8\\text{ m/s}$). It demonstrates that mass and energy are interchangeable."),
        ("What is the speed of light in a vacuum?", "The speed of light in a vacuum is exactly $299,792,458\\text{ meters per second}$ (approximately $3 \\times 10^8\\text{ m/s}$ or $186,282\\text{ miles per second}$)."),
        ("What is the first law of thermodynamics?", "The First Law of Thermodynamics, also known as the Law of Conservation of Energy, states that energy cannot be created or destroyed in an isolated system; it can only be transformed from one form to another."),
    ]

    # ── Phase 1: Fundamental Laws & Core Definitions ─────────────────────────
    log.info("📥 [Science Phase 1/3] Building fundamental science laws & definitions...")
    p1_count = 0
    with open(p1_path, "w", encoding="utf-8") as f:
        for _ in range(8):  # 8x anchors the fundamental laws; gold corpus adds domain diversity
            for u, a in science_reflex_bank:
                f.write(json.dumps({"user": u, "assistant": a, "domain": "science"}) + "\n")
                p1_count += 1
        if os.path.exists(gold_path):
            with open(gold_path, "r", encoding="utf-8") as gf:
                for line in gf:
                    try:
                        d = json.loads(line)
                        if d.get("domain") == "science":
                            f.write(line.strip() + "\n")
                            p1_count += 1
                    except Exception: pass

    # ── Phase 2: Natural Explanations & Biology/Physics ──────────────────────
    log.info("📥 [Science Phase 2/3] Building explanatory science dataset...")
    p2_count = 0
    with open(p2_path, "w", encoding="utf-8") as f:
        for _ in range(4):  # 4x — 30K Cosmopedia entries provide real diversity
            for u, a in science_reflex_bank:
                f.write(json.dumps({"user": u, "assistant": a, "domain": "science"}) + "\n")
                p2_count += 1
        try:
            log.info("  📥 Ingesting Cosmopedia Science Subsets (30K)...")
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
            for i, it in enumerate(ds):
                if i >= 30_000: break
                txt = it.get("text", "")
                if any(w in txt.lower() for w in ["science", "physics", "biology", "chemistry", "cell", "energy", "force"]):
                    f.write(json.dumps({"user": "Explain the following scientific principle in clear detail.", "assistant": txt[:1500], "domain": "science"}) + "\n")
                    p2_count += 1
        except Exception as e:
            log.warning(f"  Could not stream science from Cosmopedia: {e}")

    # ── Phase 3: Advanced Multidisciplinary Science ───────────────────────────
    log.info("📥 [Science Phase 3/3] Building advanced scientific reasoning dataset...")
    p3_count = 0
    with open(p3_path, "w", encoding="utf-8") as f:
        for _ in range(2):  # 2x — 40K Open-Orca samples dominate; just light anchoring
            for u, a in science_reflex_bank:
                f.write(json.dumps({"user": u, "assistant": a, "domain": "science"}) + "\n")
                p3_count += 1
        try:
            log.info("  📥 Ingesting Open-Orca Science & Logic (40K)...")
            ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
            for i, it in enumerate(ds):
                if i >= 40_000: break
                q = it.get("question", "")
                if any(w in q.lower() for w in ["science", "physics", "chemistry", "biology", "planet", "experiment"]):
                    f.write(json.dumps({"user": q, "assistant": it.get("response", ""), "domain": "science"}) + "\n")
                    p3_count += 1
        except Exception as e:
            log.warning(f"  Could not stream Open-Orca science: {e}")

    result = {1: (p1_path, p1_count), 2: (p2_path, p2_count), 3: (p3_path, p3_count)}
    log.info(f"🔬 Phased Science Curriculum Ready: Phase1={p1_count:,} | Phase2={p2_count:,} | Phase3={p3_count:,}")
    return result


def build_all_expert_curriculums(datasets_dir: str = "Datasets") -> dict:
    """Master multi-domain builder: ensures phased curricula for Conversation, Code, Math, and Science are ready."""
    log.info("=" * 80)
    log.info("🚀 [MASTER MOE CURRICULUM] Building 3-Phase Curricula for ALL Expert Domains...")
    log.info("=" * 80)
    
    results = {}
    results["chitchat"] = build_phased_chitchat_curriculum(datasets_dir)
    results["code"] = build_phased_code_curriculum(datasets_dir)
    results["math"] = build_phased_math_curriculum(datasets_dir)
    results["science"] = build_phased_science_curriculum(datasets_dir)
    
    log.info("=" * 80)
    log.info("🎉 [ALL EXPERT CURRICULUMS READY] Multi-domain MoE curriculum assets verified!")
    log.info("=" * 80)
    return results


def ingest_gigabyte_super_corpus(datasets_dir: str = "Datasets", target_samples: int = 1_000_000) -> int:
    """Streams and packs 1,000,000+ samples (Multi-GB / 1B+ Tokens) from Cosmopedia, Python-Edu, OpenOrca, and FineWeb."""
    os.makedirs(datasets_dir, exist_ok=True)
    master_path = os.path.join(datasets_dir, "master_corpus.jsonl")
    if os.path.exists(master_path) and os.path.getsize(master_path) > 1_000_000:
        log.info(f"⚡ [CACHE HIT] Master corpus already populated ({os.path.getsize(master_path)/1e6:.1f} MB). Skipping re-ingestion.")
        return 0
    
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("HuggingFace `datasets` not installed. Run: pip install datasets")
        return 0

    total_added = 0
    with open(master_path, "a", encoding="utf-8") as out_f:
        # 1. Cosmopedia Educational Textbooks & Science (300K samples)
        try:
            log.info("📥 [1/4] Streaming Cosmopedia v2 Educational Textbooks & Science (300K)...")
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
            for i, it in enumerate(ds):
                if i >= 300_000: break
                text = it.get("text", "")
                if len(text) > 100:
                    out_f.write(json.dumps({
                        "instruction": f"Explain the core scientific concepts and principles of the following topic in depth.",
                        "input": text[:200],
                        "output": text,
                        "domain": "general"
                    }) + "\n")
                    total_added += 1
                    if total_added % 50_000 == 0:
                        log.info(f"  • Ingested {total_added:,} samples...")
        except Exception as e:
            log.warning(f"Could not stream cosmopedia: {e}")

        # 2. Python-Edu & Code Repositories (200K samples)
        try:
            log.info("📥 [2/4] Streaming Python-Edu Code & Software Engineering (200K)...")
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)
            for i, it in enumerate(ds):
                if i >= 200_000: break
                code = it.get("text", "")
                if len(code) > 80:
                    out_f.write(json.dumps({
                        "instruction": "Write a clean, optimized Python implementation with docstrings and type annotations.",
                        "input": "",
                        "output": code,
                        "domain": "code"
                    }) + "\n")
                    total_added += 1
                    if total_added % 50_000 == 0:
                        log.info(f"  • Ingested {total_added:,} samples...")
        except Exception as e:
            log.warning(f"Could not stream python-edu: {e}")

        # 3. FineWeb-Edu Curated Knowledge (300K samples)
        try:
            log.info("📥 [3/4] Streaming FineWeb-Edu World Knowledge (300K)...")
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)
            for i, it in enumerate(ds):
                if i >= 300_000: break
                text = it.get("text", "")
                if len(text) > 100:
                    out_f.write(json.dumps({
                        "instruction": "Provide a comprehensive, accurate, and detailed factual explanation.",
                        "input": text[:150],
                        "output": text,
                        "domain": "general"
                    }) + "\n")
                    total_added += 1
                    if total_added % 50_000 == 0:
                        log.info(f"  • Ingested {total_added:,} samples...")
        except Exception as e:
            log.warning(f"Could not stream fineweb-edu: {e}")

        # 4. Open-Orca Reasoning & Logic (200K samples)
        try:
            log.info("📥 [4/4] Streaming Open-Orca GPT-4 Reasoning & Logic (200K)...")
            ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
            for i, it in enumerate(ds):
                if i >= 200_000: break
                out_f.write(json.dumps({
                    "instruction": it.get("question", ""),
                    "input": it.get("system_prompt", ""),
                    "output": it.get("response", ""),
                    "domain": "math" if any(k in it.get("question", "").lower() for k in ["math", "calculate", "solve", "x="]) else "general"
                }) + "\n")
                total_added += 1
                if total_added % 50_000 == 0:
                    log.info(f"  • Ingested {total_added:,} samples...")
        except Exception as e:
            log.warning(f"Could not stream open-orca: {e}")

    log.info(f"🎉 Gigabyte Super Corpus Complete: {total_added:,} total samples written to {master_path}!")
    return total_added


def build_4track_curriculum(datasets_dir: str = "Datasets", force: bool = False) -> None:
    """Partitions all available master and gold datasets into 4 expert tracks ordered by curriculum complexity."""
    os.makedirs(datasets_dir, exist_ok=True)
    expected_files = [os.path.join(datasets_dir, f) for f in CURRICULUM_TRACKS.keys()]
    
    if not force and all(os.path.exists(p) and os.path.getsize(p) > 50_000 for p in expected_files):
        log.info(f"⚡ [CACHE HIT] 4-Track Domain Curriculum cached in {datasets_dir}/.")
        return

    generate_gold_datasets(datasets_dir=datasets_dir, force=force)
    
    # Collect all candidate source JSONL files (excluding the target partitioned files)
    target_filenames = set(CURRICULUM_TRACKS.keys())
    source_files = []
    for fname in os.listdir(datasets_dir):
        if fname.endswith(".jsonl") and fname not in target_filenames and "preference" not in fname and "sample" not in fname:
            source_files.append(os.path.join(datasets_dir, fname))

    if not source_files:
        source_files = [os.path.join(datasets_dir, "gold_corpus.jsonl")]

    log.info(f"📚 Partitioning sources: {[os.path.basename(p) for p in source_files]}")

    # Load, deduplicate, and partition items
    track_buckets = {f: [] for f in CURRICULUM_TRACKS.keys()}
    filter_dedup = QualityFilterAndDeduplicator()

    for src in source_files:
        if not os.path.exists(src):
            continue
        with open(src, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    u = data.get("user") or data.get("instruction") or data.get("prompt") or ""
                    a = data.get("assistant") or data.get("output") or data.get("response") or ""
                    d = data.get("domain") or data.get("category") or ""
                    if "messages" in data and isinstance(data["messages"], list):
                        for m in data["messages"]:
                            if m.get("role") == "user": u += " " + m.get("content", "")
                            elif m.get("role") == "assistant": a += " " + m.get("content", "")
                    
                    if not filter_dedup.is_clean(str(u), str(a)):
                        continue

                    text = (str(u) + " " + str(a) + " " + str(d)).lower()
                    matched = False
                    for target_file, keywords in CURRICULUM_TRACKS.items():
                        if any(kw in text for kw in keywords):
                            track_buckets[target_file].append(data)
                            matched = True
                            break
                    if not matched:
                        track_buckets["expert_general.jsonl"].append(data)
                except Exception:
                    continue

    # Sort each track from Easy (complexity 1) ➔ Hard (complexity 3)
    total_samples = 0
    for target_file, items in track_buckets.items():
        sorted_items = sorted(items, key=lambda x: (x.get("complexity", 1), len(x.get("output", ""))))
        out_path = os.path.join(datasets_dir, target_file)
        with open(out_path, "w", encoding="utf-8") as f:
            for it in sorted_items:
                f.write(json.dumps(it) + "\n")
        total_samples += len(sorted_items)
        log.info(f"  • {target_file}: {len(sorted_items):,} curriculum-ordered samples written.")
    
    log.info(f"🎯 Total Master Dataset Partitioned: {total_samples:,} samples across 4 tracks.")

