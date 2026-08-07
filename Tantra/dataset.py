"""
tantra/dataset.py — High-performance JSONL & raw text dataset loader for Tantra-LLM.
"""
from __future__ import annotations

import json
import os
import random
from typing import Iterator, List, Dict, Any, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader

from Tantra.utils import get_logger

log = get_logger(__name__)


def format_jsonl_prompt(item: Dict[str, Any]) -> str:
    """Format a JSONL entry into a structured conversation prompt.
    
    Supports both flat format ({system, user, assistant}) and ChatML format
    ({messages: [{role, content}, ...]}).
    """
    # Handle ChatML format: {"messages": [{"role": ..., "content": ...}]}
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
    
    # Handle flat format: {"system": ..., "user": ..., "assistant": ...}
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


class JSONLDataset(IterableDataset):
    """
    Streaming IterableDataset for JSONL files.
    Reads large dataset files line-by-line without loading entire files into RAM.
    """

    def __init__(self, jsonl_path: str, tokenizer: Any, seq_len: int = 128, max_samples: Optional[int] = None):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.seq_len = max(1, seq_len)
        self.max_samples = max_samples
        self.vocab_size = getattr(tokenizer, "vocab_size", 32000)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if not os.path.exists(self.jsonl_path):
            log.warning(f"Dataset path does not exist: {self.jsonl_path}. Returning synthetic stream.")
            return

        count = 0
        token_buffer: List[int] = []

        with open(self.jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    text = format_jsonl_prompt(item)
                except Exception:
                    text = line

                ids = self.tokenizer.encode(text, modality="text")
                if ids:
                    # Fast vector clamping via PyTorch tensor
                    t_ids = torch.tensor(ids, dtype=torch.long).clamp_(0, self.vocab_size - 1)
                    token_buffer.extend(t_ids.tolist())

                while len(token_buffer) >= self.seq_len + 1:
                    chunk = token_buffer[: self.seq_len + 1]
                    token_buffer = token_buffer[self.seq_len :]

                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)

                    yield x, y
                    count += 1

                    if self.max_samples and count >= self.max_samples:
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
