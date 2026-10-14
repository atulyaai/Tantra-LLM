"""
Tantra Tokenizer
================
Small byte-level BPE (8k-32k vocab). Byte-level means no <unk>: any Hindi,
English or emoji text is encodable. A small vocab keeps the embedding table
(often the biggest part of a small model) cheap on CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

PAD, BOS, END = "<|pad|>", "<|bos|>", "<|end|>"
SYSTEM, USER, ASSISTANT = "<|system|>", "<|user|>", "<|assistant|>"
SPECIAL_TOKENS = [PAD, BOS, END, SYSTEM, USER, ASSISTANT]


class TantraTokenizer:
    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.pad_id = tok.token_to_id(PAD)
        self.bos_id = tok.token_to_id(BOS)
        self.end_id = tok.token_to_id(END)
        self.user_id = tok.token_to_id(USER)
        self.assistant_id = tok.token_to_id(ASSISTANT)
        self.system_id = tok.token_to_id(SYSTEM)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 8192, min_frequency: int = 2) -> "TantraTokenizer":
        tok = Tokenizer(models.BPE())
        tok.normalizer = normalizers.NFC()
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok)

    @classmethod
    def load(cls, path: str | Path) -> "TantraTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(path))

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    # ── chat format ──
    #   <|bos|><|system|>..<|end|><|user|>..<|end|><|assistant|>..<|end|>
    def encode_chat(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False):
        """Returns (ids, labels_mask) where mask=1 on assistant tokens (incl. their <|end|>)."""
        role_id = {"system": self.system_id, "user": self.user_id, "assistant": self.assistant_id}
        ids, mask = [self.bos_id], [0]
        for m in messages:
            body = self.encode(m["content"].strip()) + [self.end_id]
            is_asst = m["role"] == "assistant"
            ids += [role_id[m["role"]]] + body
            mask += [0] + [1 if is_asst else 0] * len(body)
        if add_generation_prompt:
            ids.append(self.assistant_id)
            mask.append(0)
        return ids, mask
