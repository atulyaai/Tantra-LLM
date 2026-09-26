"""
Tantra/tokenizer.py — Byte-level BPE tokenizer (Hindi + English).

One tokenizer, one file: Model/tokenizer.json.
Build it once with:  python main.py --mode tokenizer
Never change it in the middle of a training run.
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Iterable, List, Optional

from Tantra.config import VocabConfig

# Splits text into words while keeping Hindi vowel signs (matras, \p{M}) attached.
_SPLIT_REGEX = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}+|[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+"

_DEVA = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")


class ByteBPETokenizer:
    """Byte-level BPE: any text can be encoded, nothing is ever out-of-vocabulary."""

    def __init__(self, config: VocabConfig):
        self._config = config
        from tokenizers import Regex, Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split

        self._tokenizer = Tokenizer(BPE())
        self._tokenizer.pre_tokenizer = Sequence([
            Split(Regex(_SPLIT_REGEX), behavior="isolated"),
            ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        self._tokenizer.decoder = ByteLevelDecoder()

    def train(self, corpus_paths: List[str], vocab_size: int = 64000,
              special_tokens: Optional[List[str]] = None) -> None:
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
        special_tokens = special_tokens or list(self._config.special_tokens.keys())
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens,
                             initial_alphabet=ByteLevel.alphabet())
        self._tokenizer.train(corpus_paths, trainer)

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tokenizer.token_to_id(token)

    def save(self, path: str) -> None:
        self._tokenizer.save(path)

    @classmethod
    def load(cls, path: str, config: VocabConfig) -> "ByteBPETokenizer":
        from tokenizers import Tokenizer
        inst = cls(config)
        try:
            inst._tokenizer = Tokenizer.from_file(path)
        except Exception as exc:
            raise RuntimeError(f"Could not load tokenizer {path}: {exc}") from exc
        return inst

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()


class UnifiedTokenizer:
    """Thin wrapper used by the trainer, chat and WebUI (text only)."""

    def __init__(self, config: VocabConfig, bpe: ByteBPETokenizer, patcher: object = None):
        self._config = config
        self.bpe = bpe

    def encode(self, text: str, modality: str = "text") -> List[int]:
        if modality != "text":
            raise ValueError("Only text is supported. Speech/vision are not built yet.")
        if not isinstance(text, str):
            raise ValueError("Text modality expects string input.")
        return self.bpe.encode(text)

    def encode_text(self, text: str) -> List[int]:
        return self.encode(text)

    def decode(self, token_ids: List[int], modality: str = "text") -> str:
        return self.bpe.decode(token_ids)

    def decode_text(self, token_ids: List[int]) -> str:
        return self.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        return self.bpe.vocab_size


def load_tokenizer(path: str, config: Optional[VocabConfig] = None) -> UnifiedTokenizer:
    config = config or VocabConfig()
    return UnifiedTokenizer(config, ByteBPETokenizer.load(path, config))


# ── Building a tokenizer ─────────────────────────────────────────────────────

def _row_texts(line: str) -> List[str]:
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return []
    if isinstance(d.get("messages"), list):
        return [str(m.get("content", "")) for m in d["messages"] if m.get("content")]
    return [str(d[k]) for k in ("text", "system", "user", "assistant", "content") if d.get(k)]


def build_tokenizer(data_paths: Iterable[str], out_dir: str, vocab_size: int = 64000,
                    max_mb: int = 300, english_share: float = 0.25,
                    max_english_repeat: int = 4, seed: int = 7) -> str:
    """Train a Hindi+English BPE tokenizer and write it to out_dir/tokenizer.json.

    English rows are oversampled (up to max_english_repeat times) so English
    still gets good word-pieces even though the corpus is mostly Hindi.
    Special tokens keep the exact same ids as VocabConfig.special_tokens.
    """
    random.seed(seed)
    budget = max_mb * 1024 * 1024
    hindi: List[str] = []
    english: List[str] = []
    seen = 0
    for path in data_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                for t in _row_texts(line):
                    (english if len(_LATIN.findall(t)) > len(_DEVA.findall(t)) else hindi).append(t)
                    seen += len(t.encode("utf-8"))
                if seen > budget * 3:
                    break

    random.shuffle(hindi)
    corpus: List[str] = []
    hi_used = en_used = 0
    for t in hindi:
        if hi_used >= budget * (1 - english_share):
            break
        corpus.append(t)
        hi_used += len(t.encode("utf-8"))
    for _ in range(max_english_repeat if english else 0):
        for t in english:
            if en_used >= budget * english_share:
                break
            corpus.append(t)
            en_used += len(t.encode("utf-8"))
    random.shuffle(corpus)

    os.makedirs(out_dir, exist_ok=True)
    sample = os.path.join(out_dir, "_corpus_sample.txt")
    with open(sample, "w", encoding="utf-8") as f:
        for t in corpus:
            f.write(t.replace("\r", " ") + "\n")

    cfg = VocabConfig()
    specials = list(cfg.special_tokens.keys())
    bpe = ByteBPETokenizer(cfg)
    bpe.train([sample], vocab_size=vocab_size, special_tokens=specials)
    os.remove(sample)
    for i, s in enumerate(specials):
        if bpe.token_to_id(s) != i:
            raise RuntimeError(f"Special token {s!r} landed at id {bpe.token_to_id(s)}, expected {i}")
    path = os.path.join(out_dir, "tokenizer.json")
    bpe.save(path)
    print(f"Tokenizer saved: {path}  vocab={bpe.vocab_size:,}  "
          f"(sample: {hi_used/1e6:.0f} MB Hindi + {en_used/1e6:.0f} MB English)")
    return path
