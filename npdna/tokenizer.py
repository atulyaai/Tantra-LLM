"""Auto-scaling tokenizer with Hindi, Sanskrit, and English support.

Grows vocabulary automatically when encountering new tokens.
Byte-fallback ensures ANY text can be encoded, even unseen scripts.
"""

from __future__ import annotations

import json
import heapq
import logging
import math
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unicode ranges for Indic scripts
# ---------------------------------------------------------------------------

_DEVANAGARI = [(0x0900, 0x097F)]       # Hindi + Sanskrit shared
_DEVANAGARI_EXT = [(0xA8E0, 0xA8FF)]   # Devanagari Extended
_VEDIC = [(0x1CD0, 0x1CFF)]            # Vedic Extensions (Sanskrit)

_INDIC_RANGES = _DEVANAGARI + _DEVANAGARI_EXT + _VEDIC


def _indic_chars() -> list[str]:
    """All Devanagari + Vedic Unicode characters."""
    chars = []
    for start, end in _INDIC_RANGES:
        for cp in range(start, end + 1):
            try:
                chars.append(chr(cp))
            except (ValueError, OverflowError):
                pass
    return chars


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Digits are split out of the general word pattern and capped at 3 per
# pretoken (left-to-right), matching GPT-4/Llama-style number tokenization.
# Without this, "\w+" lets BPE learn merges over whole numbers, so e.g.
# "2024" and "2025" can end up as unrelated single tokens with no shared
# structure, which hurts arithmetic/numeric generalization. Splitting digits
# out also keeps code identifiers like "x1", "log2" sane: letters and digits
# just become adjacent tokens instead of one fused chunk.
_SPLIT_RE = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d]+| ?\d{1,3}| ?[^\s\w]+|\s+""", re.UNICODE)

SPECIAL_TOKENS: dict[str, int] = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}

# SPECIAL_TOKENS is a fixed module-level constant, so the split pattern used
# to peel special tokens out of raw text before BPE can be compiled once
# here instead of rebuilt on every single AtulyaTokenizer.encode() call.
_SPECIAL_TOKEN_RE = (
    re.compile(
        "(" + "|".join(re.escape(tok) for tok in sorted(SPECIAL_TOKENS, key=len, reverse=True)) + ")"
    )
    if SPECIAL_TOKENS
    else None
)

# Upper bound on the per-word BPE memoization cache (see _bpe_encode). Kept
# generous but finite so very long/diverse training runs don't grow this
# unbounded; if it fills up it's simply cleared and rebuilt.
_BPE_CACHE_MAX_SIZE = 200_000


class AtulyaTokenizer:
    """BPE tokenizer with byte-fallback and auto-growing vocabulary.

    Supports Hindi (Devanagari), Sanskrit (Vedic extensions), English,
    code, and emoji out of the box.  Vocabulary capacity grows
    automatically when ``fill_ratio`` crosses ``growth_threshold``.
    """

    def __init__(
        self,
        initial_capacity: int = 4096,
        max_capacity: int | None = None,
        growth_factor: float = 1.5,
        growth_threshold: float = 0.95,
    ):
        self.max_capacity = max_capacity
        self.growth_factor = growth_factor
        self.growth_threshold = growth_threshold
        self.growth_events: int = 0

        # Token ↔ ID mappings
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []

        # BPE merges
        self.merges: list[tuple[str, str]] = []
        self.merge_ranks: dict[tuple[str, str], int] = {}

        # Byte fallback
        self.byte_to_id: dict[int, int] = {}

        # Per-word BPE result cache. Keyed only on the word string, so it
        # must be invalidated whenever self.merges grows (see _bpe_encode).
        self._bpe_cache: dict[str, list[str]] = {}
        self._bpe_cache_merge_count: int = 0

        # Build initial vocabulary
        self._build_base_vocab()
        self._capacity = max(initial_capacity, len(self.id_to_token))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def fill_ratio(self) -> float:
        return self.size / max(1, self._capacity)

    @property
    def vocab_size(self) -> int:
        """Alias used by the model to size the embedding table."""
        return self._capacity

    # ------------------------------------------------------------------
    # Base vocabulary
    # ------------------------------------------------------------------

    def _add(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.id_to_token)
        self.token_to_id[token] = idx
        self.id_to_token.append(token)
        return idx

    def _build_base_vocab(self) -> None:
        """Populate specials + bytes + Indic characters."""
        for tok in SPECIAL_TOKENS:
            self._add(tok)

        # 256 byte tokens (universal fallback)
        for b in range(256):
            tok = f"<byte_{b:02x}>"
            idx = self._add(tok)
            self.byte_to_id[b] = idx

        # Devanagari + Vedic characters (Hindi + Sanskrit)
        for ch in _indic_chars():
            self._add(ch)

        # ASCII printable (a–z, A–Z, 0–9, punctuation)
        for cp in range(32, 127):
            self._add(chr(cp))

    def _invalidate_bpe_cache(self) -> None:
        """Drop memoized BPE encodings after direct token additions."""
        cache = getattr(self, "_bpe_cache", None)
        if cache is not None:
            cache.clear()
        self._bpe_cache_merge_count = len(self.merges)

    # ------------------------------------------------------------------
    # Capacity auto-growth
    # ------------------------------------------------------------------

    def _maybe_grow(self) -> None:
        if self.fill_ratio < self.growth_threshold:
            return
        new_cap = max(self._capacity + 1, math.ceil(self._capacity * self.growth_factor))
        if self.max_capacity is not None:
            new_cap = min(new_cap, self.max_capacity)
            if self._capacity >= self.max_capacity:
                return
        old = self._capacity
        self._capacity = new_cap
        self.growth_events += 1
        logger.info("Tokenizer grew: %d → %d (event #%d)", old, new_cap, self.growth_events)

    def ensure_capacity(self, min_capacity: int) -> None:
        """Grow capacity until it can hold at least ``min_capacity`` tokens."""
        min_capacity = max(self.size, int(min_capacity))
        if self.max_capacity is not None:
            min_capacity = min(min_capacity, self.max_capacity)
        while self._capacity < min_capacity:
            old = self._capacity
            new_cap = max(self._capacity + 1, math.ceil(self._capacity * self.growth_factor))
            if self.max_capacity is not None:
                new_cap = min(new_cap, self.max_capacity)
            if new_cap <= old:
                return
            self._capacity = new_cap
            self.growth_events += 1
            logger.info("Tokenizer capacity reserved: %d → %d (event #%d)", old, new_cap, self.growth_events)

    def add_token(self, token: str) -> int:
        """Add a token, growing capacity if needed. Returns token ID."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        if self.max_capacity is not None and self.size >= self.max_capacity:
            return self.token_to_id["<unk>"]
        self._maybe_grow()
        return self._add(token)

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str, allow_growth: bool = True) -> list[int]:
        """Encode text to token IDs. Falls back to byte encoding for unknowns."""
        ids: list[int] = []
        parts = _SPECIAL_TOKEN_RE.split(text) if _SPECIAL_TOKEN_RE is not None else [text]
        for part in parts:
            if not part:
                continue
            if part in SPECIAL_TOKENS:
                ids.append(SPECIAL_TOKENS[part])
                continue
            ids.extend(self._encode_plain(part, allow_growth=allow_growth))
        return ids

    def _encode_plain(self, text: str, allow_growth: bool = True) -> list[int]:
        ids: list[int] = []
        for chunk in _SPLIT_RE.findall(text):
            subwords = self._bpe_encode(chunk)
            for sw in subwords:
                if sw in self.token_to_id:
                    ids.append(self.token_to_id[sw])
                elif allow_growth:
                    ids.append(self.add_token(sw))
                else:
                    # Byte fallback
                    for b in sw.encode("utf-8"):
                        ids.append(self.byte_to_id.get(b, SPECIAL_TOKENS["<unk>"]))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token IDs back to text."""
        parts: list[str] = []
        byte_buf: list[int] = []

        def flush_bytes():
            if byte_buf:
                try:
                    parts.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                except Exception:
                    parts.append("".join(chr(b) for b in byte_buf))
                byte_buf.clear()

        for idx in ids:
            idx = int(idx)
            if idx < 0 or idx >= self.size:
                flush_bytes()
                continue
            tok = self.id_to_token[idx]
            if tok in SPECIAL_TOKENS:
                flush_bytes()
                continue
            if tok.startswith("<byte_") and tok.endswith(">"):
                byte_buf.append(int(tok[6:8], 16))
            else:
                flush_bytes()
                parts.append(tok)

        flush_bytes()
        return "".join(parts)

    # ------------------------------------------------------------------
    # BPE
    # ------------------------------------------------------------------

    def _bpe_encode(self, word: str) -> list[str]:
        """Apply BPE merges to a word (memoized per-word; see self._bpe_cache).

        Without caching, every occurrence of every word re-runs the full
        O(n^2) merge search from scratch, even for words like "the" that
        appear thousands of times per epoch. The cache is keyed only on the
        word string, so it must be dropped whenever self.merges changes
        (i.e. new BPE merges get learned mid-training).
        """
        if not word:
            return []

        cache = getattr(self, "_bpe_cache", None)
        if cache is None:
            # Defensive fallback for tokenizers constructed via
            # AtulyaTokenizer.__new__ (e.g. load()) that predate this cache.
            cache = self._bpe_cache = {}
            self._bpe_cache_merge_count = -1
        if self._bpe_cache_merge_count != len(self.merges):
            cache.clear()
            self._bpe_cache_merge_count = len(self.merges)

        cached = cache.get(word)
        if cached is not None:
            return cached

        if len(word) > 1 and word in self.token_to_id:
            cache[word] = [word]
            return cache[word]

        symbols = list(word)
        while len(symbols) > 1:
            best_pair = None
            best_rank = float("inf")
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None or best_pair not in self.merge_ranks:
                break
            a, b = best_pair
            merged = a + b
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        if len(cache) >= _BPE_CACHE_MAX_SIZE:
            cache.clear()
        cache[word] = symbols
        return symbols

    def train_bpe(
        self,
        texts: Iterable[str],
        target_merges: int = 4000,
        max_words: int | None = 0,
        min_pair_freq: int = 2,
        progress_callback: Callable[["AtulyaTokenizer"], None] | None = None,
        stop_callback: Callable[[], bool] | None = None,
    ) -> None:
        """Train or extend BPE merges on a text corpus.

        ``target_merges`` is the desired total merge count. Existing merges and
        token IDs are preserved so resumed training can safely improve the
        tokenizer without invalidating checkpoint weights.
        """
        if target_merges <= len(self.merges):
            logger.info("BPE already has %d/%d merges", len(self.merges), target_merges)
            return

        word_freqs: Counter[str] = Counter()
        for text in texts:
            for chunk in _SPLIT_RE.findall(text):
                word_freqs[chunk] += 1
        if max_words and max_words > 0 and len(word_freqs) > max_words:
            word_freqs = Counter(dict(word_freqs.most_common(max_words)))

        splits: dict[str, tuple[str, ...]] = {w: tuple(self._bpe_encode(w)) for w in word_freqs}
        logger.info(
            "Training BPE: %d unique words, extending %d -> %d merges",
            len(word_freqs),
            len(self.merges),
            target_merges,
        )

        pair_freqs: Counter[tuple[str, str]] = Counter()
        pair_words: dict[tuple[str, str], set[str]] = {}

        def add_pair(pair: tuple[str, str], word: str, freq: int) -> None:
            pair_freqs[pair] += freq
            pair_words.setdefault(pair, set()).add(word)

        def discard_pair(pair: tuple[str, str], word: str, freq: int) -> None:
            pair_freqs[pair] -= freq
            if pair_freqs[pair] <= 0:
                pair_freqs.pop(pair, None)
                pair_words.pop(pair, None)
                return
            words = pair_words.get(pair)
            if words is not None:
                words.discard(word)
                if not words:
                    pair_words.pop(pair, None)

        def adjacent_pairs(syms: tuple[str, ...]) -> list[tuple[str, str]]:
            return [(syms[i], syms[i + 1]) for i in range(len(syms) - 1)]

        def merge_symbols(syms: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
            a, b = pair
            merged = a + b
            out: list[str] = []
            i = 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            return tuple(out)

        for word, syms in splits.items():
            freq = word_freqs[word]
            for pair in adjacent_pairs(syms):
                add_pair(pair, word, freq)

        heap: list[tuple[int, tuple[str, str]]] = [
            (-freq, pair) for pair, freq in pair_freqs.items() if freq >= min_pair_freq
        ]
        heapq.heapify(heap)

        while len(self.merges) < target_merges:
            best_pair = None
            while heap:
                neg_freq, pair = heapq.heappop(heap)
                freq = -neg_freq
                current_freq = pair_freqs.get(pair, 0)
                if current_freq != freq:
                    if current_freq >= min_pair_freq and pair not in self.merge_ranks:
                        heapq.heappush(heap, (-current_freq, pair))
                    continue
                if pair in self.merge_ranks:
                    continue
                if current_freq < min_pair_freq:
                    break
                best_pair = pair
                break
            if best_pair is None:
                break

            if stop_callback is not None and stop_callback():
                logger.info("BPE training stopped at %d/%d merges", len(self.merges), target_merges)
                break

            a, b = best_pair
            merged = a + b

            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1
            self.add_token(merged)

            affected_words = list(pair_words.get(best_pair, set()))
            for word in affected_words:
                old_syms = splits[word]
                old_pairs = adjacent_pairs(old_syms)
                if best_pair not in old_pairs:
                    continue

                freq = word_freqs[word]
                for pair in old_pairs:
                    discard_pair(pair, word, freq)

                new_syms = merge_symbols(old_syms, best_pair)
                splits[word] = new_syms

                for pair in adjacent_pairs(new_syms):
                    add_pair(pair, word, freq)
                    if pair not in self.merge_ranks and pair_freqs[pair] >= min_pair_freq:
                        heapq.heappush(heap, (-pair_freqs[pair], pair))

            if len(self.merges) % 1000 == 0:
                logger.info("  BPE merge %d/%d, vocab=%d", len(self.merges), target_merges, self.size)
                if progress_callback is not None:
                    progress_callback(self)

        logger.info("BPE training done: %d merges, vocab=%d", len(self.merges), self.size)
        if progress_callback is not None:
            progress_callback(self)

    def dynamic_vocab_growth(
        self,
        texts: Iterable[str],
        sample_size: int = 10_000,
        merge_rounds: int = 500,
        min_pair_freq: int = 3,
        target_vocab_size: int = 0,
        return_stats: bool = False,
    ) -> int | tuple[int, dict[str, int]]:
        """Grow vocabulary from a bounded sample only when there is capacity pressure.

        Returns the number of new vocabulary tokens added.
        """
        if self.max_capacity is not None and self.size >= self.max_capacity:
            if return_stats:
                return 0, {
                    "sampled_texts": 0,
                    "target_merges": len(self.merges),
                    "forced_tokens": 0,
                }
            return 0

        sample: list[str] = []
        for text in texts:
            sample.append(text)
            if len(sample) >= sample_size:
                break
        if not sample:
            return (0, {"sampled_texts": 0, "target_merges": len(self.merges)}) if return_stats else 0

        old_size = self.size
        effective_merge_rounds = max(1, merge_rounds)
        if target_vocab_size:
            remaining_vocab = max(0, target_vocab_size - self.size)
            if remaining_vocab <= 0:
                if return_stats:
                    return 0, {
                        "sampled_texts": len(sample),
                        "target_merges": len(self.merges),
                        "forced_tokens": 0,
                    }
                return 0
            effective_merge_rounds = min(effective_merge_rounds, remaining_vocab)
        target_merges = len(self.merges) + effective_merge_rounds
        if self.max_capacity is not None:
            target_merges = min(target_merges, self.max_capacity - old_size + len(self.merges))
        self.train_bpe(sample, target_merges=target_merges, min_pair_freq=min_pair_freq)
        forced_tokens = 0
        if target_vocab_size and self.size < target_vocab_size:
            chunk_freqs: Counter[str] = Counter()
            for text in sample:
                for chunk in _SPLIT_RE.findall(text):
                    if len(chunk) > 1:
                        chunk_freqs[chunk] += 1
            for chunk, freq in chunk_freqs.most_common():
                if self.size >= target_vocab_size:
                    break
                if freq < min_pair_freq:
                    break
                before = self.size
                self.add_token(chunk)
                if self.size > before:
                    forced_tokens += 1
            if forced_tokens:
                self._invalidate_bpe_cache()
        added = max(0, self.size - old_size)
        if return_stats:
            return added, {
                "sampled_texts": len(sample),
                "target_merges": target_merges,
                "forced_tokens": forced_tokens,
            }
        return added

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "merges": self.merges,
            "capacity": self._capacity,
            "max_capacity": self.max_capacity,
            "growth_events": self.growth_events,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AtulyaTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls.__new__(cls)
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = data["id_to_token"]
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.merge_ranks = {tuple(m): i for i, m in enumerate(data["merges"])}
        tok._capacity = data.get("capacity", len(tok.id_to_token))
        tok.max_capacity = data.get("max_capacity")
        tok.growth_factor = 1.5
        tok.growth_threshold = 0.95
        tok.growth_events = data.get("growth_events", 0)
        tok.byte_to_id = {}
        tok._bpe_cache = {}
        tok._bpe_cache_merge_count = len(tok.merges)
        for b in range(256):
            t = f"<byte_{b:02x}>"
            if t in tok.token_to_id:
                tok.byte_to_id[b] = tok.token_to_id[t]
        return tok


# ---------------------------------------------------------------------------
# Audio / Vision encoders (multimodal)
# ---------------------------------------------------------------------------

@dataclass
class AudioFeatures:
    embedding: list[float]
    duration: float
    sample_rate: int
    channels: int


@dataclass
class VisionFeatures:
    embedding: list[float]
    width: int
    height: int
    channels: int


class AudioEncoder:
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

    def encode(self, audio_path: str) -> AudioFeatures:
        try:
            import wave
            with wave.open(audio_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                n_frames = wf.getnframes()
                duration = n_frames / sample_rate if sample_rate > 0 else 0
                frames = wf.readframes(min(n_frames, 1000))
        except Exception as e:
            logger.warning("AudioEncoder: failed to read %s: %s", audio_path, e)
            sample_rate = 16000; channels = 1; duration = 0; frames = b""
        embedding = self._extract_features(frames, self.embedding_dim)
        return AudioFeatures(embedding=embedding, duration=duration,
                             sample_rate=sample_rate, channels=channels)

    def _extract_features(self, audio_bytes: bytes, dim: int) -> list[float]:
        if not audio_bytes:
            return [0.0] * dim
        features = []
        for i in range(dim):
            chunk = audio_bytes[i*4:(i+1)*4] if i*4 < len(audio_bytes) else b'\x00\x00\x00\x00'
            value = struct.unpack('<I', chunk.ljust(4, b'\x00')[:4])[0]
            features.append((value % 1000) / 1000.0)
        return features


class VisionEncoder:
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim

    def encode(self, image_path: str) -> VisionFeatures:
        try:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            width, height = img.size; channels = 3
            pixels = list(img.getdata())[:1000]
        except Exception as e:
            logger.warning("VisionEncoder: failed to read %s: %s", image_path, e)
            width = 224; height = 224; channels = 3; pixels = []
        embedding = self._extract_features(pixels, width, height, self.embedding_dim)
        return VisionFeatures(embedding=embedding, width=width, height=height, channels=channels)

    def _extract_features(self, pixels: list, width: int, height: int, dim: int) -> list[float]:
        if not pixels:
            return [0.0] * dim
        features = []
        for i in range(dim):
            idx = i % len(pixels)
            pixel = pixels[idx]
            if isinstance(pixel, tuple):
                value = sum(pixel) / len(pixel) / 255.0
            else:
                value = pixel / 255.0
            features.append(value)
        while len(features) < dim:
            features.append(0.0)
        return features[:dim]
