"""
Inverted BM25 Index for Fast CPU Retrieval
==========================================
Replaces linear document scanning with an inverted posting index.
Provides O(posting_len) sub-millisecond retrieval across hundreds
of thousands of facts. Supports full Unicode (English, Hindi, Sanskrit).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

STOP_WORDS = set(
    "a an the is are was were of in on at to for and or what whats what's which who whom how many much "
    "tell me please do does did can you your i it its this that be by with from as about ka ki ke ko se me par".split()
)


def tokenize_query(text: str) -> List[str]:
    """Tokenize text into lowercase Unicode word tokens excluding stop words."""
    words = re.findall(r"[\w']+", text.lower())
    return [w for w in words if w not in STOP_WORDS]


class InvertedBM25Index:
    """Okapi BM25 index with inverted posting lists for O(posting) lookup."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # token -> list of (doc_id, term_frequency)
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_lengths: Dict[int, int] = {}
        self.total_docs: int = 0
        self.avg_doc_len: float = 0.0
        self.idf_cache: Dict[str, float] = {}

    def __len__(self) -> int:
        return self.total_docs

    def add_document(self, doc_id: int, text: str) -> None:
        """Add a document to the inverted index."""
        tokens = tokenize_query(text)
        doc_len = len(tokens)
        self.doc_lengths[doc_id] = doc_len
        tf = Counter(tokens)
        for term, count in tf.items():
            self.postings[term].append((doc_id, count))
        self.total_docs += 1
        self._update_stats()

    def _update_stats(self) -> None:
        total_len = sum(self.doc_lengths.values())
        self.avg_doc_len = total_len / max(1, self.total_docs)
        self.idf_cache.clear()

    def idf(self, term: str) -> float:
        """Calculate Okapi BM25 IDF for a term."""
        if term in self.idf_cache:
            return self.idf_cache[term]
        df = len(self.postings.get(term, []))
        if df == 0:
            val = 0.0
        else:
            val = math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))
        self.idf_cache[term] = val
        return val

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search query returning [(doc_id, bm25_score), ...] sorted descending."""
        q_tokens = tokenize_query(query)
        if not q_tokens or self.total_docs == 0:
            return []

        scores: Dict[int, float] = defaultdict(float)
        for term in set(q_tokens):
            postings_list = self.postings.get(term)
            if not postings_list:
                continue
            idf_val = self.idf(term)
            for doc_id, tf in postings_list:
                d_len = self.doc_lengths.get(doc_id, self.avg_doc_len)
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (d_len / max(1.0, self.avg_doc_len)))
                scores[doc_id] += idf_val * (num / denom)

        if not scores:
            return []
        sorted_docs = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_docs[:top_k]

    def state_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "postings": dict(self.postings),
            "doc_lengths": dict(self.doc_lengths),
            "total_docs": self.total_docs,
        }

    def load_state_dict(self, d: dict) -> None:
        self.k1 = d["k1"]
        self.b = d["b"]
        self.postings = defaultdict(list, d["postings"])
        self.doc_lengths = d["doc_lengths"]
        self.total_docs = d["total_docs"]
        self._update_stats()
