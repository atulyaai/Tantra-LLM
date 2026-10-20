"""
N-gram router
=============
Routes a user message to experts using hashed word 1-2-grams + character
3-4-grams (fastText-style). Short chat messages with typos ("what you cann do",
"whats india capital") are routed far more reliably than by averaging token
embeddings, and it costs microseconds on CPU.

Stored next to the model as router_ngram.pt; growable: adding an expert adds
one output column.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BUCKETS = 1 << 18


def features(text: str) -> List[int]:
    t = text.lower()
    words = re.findall(r"[\w']+|[^\s\w]", t)
    feats = [f"w:{w}" for w in words] + [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
    padded = f" {' '.join(words)} "
    for n in (3, 4):
        feats += [f"c{n}:{padded[i:i + n]}" for i in range(len(padded) - n + 1)]
    feats.append("bias")
    return [zlib.crc32(f.encode()) % N_BUCKETS for f in feats]


class NgramRouter(nn.Module):
    def __init__(self, names: Sequence[str], dim: int = 32):
        super().__init__()
        self.names = list(names)
        self.emb = nn.EmbeddingBag(N_BUCKETS, dim, mode="mean")
        self.head = nn.Linear(dim, len(self.names))
        nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self, batch: List[List[int]]) -> torch.Tensor:
        flat = torch.tensor([i for f in batch for i in f], dtype=torch.long)
        offsets = torch.tensor([0] + [len(f) for f in batch[:-1]], dtype=torch.long).cumsum(0)
        return self.head(self.emb(flat, offsets))

    @torch.no_grad()
    def probs(self, text: str) -> torch.Tensor:
        return self([features(text)]).softmax(-1)[0]

    def save(self, path: str | Path) -> None:
        torch.save({"names": self.names, "dim": self.emb.embedding_dim, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path) -> "NgramRouter":
        ck = torch.load(path, map_location="cpu", weights_only=True)
        r = cls(ck["names"], ck["dim"])
        r.load_state_dict(ck["state"])
        return r.eval()


def train_ngram_router(texts: List[str], labels: List[int], names: Sequence[str], epochs: int = 8,
                       lr: float = 0.5, batch_size: int = 64, seed: int = 0,
                       holdout: float = 0.1, groups: Optional[List[str]] = None) -> Tuple[NgramRouter, float]:
    """Returns (router, held-out accuracy). Duplicate texts with different labels
    (mixed messages) teach the router to split probability between experts.
    Pass `groups` (e.g. family ids) so paraphrases of one source never sit on both
    sides of the hold-out; without groups the accuracy is optimistic."""
    import random
    rng = random.Random(seed)
    torch.manual_seed(seed)
    if groups is not None:
        keys = sorted(set(groups)); rng.shuffle(keys)
        held = set(keys[:int(len(keys) * holdout)])
        hold = [i for i, g in enumerate(groups) if g in held]
        train = [i for i, g in enumerate(groups) if g not in held]
    else:
        idx = list(range(len(texts)))
        rng.shuffle(idx)
        n_hold = int(len(idx) * holdout)
        hold, train = idx[:n_hold], idx[n_hold:]
    feats = [features(t) for t in texts]
    r = NgramRouter(names)
    opt = torch.optim.Adagrad(r.parameters(), lr=lr)
    r.train()
    for _ in range(epochs):
        rng.shuffle(train)
        for s in range(0, len(train), batch_size):
            b = train[s:s + batch_size]
            loss = F.cross_entropy(r([feats[i] for i in b]), torch.tensor([labels[i] for i in b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
    r.eval()
    acc = 0.0
    if hold:
        with torch.no_grad():
            pred = r([feats[i] for i in hold]).argmax(-1)
            acc = (pred == torch.tensor([labels[i] for i in hold])).float().mean().item()
    return r, acc


def route_probs_to_experts(probs: torch.Tensor, names: Sequence[str], max_experts: int = 2,
                           cum_prob: float = 0.8, min_prob: float = 0.2) -> List[Tuple[str, float]]:
    """Adaptive count, but conservative: a second expert is only added when it is
    genuinely likely (>= min_prob). Mixing many unsure LoRA experts garbles text."""
    vals, idx = probs.sort(descending=True)
    chosen, total = [], 0.0
    for v, i in zip(vals.tolist(), idx.tolist()):
        if len(chosen) >= max_experts or total >= cum_prob or (chosen and v < min_prob):
            break
        chosen.append((names[i], v))
        total += v
    return [(n, w / total) for n, w in chosen]
