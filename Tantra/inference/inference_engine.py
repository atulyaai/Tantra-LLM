"""
Tantra inference (CPU)
======================
route -> activate top-k experts -> KV-cached generation.
Optional INT8 dynamic quantization of the base for ~4x less memory.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch

from Tantra.model.tantra_model import TantraModel, quantize_for_cpu
from Tantra.tokenizer import TantraTokenizer

DEFAULT_SYSTEM = None  # identity is learned by the identity_* experts


def resolve_model_dir(model_dir: str | Path) -> Path:
    """Use `model_dir` if it holds a complete model; otherwise fall back to the
    newest complete snapshot in checkpoints/ (e.g. while a new model is training)."""
    model_dir = Path(model_dir)
    complete = lambda d: (d / "base.pt").exists() and (d / "tokenizer.json").exists()
    if complete(model_dir):
        return model_dir
    snaps = sorted((d for d in Path("checkpoints").glob("*") if d.is_dir() and complete(d)),
                   key=lambda d: (d / "base.pt").stat().st_mtime, reverse=True)
    stable = [d for d in snaps if d.name == "stage1_stable"]
    pick = (stable or snaps or [None])[0]
    if pick is None:
        raise SystemExit(f"No trained model in {model_dir}/ or checkpoints/. Train one first: python -m Tantra stage1")
    print(f"[note] {model_dir}/ has no finished model yet (training in progress?) - using {pick}/")
    return pick


class TantraEngine:
    def __init__(self, model_dir: str | Path = "Model", int8: bool = False, max_experts: int = 2,
                 cum_prob: float = 0.8):
        model_dir = resolve_model_dir(model_dir)
        self.model_dir = model_dir
        self.tok =TantraTokenizer.load(model_dir / "tokenizer.json")
        self.model = TantraModel.load(model_dir / "base.pt", model_dir / "experts").eval()
        if int8:
            quantize_for_cpu(self.model)
        self.max_experts = max_experts
        self.max_sentences = 4
        # Prefer the n-gram router when it was trained for exactly these experts.
        self.ngram = None
        ng = model_dir / "router_ngram.pt"
        if ng.exists():
            from Tantra.model.ngram_router import NgramRouter
            r = NgramRouter.load(ng)
            if set(r.names) <= set(self.model.expert_names):
                self.ngram = r
        self.cum_prob = cum_prob

    def route(self, text: str) -> List[tuple]:
        if self.ngram is not None:
            from Tantra.model.ngram_router import route_probs_to_experts
            return route_probs_to_experts(self.ngram.probs(text), self.ngram.names, self.max_experts, self.cum_prob)
        ids = torch.tensor([self.tok.encode(text) or [self.tok.pad_id]])
        return self.model.route(ids, self.max_experts, self.cum_prob)

    def stream(self, messages: List[Dict[str, str]], max_new_tokens: int = 96, temperature: float = 0.7,
               top_p: float = 0.9, experts: Optional[List[tuple]] = None) -> Iterator[str]:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        self.last_route = experts if experts is not None else self.route(last_user)
        self.model.set_active(self.last_route)
        ids, _ = self.tok.encode_chat(messages, add_generation_prompt=True)
        # Streams sentence by sentence. Stops early (saving tokens) when the model
        # starts repeating a sentence or exceeds max_sentences - small models tend
        # to loop ("You're welcome! Take care. You're welcome! ...").
        out: List[int] = []
        sent, seen = 0, set()
        done = ""  # text already yielded
        for t in self.model.generate(torch.tensor([ids]), max_new_tokens, temperature, top_p=top_p,
                                     stop_ids=(self.tok.end_id, self.tok.user_id)):
            out.append(t)
            text = self.tok.decode(out)
            if text.endswith("�"):  # wait for complete utf-8 chars
                continue
            m = re.match(r"\s*(.+?[.!?\u0964\u0965])(\s|$)", text[len(done):])
            if m and m.group(2):  # a full sentence followed by whitespace
                s = m.group(1)
                key = re.sub(r"\W+", " ", s.lower()).strip()
                if key in seen:
                    return
                seen.add(key)
                sent += 1
                yield (" " if done else "") + s
                done = text[:len(done) + m.end(1)]
                if sent >= self.max_sentences:
                    return
        rest = self.tok.decode(out)[len(done):].strip()
        if rest and re.sub(r"\W+", " ", rest.lower()).strip() not in seen:
            yield (" " if done else "") + rest

    def chat(self, messages: List[Dict[str, str]], **kw) -> str:
        return "".join(self.stream(messages, **kw)).strip()

    @torch.no_grad()
    def benchmark(self, prompt: str = "hello, who are you?", n_tokens: int = 64) -> Dict[str, float]:
        self.model.set_active(self.route(prompt))
        ids = torch.tensor([self.tok.encode_chat([{"role": "user", "content": prompt}], True)[0]])
        t0 = time.perf_counter()
        n = sum(1 for _ in self.model.generate(ids, n_tokens, temperature=1.0, top_k=0, top_p=1.0))
        dt = time.perf_counter() - t0
        return {"tokens": n, "seconds": round(dt, 3), "tokens_per_sec": round(n / dt, 1),
                "base_params_M": round(self.model.num_params() / 1e6, 2),
                "experts": len(self.model.expert_names)}
