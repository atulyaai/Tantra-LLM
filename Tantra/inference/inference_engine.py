"""
Tantra inference (CPU)
======================
route -> activate top-k experts -> KV-cached generation.
Optional INT8 dynamic quantization of the base for ~4x less memory.
"""

from __future__ import annotations

import json
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
                 cum_prob: float = 0.8, lazy: bool = True, expert_cache: int = 8, int8_layers: bool = False):
        model_dir = resolve_model_dir(model_dir)
        self.model_dir = model_dir
        from Tantra.inference.serving_manifest import load_serving_manifest, ServingIntegrityError
        serving = load_serving_manifest(model_dir)
        self.tok =TantraTokenizer.load(model_dir / "tokenizer.json")
        # Lazy (default): experts are registered with the router, and an adapter's weights are
        # loaded only when a request routes to it (at most `expert_cache` resident).
        self.model = TantraModel.load(model_dir / "base.pt", None if lazy or serving is not None else model_dir / "experts", lazy=lazy or serving is not None).eval()
        if self.tok.vocab_size!=self.model.cfg.vocab_size:
            raise ServingIntegrityError('Model/tokenizer vocabulary mismatch')
        if serving is not None and set(serving['experts'])!=set(self.model.expert_names):
            raise ServingIntegrityError('Model router registry and serving expert declarations differ')
        self.cache = None
        if lazy or serving is not None:
            from Tantra.data.public_sources import sha256
            from Tantra.inference.expert_cache import ExpertCache
            # A serving folder's manifest records the base the adapters were trained on (its base.pt
            # also carries router entries, so its own file hash differs); otherwise hash the files.
            ids = serving or {}
            self.cache = ExpertCache(self.model, model_dir / "experts", max(expert_cache, max_experts),
                                     ids.get("base_sha256") or sha256(model_dir / "base.pt"),
                                     ids.get("tokenizer_sha256") or sha256(model_dir / "tokenizer.json"),
                                     declared_experts=serving['experts'] if serving is not None else None)
            if not lazy:
                # Eager is still integrity-checked; only the retention policy differs.
                self.cache.capacity=max(self.cache.capacity,len(self.model.expert_names))
                self.cache.ensure(self.model.expert_names)
                self.cache=None
        if int8:
            # Head-only INT8 by default: ~45% faster with ~94% top-1 agreement; layers too drops to ~84%.
            quantize_for_cpu(self.model, layers=int8_layers)
        self.max_experts = max_experts
        self.max_sentences = 4
        # Prefer the n-gram router when it was trained for exactly these experts.
        self.ngram = None
        ng = model_dir / "router_ngram.pt"
        if ng.exists():
            from Tantra.model.ngram_router import NgramRouter
            r = NgramRouter.load(ng)
            # 'none' is the router's base-only class; every other name must be a registered expert.
            if set(r.names) - {"none"} <= set(self.model.expert_names):
                self.ngram = r
        self.cum_prob = cum_prob

    def route(self, text: str) -> List[tuple]:
        if self.ngram is not None:
            from Tantra.model.ngram_router import route_probs_to_experts
            route = route_probs_to_experts(self.ngram.probs(text), self.ngram.names, self.max_experts, self.cum_prob)
            if route and route[0][0] == "none":
                return []  # the router's best guess is "no expert": answer with the base alone
            kept = [(n, w) for n, w in route if n != "none"]
            total = sum(w for _, w in kept)
            return [(n, w / total) for n, w in kept] if total else []
        ids = torch.tensor([self.tok.encode(text) or [self.tok.pad_id]])
        return self.model.route(ids, self.max_experts, self.cum_prob)

    def stream(self, messages: List[Dict[str, str]], max_new_tokens: int = 96, temperature: float = 0.7,
               top_p: float = 0.9, experts: Optional[List[tuple]] = None) -> Iterator[str]:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        t0 = time.perf_counter()
        self.last_route = experts if experts is not None else self.route(last_user)
        t1 = time.perf_counter()
        loads_before = self.cache.loads if self.cache else 0
        self._activate(self.last_route)
        t2 = time.perf_counter()
        # Per-turn latency, filled in as generation proceeds (CPU budget tracking, roadmap M6).
        self.last_timing = {"route_s": round(t1 - t0, 4), "expert_load_s": round(t2 - t1, 4),
                            "experts_loaded": (self.cache.loads - loads_before) if self.cache else 0,
                            "ttft_s": None, "tokens": 0, "tok_per_s": None}
        ids, _ = self.tok.encode_chat(messages, add_generation_prompt=True)
        # Streams sentence by sentence. Stops early (saving tokens) when the model
        # starts repeating a sentence or exceeds max_sentences - small models tend
        # to loop ("You're welcome! Take care. You're welcome! ...").
        out: List[int] = []
        sent, seen = 0, set()
        done = ""  # text already yielded
        start = time.perf_counter()
        for t in self.model.generate(torch.tensor([ids]), max_new_tokens, temperature, top_p=top_p,
                                     stop_ids=(self.tok.end_id, self.tok.user_id), no_repeat_ngram=3):
            out.append(t)
            timing = self.last_timing
            timing["tokens"] = len(out)
            if timing["ttft_s"] is None:
                timing["ttft_s"] = round(time.perf_counter() - start, 4)
            timing["tok_per_s"] = round(len(out) / max(time.perf_counter() - start, 1e-9), 1)
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

    def _activate(self, route: List[tuple]) -> None:
        if self.cache is not None:
            self.cache.activate(route)
        else:
            self.model.set_active(route)

    def chat(self, messages: List[Dict[str, str]], **kw) -> str:
        return "".join(self.stream(messages, **kw)).strip()

    @torch.no_grad()
    def benchmark(self, prompt: str = "hello, who are you?", n_tokens: int = 64) -> Dict[str, float]:
        self._activate(self.route(prompt))
        ids = torch.tensor([self.tok.encode_chat([{"role": "user", "content": prompt}], True)[0]])
        t0 = time.perf_counter()
        n = sum(1 for _ in self.model.generate(ids, n_tokens, temperature=1.0, top_k=0, top_p=1.0))
        dt = time.perf_counter() - t0
        return {"tokens": n, "seconds": round(dt, 3), "tokens_per_sec": round(n / dt, 1),
                "base_params_M": round(self.model.num_params() / 1e6, 2),
                "experts": len(self.model.expert_names),
                "experts_resident": len(self.cache.resident) if self.cache else len(self.model.expert_names)}
