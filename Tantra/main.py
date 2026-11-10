"""
Tantra CLI
==========
    python -m Tantra seed                         # write curated stage-1 data (English)
    python -m Tantra stage1 --preset tiny         # tokenizer + base + 14 experts + router
    python -m Tantra stage2                      # add 5 more experts (tasks/reasoning/multilingual)
    python -m Tantra tokenizer --vocab 8192
    python -m Tantra base --preset tiny --steps 1500
    python -m Tantra expert greeting --steps 300  # train ONE expert (others untouched)
    python -m Tantra router
    python -m Tantra checkpoint --tag my_snapshot   # save full model state to checkpoints/
    python -m Tantra chat [--int8]
    python -m Tantra ask "who are you?"
    python -m Tantra bench [--int8]
    python -m Tantra grow --layers 2              # deepen the base, output unchanged
    python -m Tantra info                         # sizes, experts, presets
"""

from __future__ import annotations



import argparse
import json
import logging
import sys
from pathlib import Path

from Tantra import PRESETS, estimate_params, load_taxonomy, validate_taxonomy

STAGE1_EXPERTS = [
    "greeting", "farewell", "smalltalk", "empathy", "apology", "clarification",
    "identity_self", "identity_capabilities", "identity_limitations", "humor", "safety_refusal",
    "gk_geography", "gk_science", "gk_math",
]
STAGE2_EXPERTS = ["summarization", "translation", "logic", "creative_writing", "lang_hi", "hindi_grammar"]


def _trainer(a):
    from Tantra.train import TantraTrainer, setup_cpu
    setup_cpu(a.threads)
    return TantraTrainer(a.model_dir, a.data)


def _targs(a, **defaults):
    from Tantra.train import TrainArgs
    kw = dict(defaults)
    for k in ("steps", "lr", "batch_size", "max_len", "max_minutes"):
        v = getattr(a, k, None)
        if v is not None:
            kw[k] = v
    return TrainArgs(**kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m Tantra", description="Tantra CPU sparse-MoA LM")
    ap.add_argument("--model-dir", default="Model")
    ap.add_argument("--data", nargs="+", default=["Datasets/curated"], help="expert data dirs (first wins)")
    ap.add_argument("--threads", type=int)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def train_opts(p):
        p.add_argument("--steps", type=int)
        p.add_argument("--lr", type=float)
        p.add_argument("--batch-size", type=int)
        p.add_argument("--max-len", type=int)
        p.add_argument("--max-minutes", type=float)

    p = sub.add_parser("seed"); p.add_argument("--n", type=int, default=1500)
    p = sub.add_parser("tokenizer"); p.add_argument("--vocab", type=int, default=8192)
    p = sub.add_parser("base"); p.add_argument("--preset", default="tiny", choices=list(PRESETS)); p.add_argument("--fresh", action="store_true"); train_opts(p)
    p = sub.add_parser("expert"); p.add_argument("names", nargs="+"); p.add_argument("--rank", type=int); train_opts(p)
    p = sub.add_parser("router"); p.add_argument("--steps", type=int, default=800)
    p = sub.add_parser("stage1"); p.add_argument("--preset", default="tiny", choices=list(PRESETS)); p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--base-steps", type=int, default=3000); p.add_argument("--expert-steps", type=int, default=300)
    p = sub.add_parser("stage2"); p.add_argument("--expert-steps", type=int, default=300)
    p.add_argument("names", nargs="*", help="experts to train (default: stage-2 set)")
    p = sub.add_parser("checkpoint"); p.add_argument("--tag", required=True, help="snapshot name under checkpoints/")
    for name in ("chat", "ask", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--int8", action="store_true", help="INT8 output head (faster, ~94%% top-1 agreement)")
        p.add_argument("--int8-full", action="store_true", help="also quantise transformer layers (more drift)")
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--max-tokens", type=int, default=96)
        p.add_argument("--no-think", action="store_true", help="skip check/search/fix loop")
        p.add_argument("--offline", action="store_true", help="never use internet or teacher model")
        p.add_argument("--eager", action="store_true", help="load every expert at start (default: on demand)")
        p.add_argument("--max-resident", type=int, default=8, help="experts kept in RAM when loading on demand")
        p.add_argument("--max-experts", type=int, default=2, help="experts active per request")
        p.add_argument("--timing", action="store_true", help="print route/load/first-token latency per turn")
        if name == "ask":
            p.add_argument("prompt")
    p = sub.add_parser("grow"); p.add_argument("--layers", type=int, default=2)
    sub.add_parser("info")
    sub.add_parser("eval")
    sub.add_parser("feedback")
    p = sub.add_parser("chats"); p.add_argument("--last", type=int, default=20)
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows console: don't crash on non-ASCII text
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    if a.cmd == "seed":
        from Tantra.conversation_train import write_seed_conversation as write_all
        from Tantra.conversation_train import write_seed_knowledge as write_knowledge
        print(json.dumps({**write_all(a.data[0], a.n), **write_knowledge(a.data[0], a.n)}, indent=1))
    elif a.cmd == "tokenizer":
        _trainer(a).train_tokenizer(a.vocab)
    elif a.cmd == "base":
        _trainer(a).train_base(a.preset, _targs(a, steps=1500), resume=not a.fresh)
    elif a.cmd == "expert":
        t = _trainer(a)
        model = t.load_model()
        for n in a.names:
            print(json.dumps(t.train_expert(n, model, _targs(a, steps=300, lr=3e-3, warmup=20), a.rank)))
        t.train_router(model)
    elif a.cmd == "router":
        _trainer(a).train_router(steps=a.steps)
    elif a.cmd == "checkpoint":
        t = _trainer(a)
        t.load_model()  # verify the model loads before snapshotting
        print(t.snapshot(a.tag))
    elif a.cmd == "stage1":
        from Tantra.conversation_train import write_seed_conversation as write_all
        from Tantra.conversation_train import write_general
        from Tantra.train import TrainArgs
        from Tantra.conversation_train import write_seed_knowledge as write_knowledge
        if not Path(a.data[0], "conversation").exists():
            write_all(a.data[0])
        if not Path(a.data[0], "knowledge").exists():
            write_knowledge(a.data[0])
        if not Path("Datasets/general", "general.jsonl").exists():
            write_general("Datasets/general")
        res = _trainer(a).run(STAGE1_EXPERTS, a.preset, a.vocab,
                              TrainArgs(steps=a.base_steps, batch_size=24, max_len=256, lr=1.5e-3, warmup=100, log_every=100),
                              TrainArgs(steps=a.expert_steps, lr=3e-3, warmup=20, log_every=100))
        for r in res:
            print(f"{r['expert']:24s} {r['params'] / 1e3:6.0f}K  loss {r['loss_base']:.3f} -> {r['loss_expert']:.3f}")
    elif a.cmd == "stage2":
        # Add stage-2 experts onto the EXISTING model (base untouched).
        from Tantra.conversation_train import write_stage2
        from Tantra.train import TrainArgs
        names = a.names or STAGE2_EXPERTS
        if not Path(a.data[0], "tasks", "summarization.jsonl").exists():
            write_stage2(a.data[0])
        if not Path(a.data[0], "language", "hindi_grammar.jsonl").exists():
            from Tantra.conversation_train import write_grammar
            write_grammar(a.data[0])
        t = _trainer(a)
        model = t.load_model()
        args = TrainArgs(steps=a.expert_steps, lr=3e-3, warmup=20, log_every=100)
        for n in names:
            print(json.dumps(t.train_expert(n, model, args)))
        acc = t.train_router(model)
        t.snapshot("stage2")
        print(f"router acc {acc:.2f} | experts: {model.expert_names}")
    elif a.cmd in ("chat", "ask", "bench"):
        from Tantra.train import setup_cpu
        setup_cpu(a.threads)
        eng = TantraEngine(a.model_dir, int8=a.int8 or a.int8_full, max_experts=a.max_experts, lazy=not a.eager,
                           expert_cache=a.max_resident, int8_layers=a.int8_full)
        timing = lambda: (f"   [route {eng.last_timing['route_s']}s | experts loaded {eng.last_timing['experts_loaded']} "
                          f"in {eng.last_timing['expert_load_s']}s | first token {eng.last_timing['ttft_s']}s | "
                          f"{eng.last_timing['tok_per_s']} tok/s]") if a.timing and getattr(eng, 'last_timing', None) else ""
        if a.cmd == "bench":
            print(json.dumps(eng.benchmark(), indent=1))
        elif a.cmd == "ask":
            msgs = [{"role": "user", "content": a.prompt}]
            if not a.no_think:
                from Tantra.reasoning import Reasoner
                reasoner = Reasoner(eng, a.data, use_web=not a.offline, use_teacher=not a.offline)
                res = reasoner.answer(msgs, a.temperature, a.max_tokens)
                print(f"[experts: {res['route']} | {res['source']}]\n{res['reply']}")
            else:
                out = eng.chat(msgs, temperature=a.temperature, max_new_tokens=a.max_tokens)
                print(f"[experts: {eng.last_route}]\n{out}{timing()}")
        else:
            from Tantra.evaluate import ChatLog
            log_ = ChatLog(a.model_dir)
            from Tantra.reasoning import Reasoner
            reasoner = None if a.no_think else Reasoner(eng, a.data, use_web=not a.offline,
                                                         use_teacher=not a.offline)
            show_trace = False
            msgs = []
            print("Tantra chat - saved to", log_.path)
            print("  /good          last reply was good (becomes training data)")
            print("  /bad           last reply was bad")
            print("  /fix <text>    give the correct reply (optionally /fix @expert <text>)")
            print("  /trace         show/hide how the answer was checked/searched")
            print("  /reset  clear history    quit  exit")
            while True:
                try:
                    u = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if u in ("quit", "exit"):
                    break
                if u == "/reset":
                    msgs = []; continue
                if u == "/trace":
                    show_trace = not show_trace; print(f"  trace {'on' if show_trace else 'off'}"); continue
                if u in ("/good", "/bad"):
                    print("  saved rating" if log_.rate(u[1:]) else "  nothing to rate yet"); continue
                if u.startswith("/fix"):
                    text, expert = u[4:].strip(), None
                    if text.startswith("@"):
                        expert, _, text = text[1:].partition(" ")
                    if text and log_.rate("fix", text.strip(), expert):
                        print(f"  saved correction{' for ' + expert if expert else ''}")
                        if msgs and msgs[-1]["role"] == "assistant":
                            msgs[-1]["content"] = text.strip()
                    else:
                        print("  usage: /fix <correct reply>")
                    continue
                if not u:
                    continue
                history = msgs[-4:]
                msgs = (msgs + [{"role": "user", "content": u}])[-6:]
                print("Tantra: ", end="", flush=True)
                if reasoner is None:
                    reply = "".join(print(c, end="", flush=True) or c for c in eng.stream(
                        msgs, temperature=a.temperature, max_new_tokens=a.max_tokens))
                    src = "model"
                else:
                    res = reasoner.answer(msgs, a.temperature, a.max_tokens)
                    reply, src = res["reply"], res["source"]
                    eng.last_route = res["route"]
                    print(reply, end="")
                print(f"   [{', '.join(f'{n}:{w:.2f}' for n, w in eng.last_route)} | {src}]" + (timing() if src == "model" else ""))
                if reasoner is not None and show_trace:
                    print("\n".join("    · " + t for t in reasoner.trace))
                msgs.append({"role": "assistant", "content": reply.strip()})
                log_.add(history, u, reply.strip(), eng.last_route)
    elif a.cmd == "eval":
        from Tantra.evaluate import print_eval, run_eval
        from Tantra.train import setup_cpu
        setup_cpu(a.threads)
        print_eval(run_eval(a.model_dir, a.data))
    elif a.cmd == "feedback":
        from Tantra.evaluate import build_feedback, chat_stats
        print("chat logs:", chat_stats())
        counts = build_feedback()
        print("feedback rows per expert:", counts or "none yet (rate replies in chat with /good or /fix)")
        if counts:
            print("now retrain those experts, e.g.: python -m Tantra expert " + " ".join(counts))
    elif a.cmd == "chats":
        from Tantra.evaluate import CHAT_DIR, chat_stats
        print(chat_stats())
        for f in sorted(CHAT_DIR.glob("*.jsonl"))[-1:]:
            for line in f.read_text(encoding="utf-8").splitlines()[-a.last:]:
                t = json.loads(line)
                route = ",".join(r[0] for r in t["route"])
                print(f"[{t['rating'] or '-':4s}] You: {t['user']}\n        Tantra ({route}): {t['fix'] or t['reply']}")
    elif a.cmd == "grow":
        t = _trainer(a)
        m = t.load_model()
        before = m.num_params()
        m.grow_depth(a.layers)
        t.save_model(m)
        for n in m.expert_names:
            from Tantra.train import save_expert
            save_expert(m, n, t.experts_dir)
        print(f"grew {before / 1e6:.2f}M -> {m.num_params() / 1e6:.2f}M params ({m.cfg.n_layers} layers). "
              f"Now continue: python -m Tantra base")
    elif a.cmd == "info":
        for k, c in PRESETS.items():
            print(f"preset {k:6s} ~{estimate_params(c) / 1e6:8.1f}M base params  d={c.d_model} L={c.n_layers} vocab={c.vocab_size}")
        tax = load_taxonomy()
        print("taxonomy:", "OK" if not validate_taxonomy(tax) else validate_taxonomy(tax))
        base = Path(a.model_dir, "base.pt")
        if base.exists():
            from Tantra.train import TantraModel
            m = TantraModel.load(base, Path(a.model_dir, "experts"))
            print(f"model: {m.num_params() / 1e6:.2f}M base, {m.cfg.n_layers} layers, experts: {m.expert_names}")
    return 0


# ──  model/ngram_router.py  ──
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



# ──  inference/inference_engine.py  ──
"""
Tantra inference (CPU)
======================
route -> activate top-k experts -> KV-cached generation.
Optional INT8 dynamic quantization of the base for ~4x less memory.
"""


import json
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch

from Tantra.train import TantraModel, quantize_for_cpu
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
            from Tantra import sha256
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
            r = NgramRouter.load(ng)
            # 'none' is the router's base-only class; every other name must be a registered expert.
            if set(r.names) - {"none"} <= set(self.model.expert_names):
                self.ngram = r
        self.cum_prob = cum_prob

    def route(self, text: str) -> List[tuple]:
        if self.ngram is not None:
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



# ──  inference/expert_cache.py  ──
"""On-demand expert (LoRA adapter) loading for CPU inference.

The model starts with every expert registered in the router but no adapter weights in
RAM. `activate(route)` loads the routed adapters from `experts_dir/<name>.pt`, keeps at
most `capacity` resident (least recently used are freed), checks each file against the
loaded base (shapes always; base/tokenizer SHA-256 when the file records them) and then
activates them. Adding experts therefore costs disk, not RAM, beyond the resident cap.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from Tantra.train import TantraModel


class ExpertMismatch(ValueError):
    """An adapter file was built for a different base or tokenizer."""


class ExpertCache:
    def __init__(self, model: TantraModel, experts_dir: str | Path, capacity: int = 8,
                 base_sha256: Optional[str] = None, tokenizer_sha256: Optional[str] = None, declared_experts: Optional[Dict] = None):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.model, self.dir, self.capacity = model, Path(experts_dir), capacity
        self.base_sha256, self.tokenizer_sha256 = base_sha256, tokenizer_sha256
        self.declared_experts = declared_experts
        self.resident: "OrderedDict[str, int]" = OrderedDict()   # name -> rank, oldest first
        self.loads = 0
        self.evictions = 0

    def available(self) -> List[str]:
        return sorted(p.stem for p in self.dir.glob("*.pt")) if self.dir.exists() else []

    def _read(self, name: str) -> Dict:
        if not isinstance(name,str) or Path(name).name!=name or any(c in name for c in '/\\:'):
            raise ExpertMismatch('Invalid expert file name')
        if self.declared_experts is not None and name not in self.declared_experts:
            raise ExpertMismatch(f"Expert '{name}' is not declared by the serving manifest")
        path = self.dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No adapter file for expert '{name}': {path}")
        if self.declared_experts is not None:
            from Tantra import sha256
            if sha256(path)!=self.declared_experts[name]['sha256']:
                raise ExpertMismatch(f"Expert '{name}' artifact checksum mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload,dict) or not isinstance(payload.get('rank'),int) or payload['rank']<1 or not isinstance(payload.get('state'),dict):
            raise ExpertMismatch(f"Expert '{name}' has invalid adapter metadata")
        if self.declared_experts is not None and payload['rank']!=self.declared_experts[name]['rank']:
            raise ExpertMismatch(f"Expert '{name}' rank disagrees with serving manifest")
        if any(not isinstance(t,torch.Tensor) or not bool(torch.isfinite(t).all()) for t in payload['state'].values()):
            raise ExpertMismatch(f"Expert '{name}' has invalid or non-finite tensors")
        for key, mine in (("base_sha256", self.base_sha256), ("tokenizer_sha256", self.tokenizer_sha256)):
            theirs = payload.get(key)
            if self.declared_experts is not None and (not theirs or not mine):
                raise ExpertMismatch(f"Expert '{name}' lacks required training identity: {key}")
            if theirs and mine and theirs != mine:
                raise ExpertMismatch(f"Expert '{name}' was trained for a different {key.split('_')[0]} "
                                     f"({theirs[:12]}... != {mine[:12]}...); retrain it for this base")
        return payload

    def ensure(self, names: Sequence[str]) -> None:
        """Make `names` resident, evicting least-recently-used experts not in `names`."""
        wanted = list(dict.fromkeys(names))
        if len(wanted) > self.capacity:
            raise ValueError(f"Route needs {len(wanted)} experts but the cache holds {self.capacity}")
        for name in wanted:
            if name in self.resident:
                self.resident.move_to_end(name)
                continue
            payload = self._read(name)
            while len(self.resident) >= self.capacity:
                old = next(n for n in self.resident if n not in wanted)
                self.model.detach_lora(old); del self.resident[old]; self.evictions += 1
            self.model.attach_lora(name, payload["state"], int(payload["rank"]))
            self.resident[name] = int(payload["rank"]); self.loads += 1

    def activate(self, route: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
        route = [(n, w) for n, w in route if n in self.model.expert_names]
        self.ensure([n for n, _ in route])
        self.model.set_active(route)
        return route

    def stats(self) -> Dict[str, object]:
        return {"resident": list(self.resident), "capacity": self.capacity, "loads": self.loads,
                "evictions": self.evictions, "available": len(self.available())}



# ──  inference/serving_manifest.py  ──
"""Strict integrity checks for newly assembled serving bundles.

Manifest-less historical folders remain supported; manifest-bearing bundles must
bind the actual served files as well as adapter training identities.
"""
import json
from pathlib import Path
from Tantra import sha256


class ServingIntegrityError(ValueError):
    pass


def load_serving_manifest(root):
    root=Path(root);path=root/'manifest.json'
    if not path.exists():return None
    try:manifest=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError) as exc:raise ServingIntegrityError('Invalid serving manifest') from exc
    if manifest.get('version')!=2:
        raise ServingIntegrityError('Serving manifest lacks version-2 artifact binding; rebuild the candidate with the current serving workflow')
    for key in ('base_sha256','tokenizer_sha256'):
        value=manifest.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ServingIntegrityError('Missing or invalid serving identity: '+key)
    artifacts=manifest.get('artifacts')
    if not isinstance(artifacts,dict):raise ServingIntegrityError('Missing serving artifact hashes')
    required={'base.pt','tokenizer.json'}
    if (root/'router_ngram.pt').exists():required.add('router_ngram.pt')
    if not required<=set(artifacts):raise ServingIntegrityError('Missing served base/tokenizer/router artifact binding')
    if set(artifacts)-{'base.pt','tokenizer.json','router_ngram.pt'}:raise ServingIntegrityError('Unknown serving artifact path')
    for name,expected in artifacts.items():
        candidate=root/name
        if not candidate.is_file() or sha256(candidate)!=expected:raise ServingIntegrityError('Serving artifact checksum mismatch: '+name)
    if artifacts['tokenizer.json']!=manifest['tokenizer_sha256']:raise ServingIntegrityError('Tokenizer artifact and training identity differ')
    experts=manifest.get('experts')
    if not isinstance(experts,dict):raise ServingIntegrityError('Missing expert declarations')
    for name,record in experts.items():
        if not isinstance(name,str) or not name or Path(name).name!=name or any(c in name for c in '/\\:'):
            raise ServingIntegrityError('Invalid serving expert name')
        if not isinstance(record,dict) or record.get('file')!=f'experts/{name}.pt' or not isinstance(record.get('rank'),int) or record['rank']<1:
            raise ServingIntegrityError('Invalid serving expert declaration: '+name)
        expected=record.get('sha256')
        if not isinstance(expected,str) or len(expected)!=64:raise ServingIntegrityError('Missing expert artifact hash: '+name)
    return manifest


if __name__ == '__main__':
    sys.exit(main())
