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

from Tantra.config import PRESETS, estimate_params, load_taxonomy, validate_taxonomy

STAGE1_EXPERTS = [
    "greeting", "farewell", "smalltalk", "empathy", "apology", "clarification",
    "identity_self", "identity_capabilities", "identity_limitations", "humor", "safety_refusal",
    "gk_geography", "gk_science", "gk_math",
]
STAGE2_EXPERTS = ["summarization", "translation", "logic", "creative_writing", "lang_hi", "hindi_grammar"]


def _trainer(a):
    from Tantra.training.tantra_trainer import TantraTrainer, setup_cpu
    setup_cpu(a.threads)
    return TantraTrainer(a.model_dir, a.data)


def _targs(a, **defaults):
    from Tantra.training.tantra_trainer import TrainArgs
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
        p.add_argument("--int8", action="store_true")
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
        from Tantra.data.seed_conversation import write_all
        from Tantra.data.seed_knowledge import write_all as write_knowledge
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
        from Tantra.data.seed_conversation import write_all
        from Tantra.data.seed_general import write_all as write_general
        from Tantra.training.tantra_trainer import TrainArgs
        from Tantra.data.seed_knowledge import write_all as write_knowledge
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
        from Tantra.data.seed_stage2 import write_all as write_stage2
        from Tantra.training.tantra_trainer import TrainArgs
        names = a.names or STAGE2_EXPERTS
        if not Path(a.data[0], "tasks", "summarization.jsonl").exists():
            write_stage2(a.data[0])
        if not Path(a.data[0], "language", "hindi_grammar.jsonl").exists():
            from Tantra.data.seed_hindi_grammar import write_all as write_grammar
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
        from Tantra.training.tantra_trainer import setup_cpu
        from Tantra.inference.inference_engine import TantraEngine
        setup_cpu(a.threads)
        eng = TantraEngine(a.model_dir, int8=a.int8, max_experts=a.max_experts, lazy=not a.eager,
                           expert_cache=a.max_resident)
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
        from Tantra.training.tantra_trainer import setup_cpu
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
            from Tantra.model.tantra_model import save_expert
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
            from Tantra.model.tantra_model import TantraModel
            m = TantraModel.load(base, Path(a.model_dir, "experts"))
            print(f"model: {m.num_params() / 1e6:.2f}M base, {m.cfg.n_layers} layers, experts: {m.expert_names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
