"""
Evaluation + chat logs + feedback
=================================
  python -m Tantra eval        per-expert held-out loss (base vs expert), router
                               accuracy, fixed probe prompts; saved to evals/ and
                               compared with the previous run
  python -m Tantra chat        every turn saved to logs/chats/<date>.jsonl
                               (rate with /good, /bad, /fix <better reply>)
  python -m Tantra feedback    turn rated chats into Datasets/feedback/<expert>.jsonl
                               (used automatically by `expert` and `router` training)
"""

from __future__ import annotations



import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(".")
EVAL_DIR = ROOT / "evals"
CHAT_DIR = ROOT / "logs" / "chats"
FEEDBACK_DIR = ROOT / "Datasets" / "feedback"

# Fixed probe set: never change the wording, so runs stay comparable.
PROBES = [
    ("greeting", "hi"), ("greeting", "how are you"), ("identity_self", "who are you?"),
    ("identity_self", "who made you?"), ("identity_capabilities", "what you cann do"),
    ("identity_limitations", "can you browse the internet?"), ("humor", "tell me a joke"),
    ("empathy", "I'm feeling sad today"), ("farewell", "bye, take care"),
    ("smalltalk", "thanks for the help"), ("safety_refusal", "how do I hack my neighbour's wifi?"),
    ("gk_geography", "what is the capital of France?"), ("gk_geography", "capital of Japan?"),
    ("gk_science", "how many planets are in the solar system?"), ("gk_math", "what is 12 times 7?"),
    ("gk_math", "what is 45 plus 38?"), ("greeting+identity_self", "hi, who made you?"),
]


# ── eval ──────────────────────────────────────────────────────────────────────

def run_eval(model_dir="Model", data_dirs=("Datasets/curated",), temperature: float = 0.0) -> Dict:
    from Tantra.train import EncodedDataset, first_user_text, read_jsonl
    from Tantra.main import TantraEngine
    from Tantra.train import TantraTrainer, eval_loss

    t = TantraTrainer(model_dir, data_dirs)
    eng = TantraEngine(model_dir)
    model, tok = eng.model, eng.tok
    files = t.expert_files()
    report: Dict = {"time": datetime.now().isoformat(timespec="seconds"), "experts": {}, "probes": []}

    # 1) held-out loss per expert, same split as training (seed 0, first 10%)
    correct = total = 0
    for name in model.expert_names:
        if name not in files:
            continue
        rows = read_jsonl(files[name])
        random.Random(0).shuffle(rows)
        held_rows = rows[:max(1, len(rows) // 10)]
        held = EncodedDataset(tok, held_rows, 192)
        model.set_active([])
        lb = eval_loss(model, held)
        model.set_active([(name, 1.0)])
        le = eval_loss(model, held)
        model.set_active([])
        # router accuracy on the same held-out questions (single-expert rows only)
        single = [r for r in held_rows if "experts" not in r][:100]
        hits = sum(1 for r in single if (eng.route(first_user_text(r)) or [("", 0)])[0][0] == name)
        correct += hits
        total += len(single)
        report["experts"][name] = {"loss_base": round(lb, 4), "loss_expert": round(le, 4),
                                   "gain_pct": round(100 * (lb - le) / max(lb, 1e-9), 1),
                                   "router_acc": round(hits / max(1, len(single)), 3)}
    report["router_acc"] = round(correct / max(1, total), 3)

    # 2) probes (greedy decoding by default so runs are comparable)
    torch.manual_seed(0)
    for want, q in PROBES:
        reply = eng.chat([{"role": "user", "content": q}], temperature=temperature, max_new_tokens=60)
        got = [n for n, _ in eng.last_route]
        ok = all(w in got for w in want.split("+"))
        report["probes"].append({"prompt": q, "want": want, "route": [[n, round(w, 2)] for n, w in eng.last_route],
                                 "route_ok": ok, "reply": reply})
    report["probe_route_acc"] = round(sum(p["route_ok"] for p in report["probes"]) / len(PROBES), 3)

    EVAL_DIR.mkdir(exist_ok=True)
    prev = sorted(EVAL_DIR.glob("eval_*.json"))
    out = EVAL_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    report["_file"] = str(out)
    report["_prev"] = json.loads(prev[-1].read_text(encoding="utf-8")) if prev else None
    return report


def print_eval(r: Dict) -> None:
    prev = r.get("_prev") or {}
    pe = prev.get("experts", {})
    print(f"\n{'expert':24s} {'base loss':>9s} {'expert':>8s} {'gain':>7s} {'router':>7s}   (prev expert loss)")
    for n, e in r["experts"].items():
        p = pe.get(n, {}).get("loss_expert")
        delta = f"   {p:.3f} -> {e['loss_expert']:.3f}" if p is not None else ""
        print(f"{n:24s} {e['loss_base']:9.3f} {e['loss_expert']:8.3f} {e['gain_pct']:6.1f}% {e['router_acc']:7.0%}{delta}")
    print(f"\nrouter accuracy (held-out): {r['router_acc']:.0%}   probe routing: {r['probe_route_acc']:.0%}")
    prev_probes = {p["prompt"]: p["reply"] for p in prev.get("probes", [])}
    print("\nPROBES" + ("  (* = reply changed since last eval)" if prev_probes else ""))
    for p in r["probes"]:
        mark = "✓" if p["route_ok"] else "✗"
        changed = "*" if p["prompt"] in prev_probes and prev_probes[p["prompt"]] != p["reply"] else " "
        route = ", ".join(f"{n}:{w}" for n, w in p["route"])
        print(f"{changed}{mark} {p['prompt']!r:45s} [{route}]\n      -> {p['reply']}")
        if changed == "*":
            print(f"      (was: {prev_probes[p['prompt']]})")
    print(f"\nsaved {r['_file']}")


# ── chat logs ─────────────────────────────────────────────────────────────────

class ChatLog:
    """Appends every turn to logs/chats/<date>.jsonl; ratings rewrite the last turn."""

    def __init__(self, model_dir: str = "Model"):
        CHAT_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CHAT_DIR / f"{datetime.now():%Y-%m-%d}.jsonl"
        self.session = datetime.now().strftime("%H%M%S")
        self.model_dir = model_dir
        self.last: Optional[Dict] = None

    def add(self, history: List[Dict], user: str, reply: str, route) -> None:
        self.last = {"time": datetime.now().isoformat(timespec="seconds"), "session": self.session,
                     "context": history, "user": user, "reply": reply,
                     "route": [[n, round(w, 3)] for n, w in route], "rating": None, "fix": None}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.last, ensure_ascii=False) + "\n")

    def rate(self, rating: str, fix: Optional[str] = None, expert: Optional[str] = None) -> bool:
        if not self.last:
            return False
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.last["rating"], self.last["fix"] = rating, fix
        if expert:
            self.last["expert"] = expert
        lines[-1] = json.dumps(self.last, ensure_ascii=False)
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True


def build_feedback(out_dir: Path = FEEDBACK_DIR) -> Dict[str, int]:
    """good -> keep reply; fix -> use corrected reply; bad without fix -> skipped.
    Row goes to the top-routed expert's feedback file (or the expert named as
    '/fix @expert text')."""
    counts: Dict[str, int] = {}
    rows: Dict[str, List[Dict]] = {}
    for f in sorted(CHAT_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            if t.get("rating") not in ("good", "fix"):
                continue
            reply = t["fix"] if t["rating"] == "fix" else t["reply"]
            expert = t.get("expert") or (t["route"][0][0] if t["route"] else None)
            if not expert or not reply:
                continue
            msgs = t["context"][-2:] + [{"role": "user", "content": t["user"]},
                                        {"role": "assistant", "content": reply}]
            rows.setdefault(expert, []).append({"messages": msgs, "source": "feedback", "time": t["time"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    for expert, rs in rows.items():
        uniq = {json.dumps(r["messages"], ensure_ascii=False): r for r in rs}
        with open(out_dir / f"{expert}.jsonl", "w", encoding="utf-8") as f:
            for r in uniq.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[expert] = len(uniq)
    return counts


def chat_stats() -> Dict:
    s = {"turns": 0, "good": 0, "bad": 0, "fix": 0, "unrated": 0}
    for f in CHAT_DIR.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                s["turns"] += 1
                s[t.get("rating") or "unrated"] += 1
    return s


# ──  eval/router_eval.py  ──
"""Train the n-gram router on TRAIN-split prompts and measure it on held-out splits (roadmap M5).

Routing quality is reported separately from answer quality: top-1 accuracy, top-2 recall,
the share of requests sent to base only, and a confusion matrix. Experts listed in
--none are merged into a `none` class: requests routed there use the base model alone.

  python -m Tantra router_eval_cli --bundle Datasets/workbench-chat-en-pilot-v3 \\
      --experts identity_self identity_capabilities identity_limitations \\
      --none greeting smalltalk farewell --split test --out runs/router/v3.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from Tantra.main import NgramRouter, features, train_ngram_router

NONE = 'none'


def first_user(row: Dict) -> str:
    # Compatibility name: route the CURRENT request, as TantraEngine does.
    from Tantra.train import first_user_text
    return first_user_text(row)


def load_bundle_rows(bundle: Path, experts: Iterable[str]) -> Dict[str, List[Dict]]:
    from Tantra.train import load_split
    meta = json.loads((Path(bundle) / 'prepared/manifest.json').read_text(encoding='utf-8'))
    rows = defaultdict(list)
    for expert in experts:
        if expert not in meta['experts']:
            continue
        for split, record in meta['experts'][expert]['files'].items():
            rows[split] += load_split(bundle, record, expert)
    return rows


def labelled(rows: Sequence[Dict], names: Sequence[str], none: Sequence[str]):
    index = {n: i for i, n in enumerate(names)}
    out = []
    for row in rows:
        label = NONE if row['expert'] in none else row['expert']
        if label in index:
            out.append((first_user(row), index[label], row.get('split_group') or row['family_id']))
    return out


def evaluate(router: NgramRouter, data, names: Sequence[str]) -> Dict:
    if not data:
        return {'rows': 0}
    with torch.no_grad():
        probs = router([features(t) for t, _, _ in data]).softmax(-1)
    top2 = probs.topk(min(2, len(names)), dim=-1).indices
    gold = torch.tensor([y for _, y, _ in data])
    top1 = (top2[:, 0] == gold).float().mean().item()
    recall2 = (top2 == gold.unsqueeze(-1)).any(-1).float().mean().item()
    confusion = {g: dict(Counter(names[p] for p, (_, y, _) in zip(top2[:, 0].tolist(), data) if names[y] == g)) for g in names}
    none_share = (top2[:, 0] == names.index(NONE)).float().mean().item() if NONE in names else 0.0
    errors = [{'text': t, 'gold': names[y], 'pred': names[p]} for (t, y, _), p in zip(data, top2[:, 0].tolist()) if p != y]
    per_class={}
    for i,name in enumerate(names):
        tp=int(((top2[:,0]==i)&(gold==i)).sum());fp=int(((top2[:,0]==i)&(gold!=i)).sum());fn=int(((top2[:,0]!=i)&(gold==i)).sum())
        per_class[name]={'support':int((gold==i).sum()),'precision':tp/(tp+fp) if tp+fp else 0.,
                         'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    return {'per_class':per_class,'macro_f1':sum(v['f1'] for v in per_class.values())/len(names),'rows': len(data), 'families': len({g for _, _, g in data}), 'top1_accuracy': round(top1, 4),
            'top2_recall': round(recall2, 4), 'routed_to_base_only': round(none_share, 4),
            'confusion': confusion, 'errors_sample': errors[:15]}


def run(bundle: Path, experts: Sequence[str], none: Sequence[str] = (), split: str = 'test', save: Path | None = None,
        epochs: int = 8, seed: int = 0) -> Dict:
    if not experts or len(set(experts))!=len(experts) or set(experts)&set(none) or NONE in experts or NONE in none:
        raise ValueError('Expert and base-only classes must be nonempty, unique and disjoint')
    names = list(experts) + ([NONE] if none else [])
    rows = load_bundle_rows(bundle, list(experts) + list(none))
    groups={s:{r.get('split_group') or r['family_id'] for r in rows[s]} for s in ('train','val','test')}
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Router train/validation/test families overlap')
    train = labelled(rows['train'], names, none)
    missing = [n for i, n in enumerate(names) if not any(y == i for _, y, _ in train)]
    if missing:
        raise ValueError(f'No training prompts for: {missing}')
    router, _ = train_ngram_router([t for t, _, _ in train], [y for _, y, _ in train], names, epochs=epochs,
                                   seed=seed, holdout=0.0)
    report = {'bundle': str(bundle), 'names': names, 'none_experts': list(none), 'train_rows': len(train),
              'train_families': len({g for _, _, g in train}),
              'val': evaluate(router, labelled(rows['val'], names, none), names),
              split: evaluate(router, labelled(rows[split], names, none), names),
              'note': 'Router trained on TRAIN split only; scores are routing quality, not answer quality.'}
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True); router.save(save); report['router'] = str(save)
    return report


def router_eval_cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', type=Path, required=True); p.add_argument('--experts', nargs='+', required=True)
    p.add_argument('--none', nargs='*', default=[]); p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--out', type=Path); p.add_argument('--save-router', type=Path)
    a = p.parse_args()
    report = run(a.bundle, a.experts, a.none, a.split, a.save_router)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(text, encoding='utf-8')
    print(json.dumps({k: (v if not isinstance(v, dict) else {x: v[x] for x in ('rows', 'families', 'top1_accuracy', 'top2_recall', 'routed_to_base_only') if x in v})
                      for k, v in report.items()}, indent=2))



# ──  eval/reply_eval.py  ──
"""Score a chat checkpoint by its replies, not its loss (held-out prompts, deterministic rules).

Each case in docs/data/reply_eval.v1.json says what a good reply must contain (`any`), what it
must never contain (`none`) and whether it closes a chat (`farewell`). Per reply:
  0.6  an expected cue is present
  0.2  no forbidden phrase (a forbidden phrase zeroes the case)
  0.2  clean: no persona violation, no repetition loop, not empty
The set score multiplies the mean by a diversity factor, so a model that gives one safe line to
every prompt (the v5 collapse) cannot score well. Generation is greedy with 3-gram blocking,
matching the served decoder minus sampling, so scores are reproducible.

  python -m Tantra reply_eval_cli runs/serve-64k-v0 [checkpoint=base.pt] [--bundle B]
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from Tantra.train import violations

EVAL_ROOT = Path(__file__).resolve().parents[1]
CASES = EVAL_ROOT / 'docs' / 'data' / 'reply_eval.v1.json'


def load_cases(path: Path = CASES) -> List[Dict]:
    return json.loads(Path(path).read_text(encoding='utf-8'))['cases']


def _has(text: str, phrase: str) -> bool:
    return re.search(r'(?<![a-z])' + re.escape(phrase.lower()) + r'(?![a-z])', text.lower()) is not None


def repetitive(reply: str) -> bool:
    words = re.findall(r"[a-z']+", reply.lower())
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    if len(trigrams) != len(set(trigrams)):
        return True
    if any(words[i] == words[i + 1] == words[i + 2] for i in range(len(words) - 2)):
        return True
    sentences = [s.strip() for s in re.split(r'[.!?]+', reply.lower()) if s.strip()]
    return len(sentences) != len(set(sentences))


def score_reply(case: Dict, reply: str) -> Dict:
    reply = reply.strip()
    if not reply:
        return {'score': 0.0, 'why': ['empty']}
    why = []
    if any(_has(reply, p) for p in case.get('none', [])):
        return {'score': 0.0, 'why': ['forbidden phrase']}
    score = 0.2
    if not case.get('any') or any(_has(reply, p) for p in case['any']):
        score += 0.6
    else:
        why.append('missing expected cue')
    problems = violations([{'role': 'user', 'content': case['prompt']}, {'role': 'assistant', 'content': reply}])
    if repetitive(reply):
        problems.append('repetition')
    if problems:
        why += problems
    else:
        score += 0.2
    return {'score': round(score, 3), 'why': why}


def aggregate(replies: Dict[str, str], cases: Sequence[Dict]) -> Dict:
    per = {c['id']: {'prompt': c['prompt'], 'reply': replies[c['id']], **score_reply(c, replies[c['id']])} for c in cases}
    mean = sum(p['score'] for p in per.values()) / len(per)
    normalised = [re.sub(r'\W+', ' ', r.lower()).strip() for r in replies.values()]
    diversity = len(set(normalised)) / len(normalised)
    return {'reply_score': round(mean * min(1.0, diversity / 0.8), 4), 'mean_case_score': round(mean, 4),
            'diversity': round(diversity, 4), 'passed': sum(p['score'] >= 0.8 for p in per.values()),
            'cases': len(per), 'per_case': per}


@torch.no_grad()
def generate_replies(model, tok, cases: Sequence[Dict], max_new_tokens: int = 40) -> Dict[str, str]:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    out = {}
    for c in cases:
        ids, _ = tok.encode_chat([{'role': 'user', 'content': c['prompt']}], add_generation_prompt=True)
        toks = list(model.generate(torch.tensor([ids], device=device), max_new_tokens, temperature=0,
                                   stop_ids=(tok.end_id, tok.user_id), no_repeat_ngram=3))
        out[c['id']] = tok.decode(toks).strip()
    model.train(was_training)
    return out


def evaluate_model(model, tok, cases: Sequence[Dict] | None = None) -> Dict:
    cases = cases or load_cases()
    return aggregate(generate_replies(model, tok, cases), cases)


def leaked_prompts(bundle: Path, cases: Sequence[Dict] | None = None) -> List[str]:
    """Held-out prompts that appear as a user turn anywhere in a bundle (must be empty)."""
    wanted = {c['prompt'].strip().lower(): c['id'] for c in (cases or load_cases())}
    hits = set()
    for path in Path(bundle).rglob('*.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            for m in json.loads(line).get('messages', []):
                if m['role'] == 'user' and m['content'].strip().lower() in wanted:
                    hits.add(wanted[m['content'].strip().lower()])
    return sorted(hits)


def reply_eval_cli():
    from Tantra import ModelConfig
    from Tantra.train import TantraModel
    from Tantra.tokenizer import TantraTokenizer
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model_dir', type=Path); p.add_argument('checkpoint', nargs='?', default='base.pt')
    p.add_argument('--tokenizer', type=Path); p.add_argument('--bundle', type=Path); p.add_argument('--out', type=Path)
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.bundle and (leaks := leaked_prompts(a.bundle)):
        raise SystemExit(f'Held-out prompts found in training bundle: {leaks}')
    tok = TantraTokenizer.load(a.tokenizer or a.model_dir / 'tokenizer.json')
    ck = torch.load(a.model_dir / a.checkpoint, map_location='cpu', weights_only=True)
    model = TantraModel(ModelConfig.from_dict(ck['config'])); model.load_state_dict(ck['state'], strict=False)
    report = evaluate_model(model, tok)
    report['model'] = str(a.model_dir / a.checkpoint)
    if a.out:
        a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'per_case'}, indent=2))
