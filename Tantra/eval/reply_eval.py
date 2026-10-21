"""Score a chat checkpoint by its replies, not its loss (held-out prompts, deterministic rules).

Each case in docs/data/reply_eval.v1.json says what a good reply must contain (`any`), what it
must never contain (`none`) and whether it closes a chat (`farewell`). Per reply:
  0.6  an expected cue is present
  0.2  no forbidden phrase (a forbidden phrase zeroes the case)
  0.2  clean: no persona violation, no repetition loop, not empty
The set score multiplies the mean by a diversity factor, so a model that gives one safe line to
every prompt (the v5 collapse) cannot score well. Generation is greedy with 3-gram blocking,
matching the served decoder minus sampling, so scores are reproducible.

  python -m Tantra.eval.reply_eval runs/serve-64k-v0 [checkpoint=base.pt] [--bundle B]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from Tantra.data.persona_rules import violations

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / 'docs' / 'data' / 'reply_eval.v1.json'


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


def main():
    from Tantra.config import ModelConfig
    from Tantra.model.tantra_model import TantraModel
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


if __name__ == '__main__':
    main()
