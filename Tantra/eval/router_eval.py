"""Train the n-gram router on TRAIN-split prompts and measure it on held-out splits (roadmap M5).

Routing quality is reported separately from answer quality: top-1 accuracy, top-2 recall,
the share of requests sent to base only, and a confusion matrix. Experts listed in
--none are merged into a `none` class: requests routed there use the base model alone.

  python -m Tantra.eval.router_eval --bundle Datasets/workbench-chat-en-pilot-v3 \\
      --experts identity_self identity_capabilities identity_limitations \\
      --none greeting smalltalk farewell --split test --out runs/router/v3.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from Tantra.model.ngram_router import NgramRouter, features, train_ngram_router

NONE = 'none'


def first_user(row: Dict) -> str:
    # Compatibility name: route the CURRENT request, as TantraEngine does.
    from Tantra.data.expert_dataset import first_user_text
    return first_user_text(row)


def load_bundle_rows(bundle: Path, experts: Iterable[str]) -> Dict[str, List[Dict]]:
    from Tantra.data.train_prepared import load_split
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


def main():
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


if __name__ == '__main__':
    main()
