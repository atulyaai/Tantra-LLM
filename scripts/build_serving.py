"""Assemble a CPU serving folder: frozen base + expert adapters + n-gram router.

  .venv\\Scripts\\python.exe scripts\\build_serving.py --base runs/serve-64k-v0 \\
      --experts runs/experts-64k-v1/identity_self runs/experts-64k-v1/greeting ... \\
      --bundle Datasets/workbench-chat-en-pilot-v3 --none smalltalk --out runs/serve-64k-v1

The output base.pt is the source base weights with the experts registered in the router;
manifest.json records the SOURCE base/tokenizer SHA-256 that every adapter was trained
against, so the engine can refuse adapters built for anything else. Adapters whose
recorded base hash differs from the source are rejected here too. Never overwrites a folder.
"""
import argparse, json, shutil, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Tantra.data.public_sources import sha256
from Tantra.eval.router_eval import run as router_run
from Tantra.model.tantra_model import TantraModel


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', type=Path, required=True); p.add_argument('--experts', type=Path, nargs='+', required=True)
    p.add_argument('--bundle', type=Path, required=True); p.add_argument('--none', nargs='*', default=[])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; serving folders are never overwritten')
    base_hash, tok_hash = sha256(a.base / 'base.pt'), sha256(a.base / 'tokenizer.json')
    adapters = {}
    for d in a.experts:
        files = list(d.glob('*.pt'))
        if len(files) != 1:
            raise SystemExit(f'{d}: expected one adapter file, found {len(files)}')
        payload = torch.load(files[0], map_location='cpu', weights_only=True)
        if payload.get('base_sha256') != base_hash or payload.get('tokenizer_sha256') != tok_hash:
            raise SystemExit(f'{files[0]} was trained against a different base/tokenizer')
        adapters[files[0].stem] = (files[0], payload)
    model = TantraModel.load(a.base / 'base.pt')
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for name in adapters:
        model.router.add_expert(name)
    a.out.mkdir(parents=True); (a.out / 'experts').mkdir()
    ck = torch.load(a.base / 'base.pt', map_location='cpu', weights_only=False)
    state = {k: v for k, v in model.state_dict().items() if 'lora_' not in k}
    assert all(torch.equal(state[k], v) for k, v in before.items() if k in state and 'router' not in k), 'base weights changed'
    torch.save({'config': ck['config'], 'state': state, 'router_names': list(adapters),
                'expert_ranks': {n: int(p['rank']) for n, (_, p) in adapters.items()}}, a.out / 'base.pt')
    shutil.copy2(a.base / 'tokenizer.json', a.out / 'tokenizer.json')
    for name, (src, _) in adapters.items():
        shutil.copy2(src, a.out / 'experts' / f'{name}.pt')
    report = router_run(a.bundle, list(adapters), a.none, 'test', a.out / 'router_ngram.pt')
    manifest = {'version':2, 'artifacts': {name:sha256(a.out/name) for name in ('base.pt','tokenizer.json','router_ngram.pt')}, 'base_sha256': base_hash, 'tokenizer_sha256': tok_hash, 'source_base': str(a.base),
                'experts': {n: {'file': f'experts/{n}.pt', 'rank': int(p['rank']), 'sha256': sha256(a.out / 'experts' / f'{n}.pt')}
                            for n, (_, p) in adapters.items()},
                'router': {'names': report['names'], 'none_experts': list(a.none), 'val': {k: report['val'].get(k) for k in ('rows', 'top1_accuracy', 'top2_recall')},
                           'test': {k: report['test'].get(k) for k in ('rows', 'top1_accuracy', 'top2_recall')}},
                'status': 'pilot: experts are undersized_pilot_unvalidated; not a release'}
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
