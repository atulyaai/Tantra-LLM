"""Bounded expert expansion: one isolated candidate per invocation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from .expert_workbench import DEFAULT_OUT, PRIORITY, catalog
from Tantra.config import ROOT


def select_expert(manifest, attempted):
    specs = catalog()
    order = list(dict.fromkeys(PRIORITY + ['math', 'json_schema', 'csv_export'] + list(specs)))
    for name in order:
        spec = specs[name]
        if name.startswith('lang_') and name not in ('lang_hi', 'lang_en'):
            continue
        if any(language not in ('hi', 'en', 'hinglish') for language in spec['languages']):
            continue
        if name not in attempted and manifest['experts'].get(name, {}).get('state') == 'data_ready':
            return name
    return None


def run_one(bundle=DEFAULT_OUT, model_dir=ROOT/'Model', output=ROOT/'runs'/'expert-growth', steps=100):
    bundle, output = Path(bundle), Path(output)
    manifest = json.loads((bundle/'prepared'/'manifest.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive file creation prevents concurrent invocations training two candidates.
    lock = output/'growth.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(str(os.getpid()))
    try:
        attempted = {p.name for p in output.iterdir() if p.is_dir()}
        name = select_expert(manifest, attempted)
        if name is None:
            return {'state': 'blocked', 'reason': 'No unattempted data-ready expert in initial language scope'}
        candidate = output/name
        candidate.mkdir()  # Retain failed attempts too; retries require explicit review.
        from .train_prepared import train_one
        try:
            result = train_one(bundle, name, model_dir, candidate/'candidate', steps=steps, rank=8, threads=4)
            event = {'expert': name, 'state': 'candidate_trained', 'deployed': False, 'result': result,
                     'next_gate': 'Independent answer evaluation, router macro-F1 >=0.90 and conversation regression checks'}
        except Exception as exc:
            event = {'expert': name, 'state': 'failed', 'deployed': False,
                     'error': type(exc).__name__ + ': ' + str(exc)}
            (candidate/'event.json').write_text(json.dumps(event, indent=2), encoding='utf-8')
            raise
        (candidate/'event.json').write_text(json.dumps(event, indent=2), encoding='utf-8')
        return event
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--model-dir', type=Path, default=ROOT/'Model')
    parser.add_argument('--output', type=Path, default=ROOT/'runs'/'expert-growth')
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run_one(args.out, args.model_dir, args.output, args.steps), indent=2))

if __name__ == '__main__':
    main()
