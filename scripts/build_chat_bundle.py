"""Merge reviewed conversation sources into a NEW chat bundle and prepare frozen splits.

  .venv\\Scripts\\python.exe scripts\\build_chat_bundle.py --out Datasets/workbench-chat-en-v4 \\
      --sources Datasets/workbench-chat-en-pilot-v3/incoming/conversation Datasets/conversation-reviewed-bulk-v1/conversation

Every source row must already be reviewed (row-by-row or bulk spot-check). Rows that break the
persona rules (Tantra/data/persona_rules.py) are dropped and listed in <out>/persona-dropped.json,
so an approved-but-old reply never trains the model against the current persona. Exact duplicate
conversations are kept once. Never writes into an existing bundle.
"""
import argparse, json, subprocess, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Tantra.data.persona_rules import violations


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--sources', type=Path, nargs='+', required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; bundles are immutable, choose a new version')
    rows, dropped, seen, counts = {}, [], set(), Counter()
    for src in a.sources:
        for path in sorted(src.glob('*.jsonl')):
            for line in path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                if row.get('review', {}).get('status') != 'reviewed':
                    raise SystemExit(f'Unreviewed row in {path}: {row.get("family_id")}')
                problems = violations(row['messages'])
                if problems:
                    dropped.append({'source': str(path), 'family_id': row['family_id'], 'why': problems,
                                    'messages': row['messages']}); continue
                key = json.dumps([row['expert'], row['messages']], ensure_ascii=False)
                if key in seen:
                    continue
                seen.add(key); rows.setdefault(row['expert'], []).append(line); counts[row['expert']] += 1
    incoming = a.out / 'incoming' / 'conversation'; incoming.mkdir(parents=True)
    for expert, lines in rows.items():
        (incoming / f'{expert}.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (a.out / 'persona-dropped.json').write_text(json.dumps(dropped, ensure_ascii=False, indent=2), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'Tantra.data.expert_workbench', '--out', str(a.out), 'prepare'], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    print(json.dumps({'bundle': str(a.out), 'rows_by_expert': dict(counts), 'persona_dropped': len(dropped)}, indent=2))


if __name__ == '__main__':
    main()
