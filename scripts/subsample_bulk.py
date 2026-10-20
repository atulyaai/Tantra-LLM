"""Reduce template repetition in an approved bulk export without changing any row.

Bulk-v1 (9,240 rows) has only ~800 distinct replies, each seen ~10x per epoch; a model trained
on it collapsed into stitched template fragments. This keeps, per family, rows until every
assistant reply in them has appeared at most --max-repeats times, preferring rows that add
new replies. Review fields are kept as approved; the selection rule is recorded in a
`subsample` note on each row.

  .venv\\Scripts\\python.exe scripts\\subsample_bulk.py --src Datasets/conversation-reviewed-bulk-v1 --out Datasets/conversation-reviewed-bulk-v1-dedup2
"""
import argparse, json, random
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-repeats', type=int, default=2); p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists')
    rng = random.Random(a.seed); kept_total = seen_total = 0
    for path in sorted(a.src.rglob('*.jsonl')):
        rows = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
        seen_total += len(rows)
        by_family = {}
        for row in rows:
            by_family.setdefault(row['family_id'], []).append(row)
        kept = []
        for family, items in sorted(by_family.items()):
            rng.shuffle(items)
            items.sort(key=lambda r: -len(r['messages']))  # prefer multi-turn rows: more distinct replies each
            uses = Counter()
            for row in items:
                replies = [m['content'] for m in row['messages'] if m['role'] == 'assistant']
                if all(uses[r] < a.max_repeats for r in replies):
                    uses.update(replies)
                    kept.append({**row, 'subsample': f'scripts/subsample_bulk.py max_repeats={a.max_repeats} seed={a.seed}'})
        out = a.out / path.relative_to(a.src); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept), encoding='utf-8')
        kept_total += len(kept)
        print(f'{path.relative_to(a.src)}: {len(rows)} -> {len(kept)}')
    print(f'total {seen_total} -> {kept_total}')


if __name__ == '__main__':
    main()
