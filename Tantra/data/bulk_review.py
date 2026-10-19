"""Bulk review for large own-authored conversation batches (owner policy, 2026-10-19).

A batch of generated drafts counts as reviewed only when:
1. every row passes the automatic filters (schema, persona rules, language, length, duplicates), and
2. the owner marks a seeded random sample and approves at least --min-rate of it.
Rows the owner rejects in the sample are dropped. Every approved row records exactly how it
was reviewed (review_method, sample size/rate, filters), so it is never mistaken for row-by-row review.

  python -m Tantra.data.bulk_review filter  --queue Q
  python -m Tantra.data.bulk_review sample  --queue Q --n 50
  python -m Tantra.data.bulk_review approve --queue Q --sample Q/spot-check.csv --reviewer NAME --rationale TEXT --output DIR
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from .expert_workbench import catalog, validate_record
from .persona_rules import violations
from .public_sources import sha256

FILTERS = ['schema', 'persona', 'english_script', 'length', 'exact_duplicate']
MAX_CHARS = 400
MAX_TURNS = 8
NON_LATIN = re.compile(r'[ऀ-ॿ؀-ۿЀ-ӿ一-鿿]')


def message_hash(row):
    return hashlib.sha256(json.dumps([row['expert'], row['messages']], ensure_ascii=False).encode()).hexdigest()


def check(row, seen):
    """Reasons a row fails the automatic filters (empty list = pass)."""
    reasons = []
    try:
        validate_record({**row, 'review': {'status': 'reviewed', 'reviewer': 'filter-check', 'rationale': 'filter-check'}}, row['expert'])
    except (ValueError, KeyError, TypeError) as exc:
        reasons.append('schema: ' + str(exc))
    msgs = row.get('messages', [])
    reasons += ['persona: ' + v for v in violations(msgs)]
    if row.get('language') == 'en' and any(NON_LATIN.search(m['content']) for m in msgs):
        reasons.append('english_script: non-Latin text in an English row')
    if len(msgs) > MAX_TURNS or any(len(m['content']) > MAX_CHARS for m in msgs):
        reasons.append('length')
    key = message_hash(row)
    if key in seen:
        reasons.append('exact_duplicate')
    seen.add(key)
    return reasons


def load_queue(queue):
    queue = Path(queue)
    index = json.loads((queue / 'index.json').read_text(encoding='utf-8'))
    rows = []
    for name, expected in sorted(index['files'].items()):
        if sha256(queue / name) != expected:
            raise ValueError('Queue checksum mismatch: ' + name)
        rows += [json.loads(l) for l in (queue / name).read_text(encoding='utf-8').splitlines() if l.strip()]
    return rows


def run_filters(queue):
    seen = set(); passed = []; rejected = []
    for row in load_queue(queue):
        reasons = check(row, seen)
        (rejected.append({'record_id': row.get('record_id'), 'reasons': reasons}) if reasons else passed.append(row))
    report = {'rows': len(passed) + len(rejected), 'passed': len(passed), 'rejected': len(rejected),
              'filters': FILTERS, 'rejections': rejected}
    (Path(queue) / 'filter-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return passed, report


def draw_sample(queue, n=50, seed=0):
    passed, report = run_filters(queue)
    if not passed:
        raise ValueError('No rows passed the filters')
    sample = random.Random(seed).sample(passed, min(n, len(passed)))
    path = Path(queue) / 'spot-check.csv'
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['record_id', 'expert', 'conversation', 'ok (Y/N)', 'comment'])
        for row in sample:
            convo = '\n'.join(('USER: ' if m['role'] == 'user' else 'TANTRA: ') + m['content'] for m in row['messages'])
            w.writerow([row['record_id'], row['expert'], convo, '', ''])
    return {'sample': len(sample), 'passed': len(passed), 'rejected_by_filters': report['rejected'], 'path': str(path)}


def approve(queue, sample_csv, output, reviewer, rationale, min_rate=0.95):
    output = Path(output)
    if output.exists():
        raise ValueError('Reviewed export must be a new directory')
    if not (reviewer or '').strip() or not (rationale or '').strip():
        raise ValueError('A real reviewer name and rationale are required')
    passed, report = run_filters(queue)
    ids = {row['record_id'] for row in passed}
    with Path(sample_csv).open(newline='', encoding='utf-8-sig') as f:
        marks = {r['record_id']: (r.get('ok (Y/N)') or '').strip().upper() for r in csv.DictReader(f)}
    if not marks or any(m not in ('Y', 'N') for m in marks.values()):
        raise ValueError('Every sampled row must be marked Y or N')
    if set(marks) - ids:
        raise ValueError('Sample contains rows that are not in the filtered batch')
    rate = sum(m == 'Y' for m in marks.values()) / len(marks)
    if rate < min_rate:
        raise ValueError(f'Spot-check approval {rate:.0%} is below the required {min_rate:.0%}; fix the generator and redraw')
    decision = sha256(Path(sample_csv))
    review = {'status': 'reviewed', 'reviewer': f'{reviewer.strip()} (spot-checked {len(marks)} of {len(passed)}, {rate:.0%} approved)',
              'rationale': rationale.strip(), 'review_method': 'owner_sample_spot_check', 'sample_size': len(marks),
              'sample_approval_rate': rate, 'batch_size': len(passed), 'filters': FILTERS,
              'fluent_language_review': False, 'domain_review': False, 'decision_sha256': decision}
    output.mkdir(parents=True)
    counts = {}
    for row in passed:
        if marks.get(row['record_id']) == 'N':
            continue
        if row['language'] != 'en':
            raise ValueError('Bulk spot-check approval is English-only; Hindi/Hinglish need fluent review')
        expert = row['expert']; domain = catalog()[expert]['domain']
        record = validate_record({**row, 'review': review}, expert)
        (output / domain).mkdir(exist_ok=True)
        with (output / domain / f'{expert}.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        counts[expert] = counts.get(expert, 0) + 1
    return {'approved': sum(counts.values()), 'by_expert': counts, 'sample_approval_rate': rate,
            'dropped_in_sample': sum(m == 'N' for m in marks.values()), 'rejected_by_filters': report['rejected']}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('filter'); f.add_argument('--queue', type=Path, required=True)
    s = sub.add_parser('sample'); s.add_argument('--queue', type=Path, required=True)
    s.add_argument('--n', type=int, default=50); s.add_argument('--seed', type=int, default=0)
    a = sub.add_parser('approve'); a.add_argument('--queue', type=Path, required=True); a.add_argument('--sample', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True); a.add_argument('--reviewer', required=True)
    a.add_argument('--rationale', required=True); a.add_argument('--min-rate', type=float, default=0.95)
    x = p.parse_args()
    if x.command == 'filter':
        result = {k: v for k, v in run_filters(x.queue)[1].items() if k != 'rejections'}
    elif x.command == 'sample':
        result = draw_sample(x.queue, x.n, x.seed)
    else:
        result = approve(x.queue, x.sample, x.output, x.reviewer, x.rationale, x.min_rate)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
