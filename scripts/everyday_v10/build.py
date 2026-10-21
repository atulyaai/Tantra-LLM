"""Build the everyday-v10 bulk queue (PENDING owner spot-check under docs/BULK_DATA_POLICY.md).

Every row is a distinct situation with its own hand-written reply (no template recombination).
Rows are grouped into families of up to 5 consecutive rows within a theme part, which keeps
related situations in one split and keeps the stratified spot-check to ~one row per family.
Checks: unique user turns and replies, no held-out reply-eval prompt, no exact overlap with
earlier approved conversation data, persona rules.
Run: .venv\\Scripts\\python.exe scripts\\everyday_v10\\build.py
"""
import hashlib, importlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from Tantra.data.persona_rules import violations
from Tantra.eval.reply_eval import load_cases

PARTS = ['p1_weather', 'p2_food', 'p3_feelings', 'p4_work_study', 'p5_hobbies', 'p6_daily', 'p7_chat']
OUT = ROOT / 'runs' / 'conversation-bulk-v10' / 'queue'
SOURCE = {'id': 'tantra-conversation-everyday-v10', 'revision': '1', 'license': 'MIT', 'kind': 'original_ai_authored'}
FAMILY_SIZE = 5


def earlier_user_turns():
    seen = set()
    for path in (ROOT / 'Datasets').glob('conversation-reviewed-*/**/*.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            for m in json.loads(line).get('messages', []):
                if m['role'] == 'user':
                    seen.add(m['content'].strip().lower())
    return seen


def main():
    held_out = {c['prompt'].strip().lower() for c in load_cases()}
    earlier = earlier_user_turns()
    users, replies, rows, skipped = set(), set(), [], []
    for part in PARTS:
        for i, item in enumerate(importlib.import_module(part).ROWS):
            expert, turns = (item[0], list(item[1])) if isinstance(item[1], list) else (item[0], [item[1], item[2]])
            assert len(turns) % 2 == 0, (part, i)
            first = turns[0].strip().lower()
            assert first not in held_out, f'held-out eval prompt reused: {turns[0]}'
            if first in earlier:
                skipped.append(turns[0]); continue
            assert first not in users, f'duplicate situation: {turns[0]}'
            users.add(first)
            for r in turns[1::2]:
                assert r not in replies, f'duplicate reply: {r}'
                replies.add(r)
            msgs = [{'role': 'user' if j % 2 == 0 else 'assistant', 'content': t} for j, t in enumerate(turns)]
            assert not violations(msgs), (turns, violations(msgs))
            rid = hashlib.sha256(json.dumps([expert, msgs], ensure_ascii=False).encode()).hexdigest()
            rows.append({'record_id': rid, 'schema_version': 1, 'expert': expert, 'language': 'en',
                         'family_id': f'{expert}:everyday-v10:{part}:{i // FAMILY_SIZE:02d}',
                         'template_id': f'everyday-v10:{part}', 'messages': msgs, 'source': SOURCE,
                         'review': {'status': 'pending', 'reviewer': None}})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'bulk-drafts.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (OUT / 'index.json').write_text(json.dumps({'files': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
                                                'review_status': 'pending'}, indent=2) + '\n', encoding='utf-8')
    by = {}
    for r in rows: by[r['expert']] = by.get(r['expert'], 0) + 1
    print(json.dumps({'rows': len(rows), 'distinct_replies': len(replies), 'families': len({r['family_id'] for r in rows}),
                      'multi_turn': sum(len(r['messages']) > 2 for r in rows), 'by_expert': by,
                      'skipped_already_approved': skipped}, indent=2))


if __name__ == '__main__':
    main()
