"""Expand topic packs into a bulk conversation queue (PENDING, needs bulk review).

Each pack has openers, first replies (all asking the same follow-up question) and
(answer, reply) pairs that fit that question. Per pack:
  openers x first           -> single-turn conversations
  openers x first x pairs   -> two-round conversations
All rows from one pack share a family, so a topic never spans train/val/test.
Review: python -m Tantra.data.bulk_review filter|sample|approve (docs/BULK_DATA_POLICY.md).
Run: .venv\\Scripts\\python.exe scripts\\conversation_bulk_v1.py
"""
import hashlib, itertools, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from bulk_packs import greeting_farewell, smalltalk_daily, smalltalk_life, smalltalk_more
from compare_chat_pilot import FRESH

HERE = ROOT / 'runs' / 'conversation-bulk-v1'
SOURCE = {'id': 'tantra-conversation-bulk-v1', 'revision': '1', 'license': 'MIT', 'kind': 'original_ai_authored_templated'}
ASCII = str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"'})


def text(s):
    return s.translate(ASCII)


def main():
    packs = greeting_farewell.PACKS + smalltalk_daily.PACKS + smalltalk_life.PACKS + smalltalk_more.PACKS
    keys = [(p['expert'], p['topic']) for p in packs]
    assert len(keys) == len(set(keys)), 'duplicate pack topic'
    fresh = {q.lower() for q in FRESH}
    rows = []
    for pack in packs:
        assert len(pack['first']) >= 2 and len(pack['pairs']) >= 4, pack['topic']
        family = f"{pack['expert']}:bulk-v1:{pack['topic']}"
        for opener in pack['openers']:
            assert text(opener).lower() not in fresh, 'held-out test prompt in pack: ' + opener
        turn_sets = [[o, f] for o, f in itertools.product(pack['openers'], pack['first'])]
        turn_sets += [[o, f, a, r] for o, f, (a, r) in itertools.product(pack['openers'], pack['first'], pack['pairs'])]
        for turns in turn_sets:
            msgs = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': text(t)} for i, t in enumerate(turns)]
            rid = hashlib.sha256(json.dumps([pack['expert'], msgs], ensure_ascii=False).encode()).hexdigest()
            rows.append({'record_id': rid, 'schema_version': 1, 'expert': pack['expert'], 'language': 'en',
                         'family_id': family, 'template_id': f"bulk-v1:{pack['topic']}:{len(msgs) // 2}-round",
                         'messages': msgs, 'source': SOURCE, 'review': {'status': 'pending', 'reviewer': None}})
    queue = HERE / 'queue'; queue.mkdir(parents=True, exist_ok=True)
    path = queue / 'bulk-drafts.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (queue / 'index.json').write_text(json.dumps({'files': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
                                                  'review_status': 'pending', 'packs': len(packs)}, indent=2) + '\n', encoding='utf-8')
    by = {}
    for r in rows: by[r['expert']] = by.get(r['expert'], 0) + 1
    print(json.dumps({'packs (families)': len(packs), 'rows': len(rows), 'by_expert': by, 'queue': str(queue)}, indent=2))


if __name__ == '__main__':
    main()
