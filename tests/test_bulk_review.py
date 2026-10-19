import csv
import hashlib
import json
import pytest
from Tantra.data.bulk_review import approve, draw_sample, run_filters
from Tantra.data.persona_rules import violations


def record(i, user, reply, expert='smalltalk'):
    messages = [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': reply}]
    return {'record_id': f'{i:064x}', 'schema_version': 1, 'expert': expert, 'language': 'en',
            'family_id': f'{expert}:bulk:{i}', 'template_id': 'bulk:t', 'messages': messages,
            'source': {'id': 'test', 'revision': '1', 'license': 'MIT', 'kind': 'original_ai_authored'},
            'review': {'status': 'pending', 'reviewer': None}}


def make_queue(tmp_path, rows):
    queue = tmp_path / 'queue'; queue.mkdir()
    path = queue / 'drafts.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (queue / 'index.json').write_text(json.dumps({'files': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}))
    return queue


def mark(queue, value):
    path = queue / 'spot-check.csv'
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r['ok (Y/N)'] = value(i)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return path


def test_persona_rules_allow_asked_and_block_unasked():
    ask = lambda u, r: violations([{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}])
    assert ask('Hi!', "Hi! I'm an AI.") == ['unasked AI mention']
    assert ask('Are you a bot?', "Yes, I'm an AI!") == []
    assert ask('Nice day!', 'Atul Vij made me.') == ['unasked creator mention']
    assert ask('Who made you?', 'Atul Vij made me.') == []
    assert ask('Hello', 'How can I help you?') == ['service wording']


def test_filters_drop_bad_rows(tmp_path):
    rows = [record(1, 'It is raining.', 'Chai weather! Do you like rain?'),
            record(2, 'It is raining.', 'Chai weather! Do you like rain?'),       # duplicate
            record(3, 'Hello', "Hi! I'm an AI."),                                   # persona
            record(4, 'Hello', 'नमस्ते!'),                                          # script
            record(5, 'Hello', 'x' * 500)]                                          # length
    passed, report = run_filters(make_queue(tmp_path, rows))
    assert [r['record_id'] for r in passed] == [rows[0]['record_id']]
    assert report['rejected'] == 4


def test_spot_check_approval_records_method_and_drops_rejected(tmp_path):
    rows = [record(i, f'I like topic {i}.', f'Nice! Tell me more about topic {i}?') for i in range(40)]
    queue = make_queue(tmp_path, rows)
    assert draw_sample(queue, n=20)['sample'] == 20
    sample = mark(queue, lambda i: 'N' if i == 0 else 'Y')
    with pytest.raises(ValueError, match='below'):
        approve(queue, sample, tmp_path / 'strict', 'Owner', 'checked', min_rate=0.99)
    result = approve(queue, sample, tmp_path / 'out', 'Owner', 'checked sample', min_rate=0.9)
    assert result['approved'] == 39 and result['dropped_in_sample'] == 1
    saved = [json.loads(l) for l in (tmp_path / 'out/conversation/smalltalk.jsonl').read_text(encoding='utf-8').splitlines()]
    review = saved[0]['review']
    assert review['review_method'] == 'owner_sample_spot_check' and review['sample_size'] == 20
    assert 'spot-checked 20 of 40' in review['reviewer'] and review['fluent_language_review'] is False


def test_unmarked_sample_is_refused(tmp_path):
    queue = make_queue(tmp_path, [record(i, f'Topic {i}?', f'Sure, topic {i}!') for i in range(5)])
    draw_sample(queue, n=3)
    with pytest.raises(ValueError, match='marked'):
        approve(queue, queue / 'spot-check.csv', tmp_path / 'out', 'Owner', 'why')
