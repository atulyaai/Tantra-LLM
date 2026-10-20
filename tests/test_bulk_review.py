import csv
import hashlib
import json
import pytest
from Tantra.data.bulk_review import approve, draw_sample, run_filters
from Tantra.data.persona_rules import violations


def record(i, user, reply, expert='smalltalk', family=None):
    messages = [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': reply}]
    return {'record_id': f'{i:064x}', 'schema_version': 1, 'expert': expert, 'language': 'en',
            'family_id': family or f'{expert}:bulk:{i}', 'template_id': 'bulk:t', 'messages': messages,
            'source': {'id': 'test', 'revision': '1', 'license': 'MIT', 'kind': 'original_ai_authored'},
            'review': {'status': 'pending', 'reviewer': None}}


def make_queue(tmp_path, rows):
    queue = tmp_path / 'queue'; queue.mkdir()
    path = queue / 'drafts.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (queue / 'index.json').write_text(json.dumps({'files': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}))
    return queue


def read_sample(queue):
    with (queue / 'spot-check.csv').open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_sample(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return path


def mark(queue, value):
    rows = read_sample(queue)
    for i, r in enumerate(rows):
        r['ok (Y/N)'] = value(i, r)
    return write_sample(queue / 'spot-check.csv', rows)


def family_queue(tmp_path, families=6, per=10):
    rows = [record(f * per + j, f'I like topic {f} take {j}.', f'Nice! Tell me more about topic {f}, take {j}?',
                   family=f'smalltalk:bulk:topic{f}') for f in range(families) for j in range(per)]
    return make_queue(tmp_path, rows)


def test_persona_rules_allow_asked_and_block_unasked():
    ask = lambda u, r: violations([{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}])
    assert ask('Hi!', "Hi! I'm an AI.") == ['unasked AI mention']
    assert ask('Are you a bot?', "Yes, I'm an AI!") == []
    assert ask('Nice day!', 'Atul Vij made me.') == ['unasked creator mention']
    assert ask('Who made you?', 'Atul Vij made me.') == []
    assert ask('Hello', 'How can I help you?') == ['service wording']
    assert ask('Hi', "I'm a chatbot made by Atul.") == ['unasked AI mention', 'unasked creator mention']


def test_persona_rules_check_assistant_turns_after_a_system_message():
    msgs = [{'role': 'system', 'content': 'Be friendly.'}, {'role': 'user', 'content': 'Hi'},
            {'role': 'assistant', 'content': "I'm an AI from Atulya AI."}]
    assert violations(msgs) == ['unasked AI mention', 'unasked creator mention']


def test_one_early_question_does_not_license_every_later_reply():
    msgs = [{'role': 'user', 'content': 'Are you a bot?'}, {'role': 'assistant', 'content': "Yes, I'm an AI!"},
            {'role': 'user', 'content': 'Nice. It is raining here.'}, {'role': 'assistant', 'content': "Being an AI, I'm an AI who likes rain."}]
    assert violations(msgs) == ['unasked AI mention']


def test_filters_drop_bad_rows(tmp_path):
    rows = [record(1, 'It is raining.', 'Chai weather! Do you like rain?'),
            record(2, 'It is raining.', 'Chai weather! Do you like rain?'),       # duplicate
            record(3, 'Hello', "Hi! I'm an AI."),                                   # persona
            record(4, 'Hello', 'नमस्ते!'),                                          # script
            record(5, 'Hello', 'x' * 500)]                                          # length
    passed, report = run_filters(make_queue(tmp_path, rows))
    assert [r['record_id'] for r in passed] == [rows[0]['record_id']]
    assert report['rejected'] == 4


def test_stratified_sample_covers_every_family_and_rejection_drops_the_family(tmp_path):
    queue = family_queue(tmp_path)
    result = draw_sample(queue, n=50)
    sample = read_sample(queue)
    assert result['sample'] == 50 and result['families_covered'] == 6
    assert len({r['conversation'].split('topic ')[1][0] for r in sample}) == 6
    bad = sample[0]['conversation'].split('topic ')[1][0]
    path = mark(queue, lambda i, r: 'N' if i == 0 else 'Y')
    with pytest.raises(ValueError, match='below'):
        approve(queue, path, tmp_path / 'strict', 'Owner', 'checked', min_rate=0.99)
    result = approve(queue, path, tmp_path / 'out', 'Owner', 'checked sample', min_rate=0.95)
    assert result['approved'] == 50 and result['families_dropped'] == [f'smalltalk:bulk:topic{bad}']
    saved = [json.loads(l) for l in (tmp_path / 'out/conversation/smalltalk.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(row['family_id'] != f'smalltalk:bulk:topic{bad}' for row in saved)
    review = saved[0]['review']
    assert review['review_method'] == 'owner_sample_spot_check' and review['sample_size'] == 50
    assert 'spot-checked 50 of 60' in review['reviewer'] and review['fluent_language_review'] is False


def test_hand_picked_or_tiny_sample_is_refused(tmp_path):
    queue = family_queue(tmp_path)
    draw_sample(queue, n=50)
    rows = read_sample(queue)
    for r in rows: r['ok (Y/N)'] = 'Y'
    tiny = write_sample(tmp_path / 'tiny.csv', rows[:1])
    with pytest.raises(ValueError, match='recorded draw'):
        approve(queue, tiny, tmp_path / 'out', 'Owner', 'why')
    assert not (tmp_path / 'out').exists()


def test_unmarked_sample_is_refused(tmp_path):
    queue = family_queue(tmp_path)
    draw_sample(queue)
    with pytest.raises(ValueError, match='marked'):
        approve(queue, queue / 'spot-check.csv', tmp_path / 'out', 'Owner', 'why')
    assert not (tmp_path / 'out').exists()
