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


def test_content_change_with_stable_record_ids_invalidates_draw(tmp_path):
    queue = family_queue(tmp_path)
    draw_sample(queue)
    marked = mark(queue, lambda i, r: 'Y')
    # Re-indexing the queue is not a review: unchanged caller IDs must not hide new text.
    path = queue / 'drafts.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    rows[0]['messages'][-1]['content'] = 'A different otherwise valid answer.'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    (queue / 'index.json').write_text(json.dumps({'files': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}))
    with pytest.raises(ValueError, match='batch changed'):
        approve(queue, marked, tmp_path / 'out', 'Owner', 'checked')
    assert not (tmp_path / 'out').exists()


def test_displayed_sample_must_match_the_actual_queue_text_and_expert(tmp_path):
    queue = family_queue(tmp_path)
    draw_sample(queue)
    rows = read_sample(queue)
    for row in rows:
        row['ok (Y/N)'] = 'Y'
    rows[0]['conversation'] = 'USER: A safer question\nTANTRA: A safer answer'
    marked = write_sample(tmp_path / 'edited.csv', rows)
    with pytest.raises(ValueError, match='Displayed sample'):
        approve(queue, marked, tmp_path / 'edited-out', 'Owner', 'checked')
    rows[0] = {**read_sample(queue)[0], 'ok (Y/N)': 'Y', 'expert': 'greeting'}
    marked = write_sample(tmp_path / 'wrong-expert.csv', rows)
    with pytest.raises(ValueError, match='Displayed sample'):
        approve(queue, marked, tmp_path / 'wrong-expert-out', 'Owner', 'checked')


def test_duplicate_sample_or_queue_ids_are_refused(tmp_path):
    queue = family_queue(tmp_path)
    draw_sample(queue)
    rows = read_sample(queue)
    for row in rows:
        row['ok (Y/N)'] = 'Y'
    marked = write_sample(tmp_path / 'duplicates.csv', rows + [rows[0]])
    with pytest.raises(ValueError, match='Duplicate record IDs'):
        approve(queue, marked, tmp_path / 'out', 'Owner', 'checked')
    nested = tmp_path / 'nested'; nested.mkdir()
    duplicate = record(1, 'Hello', 'Hi!')
    with pytest.raises(ValueError, match='unique'):
        run_filters(make_queue(nested, [duplicate, duplicate]))


@pytest.mark.parametrize('rate', [0.0, 0.94, 1.01, float('nan'), float('inf')])
def test_owner_approval_threshold_cannot_be_weakened(tmp_path, rate):
    with pytest.raises(ValueError, match='between 0.95 and 1.0'):
        approve(tmp_path, tmp_path / 'missing.csv', tmp_path / 'out', 'Owner', 'checked', min_rate=rate)


def test_identity_safety_and_external_rows_stay_out_of_bulk_review(tmp_path):
    rows = [record(1, 'Who made you?', 'Atul Vij made me.', expert='identity_self'),
            record(2, 'Are you a machine?', 'I am an AI.', expert='smalltalk'),
            record(3, 'Who made you?', 'Atul Vij made me.', expert='smalltalk'),
            record(4, 'Should I do something dangerous?', 'Please avoid doing that.', expert='safety_refusal'),
            record(5, 'A normal question', 'A normal answer')]
    rows[-1]['source']['kind'] = 'community_conversation'
    passed, report = run_filters(make_queue(tmp_path, rows))
    assert not passed and report['rejected'] == 5
    assert all(any(reason.startswith('bulk_review_scope:') for reason in item['reasons']) for item in report['rejections'])


def test_old_id_only_draw_requires_fresh_review(tmp_path):
    queue = family_queue(tmp_path)
    draw_sample(queue)
    marked = mark(queue, lambda i, r: 'Y')
    path = queue / 'spot-check.draw.json'
    draw = json.loads(path.read_text())
    draw.pop('version')
    path.write_text(json.dumps(draw))
    with pytest.raises(ValueError, match='Legacy sample'):
        approve(queue, marked, tmp_path / 'out', 'Owner', 'checked')
