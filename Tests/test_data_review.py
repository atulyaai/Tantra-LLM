"""
Growth, workbench, bulk/manual review sheets and conversation quality.

Merged: test_auto_growth.py, test_expert_workbench.py, test_bulk_review.py, test_review_sheets.py, test_instruction_candidates.py, test_conversation_quality.py.
"""

# ──────────────────────────────────────────────────────────────────────  test_auto_growth.py  ──────────────────────────────────────────────────────────────────────

from Tantra.train import select_expert
from Tantra.train import catalog


def test_growth_requires_ready_data_and_skips_attempts():
    manifest = {'experts': {name: {'state': 'needs_data'} for name in catalog()}}
    manifest['experts']['math']['state'] = 'data_ready'
    assert select_expert(manifest, set()) == 'math'
    assert select_expert(manifest, {'math'}) is None


def test_growth_defers_other_languages():
    manifest = {'experts': {'lang_fr': {'state': 'data_ready'}}}
    assert select_expert(manifest, set()) is None

# ──────────────────────────────────────────────────────────────────────  test_expert_workbench.py  ──────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

import pytest

from Tantra.train import catalog, generated_rows, initialize, messages_of, prepare, seed, training_plan, validate_record, verify_auto


def test_all_164_scopes_and_real_parameter_budgets(tmp_path):
    specs=catalog()
    assert len(specs)==164
    assert 'arithmetic computation belongs to math' in specs['gk_math']['goal']
    assert specs['math']['adapter_parameters_small']==37888*64
    assert specs['lang_hi']['languages']==['hi']
    initialize(tmp_path/'data',tmp_path/'docs'/'experts')
    assert len(list((tmp_path/'docs'/'experts').rglob('*.md')))==164


def test_placeholder_and_malformed_rows_rejected():
    with pytest.raises(ValueError,match='placeholder'):
        validate_record({'text':'Sample 1 of 10000'},'math',False)
    for row in (None,[],{}, {'messages':[None]}):
        with pytest.raises(ValueError): messages_of(row)


def test_generated_answers_are_verified_and_owned_once():
    rows=list(generated_rows(30))
    assert len(rows)==90
    assert {r['expert'] for r in rows}=={'math','json_schema','csv_export'}
    assert all(verify_auto(r) for r in rows)
    assert len({(r['language'],r['verification']['op']) for r in rows if r['expert']=='math'})==9
    row=rows[0]
    changed={**row,'messages':[row['messages'][0],{'role':'assistant','content':'999999'}]}
    with pytest.raises(ValueError,match='verification_failed'): validate_record(changed,'math')
    bad={**row,'messages':[{'role':'user','content':'Calculate 999 - 998.'},row['messages'][1]]}
    assert not verify_auto(bad)


def test_legacy_rows_need_provenance_before_training():
    row={'messages':[{'role':'user','content':'hi'},{'role':'assistant','content':'hello'}]}
    assert validate_record(row,'greeting',False)
    with pytest.raises(ValueError,match='schema'): validate_record(row,'greeting')


def test_prepare_is_deduplicated_grouped_and_immutable(tmp_path):
    seed(tmp_path,100)
    source=tmp_path/'incoming'/'reasoning'/'verified-math.jsonl'
    with source.open('a',encoding='utf-8') as f:
        f.write(source.read_text(encoding='utf-8').splitlines()[0]+'\n')
    manifest=prepare(tmp_path)
    assert manifest['summary']['registered']==164
    assert manifest['summary']['accepted_rows']==300
    assert manifest['duplicates_removed']==1
    assert not list((tmp_path/'prepared'/'train'/'conversation').glob('*.jsonl'))
    for name in ('math','json_schema','csv_export'):
        groups={}
        for split,item in manifest['experts'][name]['files'].items():
            rows=[json.loads(line) for line in (tmp_path/'prepared'/item['path']).read_text(encoding='utf-8').splitlines()]
            groups[split]={row['split_group'] for row in rows}
        assert not (groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    tasks=training_plan(tmp_path)
    assert len(tasks)==164
    assert next(t for t in tasks if t['expert']=='greeting')['train_file'] is None
    with pytest.raises(ValueError,match='immutable'): prepare(tmp_path)


def test_prepared_training_rejects_source_overwrite(tmp_path):
    from Tantra.train import train_one
    with pytest.raises(ValueError,match='separate'):
        train_one(tmp_path,'math',tmp_path/'Model',tmp_path/'Model')

def test_language_and_nonredundant_task_ownership():
    row=next(generated_rows(1))
    with pytest.raises(ValueError,match='wrong_auto_task_owner'):
        validate_record({**row,'expert':'gk_math'},'gk_math')
    with pytest.raises(ValueError,match='unsupported_language'):
        validate_record({**row,'language':'not-a-language'},'math')
    language_row={**row,'expert':'lang_hi','language':'en','review':{'status':'reviewed','reviewer':'test reviewer'}}
    with pytest.raises(ValueError,match='expert_language_mismatch'):
        validate_record(language_row,'lang_hi')


def test_dedup_preserves_case_sensitive_json_values(tmp_path):
    base=next(r for r in generated_rows(1) if r['expert']=='json_schema')
    path=tmp_path/'incoming'/'structured_output'/'json_schema.jsonl'
    path.parent.mkdir(parents=True)
    rows=[]
    for value in ('alpha','ALPHA'):
        expected={'ID':value}
        payload=json.dumps(expected)
        rows.append({**base,'family_id':'case:'+value,'verification':{'kind':'json','expected':expected},
                     'messages':[{'role':'user','content':'Return JSON.\n'+payload},{'role':'assistant','content':payload}]})
    path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n',encoding='utf-8')
    manifest=prepare(tmp_path)
    assert manifest['summary']['accepted_rows']==2
    assert manifest['duplicates_removed']==0


def test_schema_matches_all_catalog_ids():
    from Tantra.train import dataset_schema
    schema=dataset_schema()
    assert set(schema['properties']['expert']['enum'])==set(catalog())
    assert 'template_id' in schema['required']


def test_undersized_pilot_is_explicit_and_cannot_validate_release(tmp_path):
    from Tantra.train import train_one
    prepared=tmp_path/'prepared'; prepared.mkdir()
    (prepared/'manifest.json').write_text(json.dumps({'experts':{'greeting':{'state':'needs_data'}}}))
    with pytest.raises(ValueError,match='not ready'):
        train_one(tmp_path,'greeting',tmp_path/'Model',tmp_path/'candidate')
    with pytest.raises(ValueError,match='cannot request release'):
        train_one(tmp_path,'greeting',tmp_path/'Model',tmp_path/'candidate',small_pilot=True,release_check=True)


def test_load_split_accepts_windows_manifest_paths(tmp_path):
    import hashlib
    from Tantra.train import load_split
    row={'schema_version':1,'expert':'greeting','language':'en','family_id':'greeting:t','template_id':'t',
         'messages':[{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello!'}],
         'source':{'id':'t','revision':'1','license':'MIT','kind':'original_ai_authored'},
         'review':{'status':'reviewed','reviewer':'tester','rationale':'fixture'}}
    path=tmp_path/'prepared'/'train'/'conversation'/'greeting.jsonl'; path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row)+'\n',encoding='utf-8')
    item={'path':'train\\conversation\\greeting.jsonl','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert load_split(tmp_path,item,'greeting')[0]['messages'][1]['content']=='Hello!'


def test_sheet_translation_family_shares_split_without_changing_provenance(tmp_path):
    from Tantra.train import family_key
    rows=[]
    for language, source, question, answer in (
        ('en', 'tantra-conversation-drafts-v1', 'Hello there', 'Hi!'),
        ('hi', 'tantra-conversation-human-v1', 'नमस्ते', 'नमस्ते!'),
        ('hinglish', 'tantra-conversation-human-v1', 'Namaste dost', 'Namaste!'),
    ):
        rows.append({'schema_version':1, 'expert':'greeting', 'language':language,
            'family_id':'greeting:draft-v1:welcome', 'template_id':'draft-v1:welcome',
            'messages':[{'role':'user','content':question},{'role':'assistant','content':answer}],
            'source':{'id':source,'revision':'1','license':'MIT','kind':'original_human_authored'},
            'review':{'status':'reviewed','reviewer':'fixture reviewer','fluent_language_review':True}})
    expected='tantra-conversation-drafts-v1:greeting:draft-v1:welcome'
    assert {family_key(row) for row in rows} == {expected}
    unrelated={**rows[0], 'source':{**rows[0]['source'], 'id':'independent-source'}}
    assert family_key(unrelated) != expected
    path=tmp_path/'incoming'/'conversation'/'greeting.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    manifest=prepare(tmp_path)
    outputs=[]
    for item in manifest['experts']['greeting']['files'].values():
        outputs.extend(json.loads(line) for line in (tmp_path/'prepared'/item['path']).read_text(encoding='utf-8').splitlines())
    assert len(outputs)==3
    assert len({row['split'] for row in outputs})==1
    assert {row['split_group'] for row in outputs}=={expected}
    assert {row['source']['id'] for row in outputs}=={row['source']['id'] for row in rows}

# ──────────────────────────────────────────────────────────────────────  test_bulk_review.py  ──────────────────────────────────────────────────────────────────────

import csv
import hashlib
import json
import pytest
from Tantra.dashboard import approve, draw_sample, run_filters
from Tantra.train import violations


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

# ──────────────────────────────────────────────────────────────────────  test_review_sheets.py  ──────────────────────────────────────────────────────────────────────

import csv
import json
import pytest
from Tantra.dashboard import convert, parse_conversation


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def make_drafts(tmp_path):
    draft = {'record_id': 'a' * 64, 'schema_version': 1, 'expert': 'identity_self', 'language': 'en',
             'family_id': 'identity_self:draft-v1:developer', 'template_id': 'draft-v1:developer',
             'messages': [{'role': 'user', 'content': 'Who made you?'}, {'role': 'assistant', 'content': 'I was developed by Atulya AI.'}],
             'source': {'id': 'x', 'revision': '1', 'license': 'MIT', 'kind': 'original_ai_authored'},
             'review': {'status': 'pending', 'reviewer': None}}
    path = tmp_path / 'drafts.jsonl'; path.write_text(json.dumps(draft) + '\n', encoding='utf-8')
    return path


EN_HEADER = ['record_id', 'expert', 'conversation', 'approve (Y/N)', 'fix or comment']
NATIVE_HEADER = ['expert', 'situation', 'english example (for meaning only)', 'language (hi / hinglish)',
                 'user message (write)', 'Tantra reply (write)', 'writer name']


def test_parse_conversation_requires_alternating_turns():
    assert parse_conversation('USER: hi\nTANTRA: Hello!')[1] == {'role': 'assistant', 'content': 'Hello!'}
    with pytest.raises(ValueError):
        parse_conversation('USER: hi\nUSER: again')
    with pytest.raises(ValueError):
        parse_conversation('TANTRA: hello')


def test_only_marked_english_rows_are_approved_with_edits(tmp_path):
    drafts = make_drafts(tmp_path); sheet = tmp_path / 'en.csv'
    write_csv(sheet, EN_HEADER, [['a' * 64, 'identity_self', 'USER: Who made you?\nTANTRA: Atulya AI made me.', 'Y', 'shorter']])
    result = convert(sheet, None, drafts, tmp_path / 'out', 'Test Reviewer', 'Checked facts')
    assert result['approved'] == 1
    row = json.loads((tmp_path / 'out/conversation/identity_self.jsonl').read_text(encoding='utf-8'))
    assert row['messages'][1]['content'] == 'Atulya AI made me.'
    assert row['review']['reviewer'] == 'Test Reviewer' and row['review']['fluent_language_review'] is False


def test_unmarked_rows_and_missing_reviewer_are_rejected(tmp_path):
    drafts = make_drafts(tmp_path); sheet = tmp_path / 'en.csv'
    write_csv(sheet, EN_HEADER, [['a' * 64, 'identity_self', 'USER: Who made you?\nTANTRA: Atulya AI.', '', '']])
    with pytest.raises(ValueError, match='No approved'):
        convert(sheet, None, drafts, tmp_path / 'out', 'R', 'why')
    with pytest.raises(ValueError, match='reviewer'):
        convert(sheet, None, drafts, tmp_path / 'out2', ' ', 'why')


def test_hindi_needs_writer_and_fluent_reviewer(tmp_path):
    drafts = make_drafts(tmp_path); sheet = tmp_path / 'hi.csv'
    write_csv(sheet, NATIVE_HEADER, [['identity_self', 'developer', '', 'hi', 'तुम्हें किसने बनाया?', 'मुझे Atulya AI ने बनाया है।', 'Asha']])
    with pytest.raises(ValueError, match='fluent'):
        convert(None, sheet, drafts, tmp_path / 'out', 'R', 'why')
    result = convert(None, sheet, drafts, tmp_path / 'out2', 'R', 'why', fluent_reviewer='Asha')
    assert result['by_language'] == {'hi': 1}
    row = json.loads((tmp_path / 'out2/conversation/identity_self.jsonl').read_text(encoding='utf-8'))
    assert row['review']['fluent_language_review'] is True
    assert row['family_id'] == 'identity_self:draft-v1:developer'

# ──────────────────────────────────────────────────────────────────────  test_instruction_candidates.py  ──────────────────────────────────────────────────────────────────────

import json
import pyarrow as pa
import pyarrow.parquet as pq
import Tantra.dashboard as module


def test_candidate_acquisition_does_not_auto_approve_or_train_on_source_test(tmp_path,monkeypatch):
    train=tmp_path/'train.parquet';test=tmp_path/'test.parquet'
    shared={'inputs':'Reserved question','targets':'Reserved answer','language_code':'eng','annotation_type':'human'}
    hi={'inputs':'हिंदी प्रश्न','targets':'हिंदी उत्तर','language_code':'hin','annotation_type':'human'}
    pq.write_table(pa.Table.from_pylist([shared,hi]),train);pq.write_table(pa.Table.from_pylist([shared]),test)
    lock=tmp_path/'lock.json'
    sources=[{'id':'fixture-train','repo':'fixture','revision':'test','license':'test-only','path':'train','kind':'candidate'},
             {'id':'fixture-test','repo':'fixture','revision':'test','license':'test-only','path':'test','kind':'reserved_source_test'}]
    lock.write_text(json.dumps({'sources':sources}))
    monkeypatch.setattr(module,'LOCK',lock)
    monkeypatch.setattr(module,'fetch',lambda source,root:train if source['path']=='train' else test)
    out=tmp_path/'out';module.run(out)
    report=json.loads((out/'index.json').read_text())
    assert report['approved_training_examples']==0
    assert report['source_test_duplicates_excluded_from_candidates']==1
    rows=[json.loads(line) for line in (out/'pending-review.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==1 and rows[0]['language']=='hi'
    assert rows[0]['review']=={'status':'pending','reviewer':None}
    assert rows[0]['expert'] is None

# ──────────────────────────────────────────────────────────────────────  test_conversation_quality.py  ──────────────────────────────────────────────────────────────────────

import json
import pytest
from Tantra.conversation_train import flags, export_seed, SEED
from Tantra.train import validate_record


def test_natural_greeting_flags_only_unsolicited_questions():
    def exchange(prompt,answer):
        return [{'role':'user','content':prompt},{'role':'assistant','content':answer}]
    assert flags(exchange('Hi','Hi!'),'greeting')==[]
    assert 'unrequested_greeting_question' in flags(exchange('Hi','Hi! How can I help?'),'greeting')
    assert not flags(exchange('Hi, can you help?','Sure. What do you need?'),'greeting')
    assert not flags(exchange('Who made you?','Atulya AI made me.'),'identity_self')


def test_conversation_seed_keeps_pending_languages_out_of_training(tmp_path):
    counts=export_seed(tmp_path/'bundle')
    assert counts=={'reviewed_english':705,'pending_multilingual':19}
    for path in (tmp_path/'bundle'/'incoming').rglob('*.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            assert row['language']=='en'
            assert validate_record(row,row['expert'])
            if row['expert']=='greeting': assert not flags(row['messages'],'greeting')
    with pytest.raises(ValueError,match='immutable'): export_seed(tmp_path/'bundle')
    records=json.loads(SEED.read_text(encoding='utf-8'))['records']
    assert {r['expert'] for r in records}>={'identity_self','identity_capabilities','identity_limitations'}
    assert any(len(r['messages'])==4 for r in records)
    assert all(r['review']['status']=='pending' for r in records if r['language']!='en')
