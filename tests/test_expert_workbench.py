import json
from pathlib import Path

import pytest

from Tantra.data.expert_workbench import (catalog, generated_rows, initialize, messages_of,
    prepare, seed, training_plan, validate_record, verify_auto)


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
    from Tantra.data.train_prepared import train_one
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
    from Tantra.data.expert_workbench import dataset_schema
    schema=dataset_schema()
    assert set(schema['properties']['expert']['enum'])==set(catalog())
    assert 'template_id' in schema['required']


def test_undersized_pilot_is_explicit_and_cannot_validate_release(tmp_path):
    from Tantra.data.train_prepared import train_one
    prepared=tmp_path/'prepared'; prepared.mkdir()
    (prepared/'manifest.json').write_text(json.dumps({'experts':{'greeting':{'state':'needs_data'}}}))
    with pytest.raises(ValueError,match='not ready'):
        train_one(tmp_path,'greeting',tmp_path/'Model',tmp_path/'candidate')
    with pytest.raises(ValueError,match='cannot request release'):
        train_one(tmp_path,'greeting',tmp_path/'Model',tmp_path/'candidate',small_pilot=True,release_check=True)


def test_load_split_accepts_windows_manifest_paths(tmp_path):
    import hashlib
    from Tantra.data.train_prepared import load_split
    row={'schema_version':1,'expert':'greeting','language':'en','family_id':'greeting:t','template_id':'t',
         'messages':[{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello!'}],
         'source':{'id':'t','revision':'1','license':'MIT','kind':'original_ai_authored'},
         'review':{'status':'reviewed','reviewer':'tester','rationale':'fixture'}}
    path=tmp_path/'prepared'/'train'/'conversation'/'greeting.jsonl'; path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row)+'\n',encoding='utf-8')
    item={'path':'train\\conversation\\greeting.jsonl','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert load_split(tmp_path,item,'greeting')[0]['messages'][1]['content']=='Hello!'
