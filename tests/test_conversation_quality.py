import json
import pytest
from Tantra.data.conversation_quality import flags, export_seed, SEED
from Tantra.data.expert_workbench import validate_record


def test_natural_greeting_flags_only_unsolicited_questions():
    def exchange(prompt,answer):
        return [{'role':'user','content':prompt},{'role':'assistant','content':answer}]
    assert flags(exchange('Hi','Hi!'),'greeting')==[]
    assert 'unrequested_greeting_question' in flags(exchange('Hi','Hi! How can I help?'),'greeting')
    assert not flags(exchange('Hi, can you help?','Sure. What do you need?'),'greeting')
    assert not flags(exchange('Who made you?','Atulya AI made me.'),'identity_self')


def test_conversation_seed_keeps_pending_languages_out_of_training(tmp_path):
    counts=export_seed(tmp_path/'bundle')
    assert counts=={'reviewed_english':98,'pending_multilingual':19}
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
