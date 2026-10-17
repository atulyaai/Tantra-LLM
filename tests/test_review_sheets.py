import csv
import json
import pytest
from Tantra.data.review_sheets import convert, parse_conversation


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
