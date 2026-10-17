"""Turn filled human review sheets into approved records via the shared approve gate.

English sheet: rows marked Y are approved; an edited conversation cell is used as
written. Hindi/Hinglish sheet: rows a fluent person wrote, imported only when the
command names that fluent reviewer. Reviewer and rationale always come from the
human running the command; nothing is defaulted or self-certified.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from .conversation_sources import approve

SOURCE_EN = {'id': 'tantra-conversation-drafts-v1', 'revision': '1', 'license': 'MIT', 'kind': 'original_ai_authored'}
SOURCE_NATIVE = {'id': 'tantra-conversation-human-v1', 'revision': '1', 'license': 'MIT', 'kind': 'original_human_authored'}


def parse_conversation(text):
    """'USER: ...' / 'TANTRA: ...' lines -> alternating messages ending with the assistant."""
    messages = []
    for line in text.replace('\r\n', '\n').split('\n'):
        line = line.strip()
        if not line:
            continue
        head, sep, body = line.partition(':')
        role = {'USER': 'user', 'TANTRA': 'assistant'}.get(head.strip().upper()) if sep else None
        if role is None:
            if not messages:
                raise ValueError(f'Conversation must start with "USER:": {line!r}')
            messages[-1]['content'] += ' ' + line  # wrapped continuation line
            continue
        if not body.strip():
            raise ValueError(f'Empty {head.strip()} turn')
        messages.append({'role': role, 'content': body.strip()})
    expected = ['user', 'assistant'] * (len(messages) // 2)
    if not messages or [m['role'] for m in messages] != expected:
        raise ValueError('Turns must alternate USER/TANTRA and end with TANTRA')
    return messages


def read_sheet(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def record_id(expert, language, messages):
    return hashlib.sha256(json.dumps([expert, language, messages], ensure_ascii=False).encode()).hexdigest()


def collect(english_sheet, native_sheet, drafts, reviewer, rationale, fluent_reviewer=None):
    if not (reviewer or '').strip() or not (rationale or '').strip():
        raise ValueError('A real reviewer name and rationale are required')
    original = {}
    for line in Path(drafts).read_text(encoding='utf-8').splitlines():
        row = json.loads(line); original[row['record_id']] = row
    rows, decisions = [], []
    for sheet_row in read_sheet(english_sheet) if english_sheet else []:
        mark = (sheet_row.get('approve (Y/N)') or '').strip().upper()
        if mark not in ('Y', 'YES'):
            continue
        draft = original.get(sheet_row['record_id'])
        if draft is None:
            raise ValueError('Sheet row not in draft queue: ' + sheet_row['record_id'])
        messages = parse_conversation(sheet_row['conversation'])
        note = (sheet_row.get('fix or comment') or '').strip()
        row = {**draft, 'messages': messages, 'source': SOURCE_EN}
        row['record_id'] = record_id(row['expert'], 'en', messages)
        rows.append(row)
        decisions.append({'record_id': row['record_id'], 'status': 'approved', 'expert': row['expert'],
                          'reviewer': reviewer.strip(), 'rationale': rationale.strip() + (f' Note: {note}' if note else '')})
    for sheet_row in read_sheet(native_sheet) if native_sheet else []:
        user = (sheet_row.get('user message (write)') or '').strip()
        reply = (sheet_row.get('Tantra reply (write)') or '').strip()
        writer = (sheet_row.get('writer name') or '').strip()
        if not (user and reply):
            continue
        language = sheet_row['language (hi / hinglish)'].strip().lower()
        if language not in ('hi', 'hinglish'):
            raise ValueError('Language must be hi or hinglish: ' + language)
        if not writer:
            raise ValueError(f'Written {language} row has no writer name: {user!r}')
        if not (fluent_reviewer or '').strip():
            raise ValueError('Hindi/Hinglish rows need --fluent-reviewer (a real fluent person)')
        expert, situation = sheet_row['expert'].strip(), sheet_row['situation'].strip()
        messages = [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': reply}]
        row = {'schema_version': 1, 'expert': expert, 'language': language,
               # Same situation as the English family: translations stay in one family/split.
               'family_id': f'{expert}:draft-v1:{situation}', 'template_id': f'draft-v1:{situation}',
               'messages': messages, 'source': {**SOURCE_NATIVE, 'author': writer},
               'review': {'status': 'pending', 'reviewer': None}}
        row['record_id'] = record_id(expert, language, messages)
        rows.append(row)
        decisions.append({'record_id': row['record_id'], 'status': 'approved', 'expert': expert,
                          'reviewer': f'{writer} (writer); {fluent_reviewer.strip()} (fluent review)',
                          'rationale': rationale.strip(), 'fluent_language_review': True})
    ids = [r['record_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate conversations in the sheets')
    return rows, decisions


def convert(english_sheet, native_sheet, drafts, output, reviewer, rationale, fluent_reviewer=None):
    output = Path(output)
    if output.exists():
        raise ValueError('Output must be a new directory')
    rows, decisions = collect(english_sheet, native_sheet, drafts, reviewer, rationale, fluent_reviewer)
    if not rows:
        raise ValueError('No approved or written rows found in the sheets')
    work = output.parent / (output.name + '-queue')
    if work.exists():
        raise ValueError('Queue directory already exists: ' + str(work))
    work.mkdir(parents=True)
    queue = work / 'reviewed-sheet-rows.jsonl'
    queue.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (work / 'index.json').write_text(json.dumps({'files': {queue.name: hashlib.sha256(queue.read_bytes()).hexdigest()}}, indent=2), encoding='utf-8')
    decision_path = work / 'decisions.jsonl'
    decision_path.write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in decisions), encoding='utf-8')
    result = approve(work, decision_path, output)
    by_language = {}
    for r in rows:
        by_language[r['language']] = by_language.get(r['language'], 0) + 1
    return {**result, 'by_language': by_language, 'output': str(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--english', type=Path); p.add_argument('--native', type=Path)
    p.add_argument('--drafts', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reviewer', required=True); p.add_argument('--rationale', required=True)
    p.add_argument('--fluent-reviewer')
    a = p.parse_args()
    if not (a.english or a.native):
        p.error('Give --english and/or --native')
    print(json.dumps(convert(a.english, a.native, a.drafts, a.output, a.reviewer, a.rationale, a.fluent_reviewer), indent=2))


if __name__ == '__main__':
    main()
