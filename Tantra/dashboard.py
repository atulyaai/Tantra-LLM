"""Colored animated read-only dashboard: python -m Tantra dashboard."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import psutil
from rich.console import Console,Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress_bar import ProgressBar
from rich.text import Text
from Tantra import ROOT
from Tantra.train import snapshot


def read_json(path):
    try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return {}


def metrics(stage):
    rows=[]
    try:
        for line in (Path(stage)/'base-training/metrics.jsonl').read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            # Resume may roll back uncheckpointed updates: start a new displayed segment.
            if rows and row['update']<=rows[-1]['update']:rows=[]
            rows.append(row)
    except OSError:pass
    return rows


def speed(rows):
    timed=[r for r in rows if 'timestamp' in r]
    if len(timed)<2:return None
    first,last=timed[max(0,len(timed)-21)],timed[-1]
    seconds=last['timestamp']-first['timestamp']
    return (last['target_tokens']-first['target_tokens'])/seconds if seconds>0 else None


def spark(values,width=48):
    values=values[-width:]
    if not values:return 'Waiting for training metrics'
    lo,hi=min(values),max(values)
    chars='._:-=+*#'
    return ''.join(chars[min(7,int((v-lo)/max(hi-lo,1e-9)*7))] for v in values)


def render(stage,frame=0):
    s=snapshot(stage);rows=metrics(stage);row=rows[-1] if rows else s['latest']
    report=read_json(Path(stage)/'base-training/report.json')
    config=read_json(ROOT/'Tantra/pretrain_64k.json')
    runtime=read_json(Path(stage)/'base-training/runtime.json');model=runtime.get('model',{})
    manifest=read_json(Path(stage)/'corpus-19m/manifest.json')
    alive=bool(s['worker_pids']);rate=speed(rows);tokens=row.get('target_tokens',0);total=s['target_tokens']
    activity='|/-\\'[frame%4]
    state='RUNNING' if alive else ('COMPLETE' if report.get('state')=='pilot_complete' else 'WORKER STOPPED')
    color='green' if alive else ('cyan' if state=='COMPLETE' else 'yellow')
    header=Text(f'{activity} TANTRA TRAINING  |  {state}',style='bold '+color)
    bar=ProgressBar(total=max(1,total),completed=tokens,width=None,complete_style='cyan',finished_style='green')
    progress=Text(f"Update {row.get('update',0):,}/{math.ceil(total/4096):,}   |   {tokens:,}/{total:,} tokens   |   {100*tokens/max(total,1):.2f}%",style='bold white')
    table=Table.grid(expand=True,padding=(0,2));table.add_column(style='cyan');table.add_column();table.add_column(style='cyan');table.add_column()
    eta=f'{max(0,total-tokens)/rate/3600:.2f} h' if rate and rate>0 and alive else ('paused' if not alive else 'measuring')
    table.add_row('Train loss',f"{row.get('loss',0):.4f}",'Throughput',f'{rate:.1f} tok/s' if rate and alive else ('stopped' if not alive else 'measuring'))
    table.add_row('Learning rate',f"{row.get('learning_rate',config.get('learning_rate',0)):.2e}",'ETA',eta)
    table.add_row('Gradient norm',f"{row['gradient_norm']:.3f}" if 'gradient_norm' in row else 'not recorded','Sampled peak RAM',f"{row.get('rss_bytes',0)/1024**3:.2f} GiB")
    vm=psutil.virtual_memory();table.add_row('CPU load',f'{psutil.cpu_percent():.0f}% system','System RAM free',f'{vm.available/1024**3:.2f} GiB')
    model_text=f"{runtime['base_parameters']:,} params | width {model.get('d_model')} | {model.get('n_layers')} layers" if runtime else 'runtime metadata not recorded'
    table.add_row('Model',model_text,'Vocabulary / context',f"{model.get('vocab_size','unknown')} / {model.get('max_seq_len','unknown')} tokens")
    table.add_row('Optimizer',f"{runtime.get('optimizer','unknown')} | {runtime.get('precision','unknown')} | {runtime.get('threads','?')} threads",'Microbatch / targets',f"{runtime.get('microbatch','unknown')} / {runtime.get('tokens_per_update','unknown')}")
    shares=manifest.get('language_token_shares',{})
    table.add_row('Language mix',' / '.join(f'{lang}: {value:.0%}' for lang,value in shares.items()) or 'not recorded','Hinglish','No chat-training metric available')
    val=next((r['validation'] for r in reversed(rows) if 'validation' in r),{})
    origin='Latest logged evaluation'
    if not val:
        val=report.get('latest_validation',{});origin='Last completed session evaluation (not current update)'
    vt=Table(expand=True);vt.add_column('Language',style='cyan');vt.add_column('Val loss');vt.add_column('Perplexity');vt.add_column('Top-1');vt.add_column('Top-5');vt.add_column('Windows')
    for lang in ('hi','en'):
        v=val.get(lang,{})
        vt.add_row('Hindi' if lang=='hi' else 'English',f"{v['loss']:.4f}" if 'loss' in v else 'pending',
            f"{v['perplexity']:.1f}" if 'perplexity' in v else '-',
            f"{v['top1_percent']:.1f}%" if 'top1_percent' in v else '-',f"{v['top5_percent']:.1f}%" if 'top5_percent' in v else '-',str(v.get('windows','-')))
    age=s['checkpoint_age_seconds'];checkpoint=Text(f"Checkpoint: {s['checkpoint'] or 'not saved'}\nAge: {int(age) if age is not None else '-'} seconds | PID: {s['worker_pids'] or 'none'}\nEvaluation interval: {runtime.get('evaluation_every','unknown')} updates | checkpoint interval: {runtime.get('checkpoint_updates','unknown')} updates",style='dim')
    last_time=row.get('timestamp');idle=time.time()-last_time if last_time else None
    note='LIVE: system CPU/free RAM and worker liveness. SAVED: loss/tokens/validation from training logs. Animation is viewer refresh.'
    if not alive:note+=' TRAINING IS STOPPED; saved measurements are frozen.'
    if alive and idle and idle>180:note+=' No new update for >3 minutes: evaluation/checkpoint or stalled work; inspect console.'
    probe_rows=[];probe_update=None
    histories=[p for p in (Path(stage)/'base-training/probe_history.jsonl',Path(stage)/'probe-check-now/probe_history.jsonl') if p.exists()]
    for history in histories:
        try:
            with history.open(encoding='utf-8') as reader:
                for line in reader:
                    try:entry=json.loads(line)
                    except ValueError:continue
                    responses=entry.get('responses',[])
                    if any('answer' in row for row in responses) and (probe_update is None or entry.get('update',0)>=probe_update):
                        probe_rows=responses;probe_update=entry.get('update',0)
        except OSError:pass
    samples=[r for r in probe_rows if 'answer' in r][:3]
    probe_text='\n'.join(f"[{r['language']}] Q: {r['question']}\nA: {r['answer']}" for r in samples) or 'No generated question checks recorded yet. Probe metrics are development diagnostics, not certified correctness.'
    return Group(Panel(Group(header,progress,bar),border_style=color),
                 Panel(table,title='MODEL + TRAINING',border_style='blue'),
                 Panel(Group(Text(origin,style='dim'),vt),title='HELD-OUT VALIDATION',border_style='magenta'),
                 Panel(Text(spark([r['loss'] for r in rows])+'  latest train-loss trend',style='yellow'),title='LOSS HISTORY'),
                 Panel(checkpoint,border_style='blue'),Panel(Text(probe_text),title=f'RECORDED QUESTION CHECKS | UPDATE {probe_update if probe_update is not None else "pending"}',border_style='green'),Text(note,style='dim'),Text('Ctrl+C closes this dashboard only. '+('Training remains in its worker.' if alive else 'No live training worker. Resume with python -m Tantra.pretrain --resume --microbatch 16.'),style='bold yellow'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,default=ROOT/'runs/pretraining-20m-v1')
    p.add_argument('--once',action='store_true');p.add_argument('--export-html',type=Path);a=p.parse_args()
    console=Console(record=bool(a.export_html))
    if a.once or a.export_html:
        console.print(render(a.stage))
        if a.export_html:console.save_html(str(a.export_html))
        return
    try:
        with Live(render(a.stage),console=console,refresh_per_second=4,screen=True) as live:
            frame=0
            while True:
                time.sleep(.5);frame+=1;live.update(render(a.stage,frame))
    except KeyboardInterrupt:pass


# ──  data/bulk_review.py  ──
"""Bulk review for large own-authored conversation batches (owner policy, 2026-10-19).

A batch of generated drafts counts as reviewed only when:
1. every row passes the automatic filters (schema, persona rules, language, length, duplicates), and
2. the owner marks a seeded random sample and approves at least --min-rate of it.
Rows the owner rejects in the sample are dropped. Every approved row records exactly how it
was reviewed (review_method, sample size/rate, filters), so it is never mistaken for row-by-row review.

  python -m Tantra review_cli filter  --queue Q
  python -m Tantra review_cli sample  --queue Q --n 50
  python -m Tantra review_cli approve --queue Q --sample Q/spot-check.csv --reviewer NAME --rationale TEXT --output DIR
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path
from Tantra.train import catalog, validate_record
from Tantra.train import violations, SAYS_AI, SAYS_CREATOR
from Tantra import sha256

FILTERS = ['schema', 'bulk_review_scope', 'persona', 'english_script', 'length', 'exact_duplicate']
DRAW_VERSION = 2
EXPERT_DOMAINS = {name: spec['domain'] for name, spec in catalog().items()}
MAX_CHARS = 400
MAX_TURNS = 8
MIN_SAMPLE = 50
NON_LATIN = re.compile(r'[\u0900-\u097f\u0600-\u06ff\u0400-\u04ff\u4e00-\u9fff]')


def message_hash(row):
    return hashlib.sha256(json.dumps([row['expert'], row['messages']], ensure_ascii=False).encode()).hexdigest()


def check(row, seen):
    """Reasons a row fails the automatic filters (empty list = pass)."""
    reasons = []
    if not isinstance(row, dict):
        return ['schema: row must be an object']
    if row.get('expert') not in EXPERT_DOMAINS:
        return ['schema: unknown expert']
    try:
        validate_record({**row, 'review': {'status': 'reviewed', 'reviewer': 'filter-check', 'rationale': 'filter-check'}}, row['expert'])
    except (ValueError, KeyError, TypeError) as exc:
        return ['schema: ' + str(exc)]
    msgs = row.get('messages', [])
    # The owner policy permits sampled review for ordinary own-authored English
    # conversation only. Identity/persona and safety still require individual review.
    expert = row.get('expert', '')
    if (expert.startswith('identity_') or EXPERT_DOMAINS.get(expert) == 'safety'
            or expert == 'safety_refusal'
            or any(SAYS_AI.search(m.get('content', '')) or SAYS_CREATOR.search(m.get('content', ''))
                   for m in msgs if m.get('role') == 'assistant')):
        reasons.append('bulk_review_scope: identity/persona/safety needs row-by-row review')
    if row.get('source', {}).get('kind') not in ('original_ai_authored', 'original_ai_authored_templated', 'original_human_authored'):
        reasons.append('bulk_review_scope: sampled review is only for own-authored conversation')
    reasons += ['persona: ' + v for v in violations(msgs)]
    if row.get('language') == 'en' and any(NON_LATIN.search(m['content']) for m in msgs):
        reasons.append('english_script: non-Latin text in an English row')
    if len(msgs) > MAX_TURNS or any(len(m['content']) > MAX_CHARS for m in msgs):
        reasons.append('length')
    key = message_hash(row)
    if key in seen:
        reasons.append('exact_duplicate')
    seen.add(key)
    return reasons


def load_queue(queue):
    queue = Path(queue)
    index = json.loads((queue / 'index.json').read_text(encoding='utf-8'))
    rows = []
    for name, expected in sorted(index['files'].items()):
        if sha256(queue / name) != expected:
            raise ValueError('Queue checksum mismatch: ' + name)
        rows += [json.loads(l) for l in (queue / name).read_text(encoding='utf-8').splitlines() if l.strip()]
    ids = [row.get('record_id') for row in rows]
    if any(not isinstance(identity, str) or not identity for identity in ids) or len(ids) != len(set(ids)):
        raise ValueError('Queue record IDs must be non-empty and unique')
    return rows


def run_filters(queue):
    seen = set(); passed = []; rejected = []
    for row in load_queue(queue):
        reasons = check(row, seen)
        (rejected.append({'record_id': row.get('record_id'), 'reasons': reasons}) if reasons else passed.append(row))
    report = {'rows': len(passed) + len(rejected), 'passed': len(passed), 'rejected': len(rejected),
              'filters': FILTERS, 'rejections': rejected}
    (Path(queue) / 'filter-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return passed, report


def draw_sample(queue, n=None, seed=0):
    """Stratified draw: at least one row from every family, then random rows up to n.
    The draw (seed, ids, batch hash) is recorded so approve() only accepts this exact sample."""
    passed, report = run_filters(queue)
    if not passed:
        raise ValueError('No rows passed the filters')
    rng = random.Random(seed); families = {}
    for row in passed:
        families.setdefault(row['family_id'], []).append(row)
    n = max(n or 0, MIN_SAMPLE, len(families))
    sample = [rng.choice(rows) for _, rows in sorted(families.items())]
    chosen = {r['record_id'] for r in sample}
    rest = [r for r in passed if r['record_id'] not in chosen]
    sample += rng.sample(rest, min(max(0, n - len(sample)), len(rest)))
    rng.shuffle(sample)
    path = Path(queue) / 'spot-check.csv'
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['record_id', 'expert', 'conversation', 'ok (Y/N)', 'comment'])
        for row in sample:
            convo = sample_conversation(row)
            w.writerow([row['record_id'], row['expert'], convo, '', ''])
    draw = {'version': DRAW_VERSION, 'seed': seed, 'n': len(sample), 'families': len(families), 'record_ids': sorted(r['record_id'] for r in sample),
            'batch_sha256': batch_hash(passed)}
    (Path(queue) / 'spot-check.draw.json').write_text(json.dumps(draw, indent=2), encoding='utf-8')
    return {'sample': len(sample), 'families_covered': len(families), 'passed': len(passed),
            'rejected_by_filters': report['rejected'], 'path': str(path)}


def batch_hash(rows):
    # Bind all text and provenance, not merely caller-controlled record IDs.
    payloads = sorted(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) for row in rows)
    return hashlib.sha256('\n'.join(payloads).encode('utf-8')).hexdigest()


def sample_conversation(row):
    return '\n'.join(('USER: ' if m['role'] == 'user' else 'TANTRA: ') + m['content'] for m in row['messages'])



def approve(queue, sample_csv, output, reviewer, rationale, min_rate=0.95):
    output = Path(output)
    if not isinstance(min_rate, (int, float)) or not math.isfinite(min_rate) or not 0.95 <= min_rate <= 1.0:
        raise ValueError('Approval threshold must be finite and between 0.95 and 1.0 (owner policy)')
    if output.exists():
        raise ValueError('Reviewed export must be a new directory')
    if not (reviewer or '').strip() or not (rationale or '').strip():
        raise ValueError('A real reviewer name and rationale are required')
    passed, report = run_filters(queue)
    draw_path = Path(queue) / 'spot-check.draw.json'
    if not draw_path.exists():
        raise ValueError('No recorded draw; run the sample command first')
    draw = json.loads(draw_path.read_text(encoding='utf-8'))
    if draw.get('version') != DRAW_VERSION:
        raise ValueError('Legacy sample draw does not bind content; run sample again before review')
    if draw['batch_sha256'] != batch_hash(passed):
        raise ValueError('The batch changed since the sample was drawn; draw a new sample')
    with Path(sample_csv).open(newline='', encoding='utf-8-sig') as f:
        sample_rows = list(csv.DictReader(f))
    sample_ids = [row.get('record_id') for row in sample_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError('Duplicate record IDs in the marked sample')
    marks = {r['record_id']: (r.get('ok (Y/N)') or '').strip().upper() for r in sample_rows}

    if sorted(marks) != draw['record_ids']:
        raise ValueError('The marked sample is not the recorded draw')
    by_id = {row['record_id']: row for row in passed}
    for shown in sample_rows:
        source = by_id[shown['record_id']]
        if shown.get('expert') != source['expert'] or shown.get('conversation') != sample_conversation(source):
            raise ValueError('Displayed sample conversation/expert changed; fixes require a new queue and draw')
    if len(marks) < MIN_SAMPLE:

        raise ValueError(f'Sample has {len(marks)} rows; at least {MIN_SAMPLE} are required')
    if any(m not in ('Y', 'N') for m in marks.values()):
        raise ValueError('Every sampled row must be marked Y or N')
    rate = sum(m == 'Y' for m in marks.values()) / len(marks)
    if rate < min_rate:
        raise ValueError(f'Spot-check approval {rate:.0%} is below the required {min_rate:.0%}; fix the generator and redraw')
    if any(row['language'] != 'en' for row in passed):
        raise ValueError('Bulk spot-check approval is English-only; Hindi/Hinglish need fluent review')
    # A rejected sample row condemns its whole family: siblings share the same template.
    by_id = {row['record_id']: row for row in passed}
    bad_families = {by_id[i]['family_id'] for i, m in marks.items() if m == 'N'}
    decision = sha256(Path(sample_csv))
    review = {'status': 'reviewed', 'reviewer': f'{reviewer.strip()} (spot-checked {len(marks)} of {len(passed)}, {rate:.0%} approved)',
              'rationale': rationale.strip(), 'review_method': 'owner_sample_spot_check', 'sample_size': len(marks),
              'sample_approval_rate': rate, 'batch_size': len(passed), 'families_sampled': draw['families'],
              'filters': FILTERS, 'fluent_language_review': False, 'domain_review': False, 'decision_sha256': decision}
    records = [(catalog()[row['expert']]['domain'], row['expert'], validate_record({**row, 'review': review}, row['expert']))
               for row in passed if row['family_id'] not in bad_families]
    output.mkdir(parents=True)
    counts = {}
    for domain, expert, record in records:
        (output / domain).mkdir(exist_ok=True)
        with (output / domain / f'{expert}.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        counts[expert] = counts.get(expert, 0) + 1
    return {'approved': sum(counts.values()), 'by_expert': counts, 'sample_approval_rate': rate,
            'rejected_in_sample': sum(m == 'N' for m in marks.values()), 'families_dropped': sorted(bad_families), 'rejected_by_filters': report['rejected']}


def review_cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('filter'); f.add_argument('--queue', type=Path, required=True)
    s = sub.add_parser('sample'); s.add_argument('--queue', type=Path, required=True)
    s.add_argument('--n', type=int, default=None, help='rows to draw; at least one per family and at least MIN_SAMPLE'); s.add_argument('--seed', type=int, default=0)
    a = sub.add_parser('approve'); a.add_argument('--queue', type=Path, required=True); a.add_argument('--sample', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True); a.add_argument('--reviewer', required=True)
    a.add_argument('--rationale', required=True); a.add_argument('--min-rate', type=float, default=0.95)
    x = p.parse_args()
    if x.command == 'filter':
        result = {k: v for k, v in run_filters(x.queue)[1].items() if k != 'rejections'}
    elif x.command == 'sample':
        result = draw_sample(x.queue, x.n, x.seed)
    else:
        result = approve(x.queue, x.sample, x.output, x.reviewer, x.rationale, x.min_rate)
    print(json.dumps(result, indent=2))



# ──  data/review_sheets.py  ──
"""Turn filled human review sheets into approved records via the shared approve gate.

English sheet: rows marked Y are approved; an edited conversation cell is used as
written. Hindi/Hinglish sheet: rows a fluent person wrote, imported only when the
command names that fluent reviewer. Reviewer and rationale always come from the
human running the command; nothing is defaulted or self-certified.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from Tantra.conversation_train import cs_approve

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
    result = cs_approve(work, decision_path, output)
    by_language = {}
    for r in rows:
        by_language[r['language']] = by_language.get(r['language'], 0) + 1
    return {**result, 'by_language': by_language, 'output': str(output)}


def sheets_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--english', type=Path); p.add_argument('--native', type=Path)
    p.add_argument('--drafts', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reviewer', required=True); p.add_argument('--rationale', required=True)
    p.add_argument('--fluent-reviewer')
    a = p.parse_args()
    if not (a.english or a.native):
        p.error('Give --english and/or --native')
    print(json.dumps(convert(a.english, a.native, a.drafts, a.output, a.reviewer, a.rationale, a.fluent_reviewer), indent=2))



# ──  data/instruction_candidates.py  ──
"""Acquire licensed human-authored instruction candidates; never auto-approve."""
import argparse
from collections import Counter
import hashlib, json
from pathlib import Path
from Tantra import ROOT
from Tantra.pretrain import fetch, location
from Tantra import sha256

RAW=ROOT/'Datasets/raw/conversation-sources-v1'
LOCK=ROOT/'docs/data/conversation_sources.lock.json'


def run(output):
    import pyarrow.parquet as pq
    output=Path(output)
    if output.exists():raise ValueError('Use new immutable review output')
    sources=json.loads(LOCK.read_text(encoding='utf-8'))['sources']
    output.mkdir(parents=True);counts=Counter();seen={};overlap=0
    # Process reserved test FIRST so any matching training content remains excluded.
    for source in reversed(sources):
        path=fetch(source,RAW);reserved=source['kind']=='reserved_source_test'
        filename='reserved-source-test.jsonl' if reserved else 'pending-review.jsonl'
        with (output/filename).open('w',encoding='utf-8') as writer:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for raw in batch.to_pylist():
                    code=raw.get('language_code');language={'eng':'en','hin':'hi'}.get(code)
                    if language is None:continue
                    question,answer=raw.get('inputs'),raw.get('targets')
                    if not isinstance(question,str) or not isinstance(answer,str) or not question.strip() or not answer.strip():
                        counts['malformed']+=1;continue
                    key=hashlib.sha256(json.dumps([question,answer],ensure_ascii=False).encode()).hexdigest()
                    if key in seen:
                        counts['duplicate']+=1
                        if not reserved and seen[key]=='test':overlap+=1
                        continue
                    seen[key]='test' if reserved else 'candidate'
                    row={'candidate_id':key,'language':language,'messages':[{'role':'user','content':question.strip()},{'role':'assistant','content':answer.strip()}],
                         'source':{'id':source['repo'],'revision':source['revision'],'license':source['license'],'file':source['path'],
                                   'annotation_type':raw.get('annotation_type')},
                         'expert':None,'review':{'status':'pending','reviewer':None},
                         'source_split':'reserved_test' if reserved else 'candidate',
                         'warning':'Human source annotation does not certify Tantra identity, expert ownership or fluent-review approval.'}
                    writer.write(json.dumps(row,ensure_ascii=False)+'\n');counts[('reserved_' if reserved else 'candidate_')+language]+=1
    report={'counts':dict(counts),'source_test_duplicates_excluded_from_candidates':overlap,'source_lock_sha256':sha256(LOCK),
            'review_status':'pending','approved_training_examples':0,'hinglish':'Not inferred from Hindi or fabricated by translation',
            'limitations':['Source may contain mislabeled language or unsafe/inaccurate content','Source IDs do not establish semantic family independence; review grouping before splitting','No expert is marked data_ready by acquisition']}
    (output/'index.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


def instruct_cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)



# ──  utils/health_monitor.py  ──
"""
Tantra Expert Health Monitor
==============================
Monitor per-expert training health:
- Loss spikes (wrong data detection)
- Gradient explosion
- Activation distribution shifts
- Usage patterns
"""


import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

log = logging.getLogger("Tantra.health")


class HealthStatus(Enum):
    OK = "OK"
    WARMING_UP = "WARMING_UP"
    LOSS_SPIKE = "LOSS_SPIKE"
    GRADIENT_EXPLOSION = "GRADIENT_EXPLOSION"
    DISTRIBUTION_SHIFT = "DISTRIBUTION_SHIFT"
    LOW_USAGE = "LOW_USAGE"
    DEAD_EXPERT = "DEAD_EXPERT"


@dataclass
class ExpertHealth:
    """Health metrics for a single expert."""
    name: str
    window_size: int = 500
    
    losses: deque = field(default_factory=lambda: deque(maxlen=500))
    grad_norms: deque = field(default_factory=lambda: deque(maxlen=500))
    activation_means: deque = field(default_factory=lambda: deque(maxlen=500))
    activation_stds: deque = field(default_factory=lambda: deque(maxlen=500))
    router_weights: deque = field(default_factory=lambda: deque(maxlen=500))
    
    # Thresholds
    loss_spike_threshold: float = 3.0
    grad_explosion_threshold: float = 5.0
    distribution_shift_sigma: float = 2.0
    min_activation_std: float = 1e-4
    
    def update(
        self,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        activation_mean: Optional[float] = None,
        activation_std: Optional[float] = None,
        router_weight: Optional[float] = None,
    ) -> None:
        """Update metrics with new values."""
        if loss is not None:
            self.losses.append(loss)
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
        if activation_mean is not None:
            self.activation_means.append(activation_mean)
        if activation_std is not None:
            self.activation_stds.append(activation_std)
        if router_weight is not None:
            self.router_weights.append(router_weight)
    
    def check(self) -> HealthStatus:
        """Check health status."""
        if len(self.losses) < 50:
            return HealthStatus.WARMING_UP
        
        # Check loss spike
        if len(self.losses) > 100:
            recent = np.mean(list(self.losses)[-20:])
            historical = np.mean(list(self.losses)[:-20])
            if recent > historical * self.loss_spike_threshold:
                return HealthStatus.LOSS_SPIKE
        
        # Check gradient explosion
        if len(self.grad_norms) > 50:
            recent_grad = np.mean(list(self.grad_norms)[-10:])
            historical_grad = np.percentile(list(self.grad_norms)[:-10], 99)
            if recent_grad > historical_grad * self.grad_explosion_threshold:
                return HealthStatus.GRADIENT_EXPLOSION
        
        # Check activation distribution shift
        if len(self.activation_means) > 50:
            recent_mean = np.mean(list(self.activation_means)[-20:])
            hist_means = list(self.activation_means)[:-20]
            hist_mean = np.mean(hist_means)
            hist_std = np.std(hist_means)
            if hist_std > 0 and abs(recent_mean - hist_mean) > self.distribution_shift_sigma * hist_std:
                return HealthStatus.DISTRIBUTION_SHIFT
        
        # Check dead expert (near-zero activation std)
        if len(self.activation_stds) > 20:
            recent_std = np.mean(list(self.activation_stds)[-10:])
            if recent_std < self.min_activation_std:
                return HealthStatus.DEAD_EXPERT
        
        # Check low usage
        if len(self.router_weights) > 100:
            avg_usage = np.mean(self.router_weights)
            if avg_usage < 0.001:
                return HealthStatus.LOW_USAGE
        
        return HealthStatus.OK
    
    def get_stats(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "avg_loss": float(np.mean(self.losses)) if self.losses else 0.0,
            "avg_grad_norm": float(np.mean(self.grad_norms)) if self.grad_norms else 0.0,
            "avg_activation_mean": float(np.mean(self.activation_means)) if self.activation_means else 0.0,
            "avg_activation_std": float(np.mean(self.activation_stds)) if self.activation_stds else 0.0,
            "avg_router_weight": float(np.mean(self.router_weights)) if self.router_weights else 0.0,
            "num_updates": len(self.losses),
        }


class ExpertHealthMonitor:
    """
    Monitor health of all experts during training.
    Detects wrong data, gradient issues, distribution shifts.
    """
    
    def __init__(
        self,
        expert_names: List[str],
        window_size: int = 500,
        alert_callback: Optional[callable] = None,
    ):
        self.experts = {
            name: ExpertHealth(name, window_size)
            for name in expert_names
        }
        self.alert_callback = alert_callback
        self.alert_history: List[Dict] = []
    
    def update(
        self,
        expert_name: str,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        activation: Optional[Any] = None,
        router_weight: Optional[float] = None,
    ) -> None:
        """Update metrics for an expert."""
        if expert_name not in self.experts:
            return
        
        health = self.experts[expert_name]
        
        # Extract activation stats if tensor provided
        act_mean = act_std = None
        if activation is not None and hasattr(activation, 'mean'):
            act_mean = float(activation.mean().item())
            act_std = float(activation.std().item())
        
        health.update(
            loss=loss,
            grad_norm=grad_norm,
            activation_mean=act_mean,
            activation_std=act_std,
            router_weight=router_weight,
        )
        
        # Check health
        status = health.check()
        if status != HealthStatus.OK and status != HealthStatus.WARMING_UP:
            self._alert(expert_name, status, health.get_stats())
    
    def _alert(self, expert_name: str, status: HealthStatus, stats: Dict) -> None:
        """Handle health alert."""
        alert = {
            "expert": expert_name,
            "status": status.value,
            "stats": stats,
        }
        self.alert_history.append(alert)
        
        log.warning(f"⚠️ EXPERT HEALTH ALERT: {expert_name} - {status.value}")
        log.warning(f"   Stats: {stats}")
        
        if self.alert_callback:
            self.alert_callback(expert_name, status, stats)
    
    def check_all(self) -> Dict[str, HealthStatus]:
        """Check health of all experts."""
        return {name: health.check() for name, health in self.experts.items()}
    
    def get_report(self) -> Dict[str, Any]:
        """Get full health report."""
        return {
            "experts": {
                name: {
                    "status": health.check().value,
                    "stats": health.get_stats(),
                }
                for name, health in self.experts.items()
            },
            "alerts": self.alert_history[-50:],  # Last 50 alerts
        }
    
    def get_anomalies(self) -> Dict[str, HealthStatus]:
        """Get only experts with anomalies."""
        return {
            name: status
            for name, status in self.check_all().items()
            if status not in (HealthStatus.OK, HealthStatus.WARMING_UP)
        }
    
    def should_quarantine_batch(self, expert_name: str) -> bool:
        """Check if current batch should be quarantined for this expert."""
        status = self.experts[expert_name].check()
        return status in (HealthStatus.LOSS_SPIKE, HealthStatus.GRADIENT_EXPLOSION)
    
    def reset_expert(self, expert_name: str) -> None:
        """Reset health tracking for an expert (e.g., after data fix)."""
        if expert_name in self.experts:
            self.experts[expert_name] = ExpertHealth(expert_name, self.experts[expert_name].window_size)
            log.info(f"Health monitor reset for {expert_name}")


# ── Automated Data Quality Check ─────────────────────────────────────────────

def check_batch_quality(
    batch: Dict,
    tokenizer,
    expert_name: str,
) -> Dict[str, Any]:
    """
    Quick quality checks on a batch before training.
    Returns dict with quality flags.
    """
    texts = batch.get("text", [])
    input_ids = batch.get("input_ids")
    
    issues = []
    
    # Check for empty/short texts
    short_count = sum(1 for t in texts if len(t) < 20)
    if short_count > len(texts) * 0.1:
        issues.append(f"SHORT_TEXTS: {short_count}/{len(texts)}")
    
    # Check for repeated tokens (degenerate)
    if input_ids is not None:
        for i, ids in enumerate(input_ids):
            unique_ratio = len(set(ids.tolist())) / len(ids)
            if unique_ratio < 0.1:
                issues.append(f"REPEATED_TOKENS sample {i}: unique_ratio={unique_ratio:.2f}")
    
    # Check for wrong language
    # (would need language detection)
    
    return {
        "expert": expert_name,
        "batch_size": len(texts),
        "issues": issues,
        "pass": len(issues) == 0,
    }


if __name__ == '__main__':
    main()
