"""Unified safe training entry point: python -m Tantra.train [run|status|watch]."""
import argparse
import json
import math
from pathlib import Path
import time
import psutil
from Tantra.config import ROOT


def snapshot(stage):
    stage=Path(stage);base=stage/'base-training'
    def read(path):return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    status=read(stage/'status.json');manifest=read(stage/'corpus-19m'/'manifest.json')
    rows=[]
    if (base/'metrics.jsonl').exists():
        for line in (base/'metrics.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass
    alive=[]
    for lock in (stage/'worker.lock',base/'training.lock'):
        try:
            pid=int(lock.read_text());process=psutil.Process(pid)
            command=' '.join(process.cmdline())
            if 'Tantra.data.continue_pretraining' in command or 'Tantra.data.train_base_pilot' in command or 'Tantra.pretrain' in command:alive.append(pid)
        except (OSError,ValueError,psutil.Error):pass
    total=sum(v['tokens'] for k,v in manifest.get('files',{}).items() if k.endswith('-train'))
    validation=next((r['validation'] for r in reversed(rows) if 'validation' in r),{})
    checkpoint=base/'latest.pt'
    return {'stage':status.get('stage','not_started'),'worker_pids':sorted(set(alive)),
        'latest':rows[-1] if rows else {},'validation':validation,'target_tokens':total,
        'checkpoint':str(checkpoint) if checkpoint.exists() else None,
        'checkpoint_age_seconds':time.time()-checkpoint.stat().st_mtime if checkpoint.exists() else None}


def display(s,rate=None):
    row=s['latest'];tokens=row.get('target_tokens',0);target=s['target_tokens']
    print(f"State: {s['stage']} | worker: {s['worker_pids'] or 'STOPPED'}",flush=True)
    print(f"Update {row.get('update',0)}/{math.ceil(target/4096) if target else '?'} | train loss {row.get('loss','-')} | tokens {tokens:,}/{target:,}",flush=True)
    if rate and rate>0:print(f"Speed {rate:.1f} tokens/s | ETA {max(0,target-tokens)/rate/3600:.2f} hours",flush=True)
    for lang,value in s['validation'].items():print(f"Validation {lang}: {value['loss']:.4f} ({value['windows']} windows)",flush=True)
    print(f"Checkpoint: {s['checkpoint'] or 'not saved yet'} | age: {round(s['checkpoint_age_seconds']) if s['checkpoint_age_seconds'] is not None else '-'} seconds",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',nargs='?',choices=['run','status','watch','stop'],default='run')
    p.add_argument('--stage',type=Path,default=ROOT/'runs/pretraining-20m-v1');a=p.parse_args()
    s=snapshot(a.stage)
    if a.command=='stop':
        if not s['worker_pids']:p.error('No verified live worker for this stage')
        (a.stage/'base-training'/'stop.request').write_text('Stop after current optimizer update; save checkpoint.\n')
        print('Clean stop requested; wait for worker to exit before resume.');return
    if a.command=='run' and not s['worker_pids']:
        if (a.stage/'worker.lock').exists():p.error('Stale worker lock: verify its PID and archive the lock before resuming.')
        from Tantra.data.continue_pretraining import main as train
        import sys
        sys.argv=[sys.argv[0],'--output',str(a.stage),'--pilot',str(ROOT/'runs/vocabulary-64k-v1'),'--initial-model',str(ROOT/'runs/model-64k-initial-v1')]
        train();return
    if a.command=='status':display(s);return
    print('Attached to existing training; Ctrl+C closes this viewer without stopping its worker.',flush=True)
    start=time.monotonic();initial=s['latest'].get('target_tokens',0);previous=None
    try:
        while True:
            s=snapshot(a.stage);signature=(s['latest'].get('update'),bool(s['worker_pids']),s['stage'])
            if signature!=previous:
                elapsed=time.monotonic()-start;delta=s['latest'].get('target_tokens',0)-initial
                display(s,delta/elapsed if elapsed and delta>0 else None);previous=signature
            if not s['worker_pids']:break
            time.sleep(2)
    except KeyboardInterrupt:pass

if __name__=='__main__':main()
