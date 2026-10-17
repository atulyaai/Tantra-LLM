"""Prepare new non-overlapping data, then continue pretraining after pilot gates."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time
import psutil
from .public_training_job import write_status
from .public_sources import sha256


def pilot_passed(report):
    if report.get('state')!='pilot_complete': return False
    try:
        return all(math.isfinite(report['baseline_validation'][lang]['loss']) and math.isfinite(report['latest_validation'][lang]['loss']) and
                   report['latest_validation'][lang]['loss']<report['baseline_validation'][lang]['loss']
                   for lang in ('hi','en'))
    except (KeyError,TypeError): return False


def run(output,pilot,initial_model,new_tokens=19_000_000):
    from .prepare_corpus import prepare
    from .train_base_pilot import run as train
    output,pilot,initial_model=Path(output),Path(pilot),Path(initial_model)
    corpus=output/'corpus-19m'
    write_status(output,'preparing_new_balanced_data',new_training_tokens=new_tokens,exclude=str(pilot/'corpus-1m'))
    if not (corpus/'manifest.json').exists():
        prepare(corpus,initial_model,new_tokens,exclude_corpora=[pilot/'corpus-1m'])
    meta=json.loads((corpus/'manifest.json').read_text(encoding='utf-8'))
    if not meta['ready']:
        write_status(output,'blocked_data_shortage',files=meta['files']);return
    deadline=time.monotonic()+8*3600
    while True:
        path=pilot/'base-pilot'/'report.json'
        report=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if report.get('state')=='pilot_complete':break
        if (pilot/'base-pilot'/'failure.json').exists() or (pilot/'status.json').exists() and json.loads((pilot/'status.json').read_text(encoding='utf-8')).get('stage')=='failed':
            write_status(output,'blocked_pilot_failed');return
        lock=pilot/'worker.lock'
        if not lock.exists() or not psutil.pid_exists(int(lock.read_text())):
            write_status(output,'blocked_pilot_not_running',previous_state=report.get('state'));return
        if time.monotonic()>deadline:
            write_status(output,'blocked_wait_timeout');return
        write_status(output,'waiting_for_pilot_quality_gate',new_data_ready=True,new_training_tokens=new_tokens)
        time.sleep(30)
    if not pilot_passed(report):
        write_status(output,'blocked_pilot_language_quality_gate',validation=report.get('latest_validation'));return
    source=output/'source-model'
    if not source.exists():
        source.mkdir()
        shutil.copy2(pilot/'base-pilot'/'base.pt',source/'base.pt')
        shutil.copy2(pilot/'base-pilot'/'tokenizer.json',source/'tokenizer.json')
    if sha256(source/'tokenizer.json')!=meta['tokenizer_sha256']:raise ValueError('Stage tokenizer mismatch')
    write_status(output,'continued_pretraining',new_training_tokens=new_tokens,
                 note='Pilot weights; fresh optimizer for this new data stage; no deployment')
    base=output/'base-training'
    result=train(corpus,base,source,resume=(base/'latest.pt').exists(),threads=4,checkpoint_updates=10)
    write_status(output,'stage_finished' if result['state']=='pilot_complete' else result['state'],
                 trained_target_tokens=result['trained_target_tokens'],validation=result['latest_validation'],deployed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--initial-model',type=Path,required=True)
    p.add_argument('--new-tokens',type=int,default=19_000_000);a=p.parse_args()
    if not 1000<=a.new_tokens<=19_000_000:p.error('New-stage token budget must be 1000..19000000')
    a.output.mkdir(parents=True,exist_ok=True);lock=a.output/'worker.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:run(a.output,a.pilot,a.initial_model,a.new_tokens)
    except Exception as exc:
        write_status(a.output,'failed',error=type(exc).__name__+': '+str(exc));raise
    finally:lock.unlink()

if __name__=='__main__':main()
