"""Queue the isolated 64K pilot after the earlier CPU job and its quality gates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from Tantra.config import ROOT
from .public_training_job import write_status


def run(output,model_dir,previous=None):
    from .prepare_corpus import prepare
    from .train_base_pilot import run as train
    output,model_dir=Path(output),Path(model_dir)
    previous=Path(previous) if previous is not None else None
    output.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+8*3600
    while previous is not None and not (previous/'report.json').exists():
        write_status(output,'queued_after_existing_cpu_pilot',previous=str(previous))
        if (previous/'failure.json').exists():
            write_status(output,'blocked_previous_pilot_failed');return
        if time.monotonic()>deadline:
            write_status(output,'blocked_previous_pilot_timeout');return
        time.sleep(30)
    if previous is not None:
        prior=json.loads((previous/'report.json').read_text(encoding='utf-8'))
        if prior['state']!='pilot_complete' or any(prior['latest_validation'][lang]['loss']>=prior['baseline_validation'][lang]['loss'] for lang in ('hi','en')):
            write_status(output,'blocked_previous_language_quality_gate',previous_state=prior['state']);return
    try:
        corpus=output/'corpus-1m'
        write_status(output,'preparing_balanced_64k_corpus')
        if not (corpus/'manifest.json').exists(): prepare(corpus,model_dir,1_000_000)
        write_status(output,'training_64k_pilot',model_dir=str(model_dir),base_token_shares={'hi':.5,'en':.5})
        base=output/'base-pilot'
        result=train(corpus,base,model_dir,resume=(base/'latest.pt').exists(),threads=4,checkpoint_updates=10)
        write_status(output,result['state'],updates=result['updates'],trained_target_tokens=result['trained_target_tokens'],
                     validation=result['latest_validation'],deployed=False)
    except Exception as exc:
        write_status(output,'failed',error=type(exc).__name__+': '+str(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--previous',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import os
    lock=a.output/'worker.lock'
    with lock.open('x') as handle:handle.write(str(os.getpid()))
    try:run(a.output,a.model_dir,a.previous)
    finally:lock.unlink()

if __name__=='__main__':main()
