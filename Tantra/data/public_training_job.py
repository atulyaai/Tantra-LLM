"""Run the public-data pilot sequentially, stopping at missing data/quality gates."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
from Tantra.config import ROOT
from .public_sources import sources, location


def write_status(output, stage, **details):
    temp=output/'status.tmp'
    temp.write_text(json.dumps({'stage':stage,**details},indent=2)+'\n',encoding='utf-8')
    os.replace(temp,output/'status.json')
    print(json.dumps({'stage':stage,**details}),flush=True)


def run(output, wait_minutes=90, fetch_sources=False):
    from .prepare_corpus import prepare as prepare_corpus
    from .conversation_sources import prepare as prepare_conversations
    from .train_base_pilot import run as train_base
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    lock=output/'job.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        if fetch_sources:
            from .public_sources import fetch
            write_status(output,'acquiring_pinned_sources',note='Resume complete byte ranges; verify full hashes before preparation')
            for source in sources(): fetch(source)
        deadline=time.monotonic()+wait_minutes*60
        while True:
            missing=[s['id'] for s in sources() if not location(s).exists()]
            if not missing: break
            write_status(output,'waiting_verified_downloads',missing=missing)
            if time.monotonic()>=deadline: return
            time.sleep(min(30,max(0,deadline-time.monotonic())))
        review=output/'review-queue'
        if not (review/'index.json').exists():
            write_status(output,'preparing_conversation_review_queue')
            prepare_conversations(review)
        corpus=output/'corpus-1m'
        if not (corpus/'manifest.json').exists():
            write_status(output,'preparing_bilingual_corpus',target_tokens=1_000_000)
            prepare_corpus(corpus)
        write_status(output,'base_pilot_training',note='One pass over the 1M-token pilot; original model preserved')
        base=output/'base-pilot'
        result=train_base(corpus,base,resume=(base/'latest.pt').exists(),threads=4)
        write_status(output,'base_pilot_finished_expert_review_required',base_state=result['state'],
            trained_target_tokens=result['trained_target_tokens'],validation=result['latest_validation'],
            review_index=str(review/'index.json'),
            next_gate='Review conversation correctness, expert ownership and fluent Hindi/Hinglish before expert training. No automatic deployment.')
    except Exception as exc:
        write_status(output,'failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'runs'/'public-campaign-v1')
    p.add_argument('--wait-minutes',type=float,default=90)
    p.add_argument('--fetch',action='store_true',help='Acquire/resume sources inside this job before preparation')
    a=p.parse_args()
    if a.wait_minutes<0: p.error('wait-minutes must be nonnegative')
    run(a.output,a.wait_minutes,a.fetch)

if __name__=='__main__': main()
