"""Colored animated read-only dashboard: python -m Tantra.dashboard."""
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
from Tantra.config import ROOT
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
    config=read_json(ROOT/'Tantra/config/pretrain_64k.json')
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

if __name__=='__main__':main()
