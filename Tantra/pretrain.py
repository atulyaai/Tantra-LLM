"""Configured CPU base training. Uses the shared resumable trainer, not a second loop."""
import argparse
import json
from pathlib import Path
import psutil
from Tantra.config import ROOT
from Tantra.train import snapshot,display



def archive_stale_locks(output):
    """Refuse live/unknown owners; retain a recoverable copy of dead-owner locks."""
    import uuid
    output=Path(output)
    for lock in (output/'training.lock',output.parent/'worker.lock'):
        if not lock.exists():continue
        try:pid=int(lock.read_text().strip())
        except (ValueError,OSError) as exc:raise RuntimeError('Cannot verify lock owner: '+str(lock)) from exc
        if psutil.pid_exists(pid):raise RuntimeError(f'Lock owner PID {pid} is alive; inspect before resuming: {lock}')
        archived=lock.with_name(lock.name+'.stale-'+uuid.uuid4().hex)
        lock.rename(archived)
        print(f'Archived stale lock for stopped PID {pid}: {archived}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'Tantra/config/pretrain_64k.json')
    p.add_argument('--dashboard',action='store_true',help='Open a separate live dashboard window alongside training')
    p.add_argument('--resume',action='store_true');p.add_argument('--info',action='store_true')
    batch=p.add_mutually_exclusive_group()
    batch.add_argument('--benchmark-report',type=Path)
    batch.add_argument('--microbatch',type=int,choices=(1,2,4,8,16),help='Explicit RAM/batch override; effective targets remain 4096')
    p.add_argument('--threads',type=int);p.add_argument('--max-updates',type=int)
    p.add_argument('--early-stopping-patience',type=int,help='Explicit override; saved bad-eval count is kept on resume')
    a=p.parse_args();c=json.loads(a.config.read_text(encoding='utf-8'))
    for key in ('model_dir','corpus','output'):
        path=Path(c[key]);c[key]=str(path if path.is_absolute() else ROOT/path)
    if a.threads is not None:c['threads']=a.threads
    if a.max_updates is not None:c['max_updates']=a.max_updates
    if a.early_stopping_patience is not None:c['early_stopping_patience']=a.early_stopping_patience
    fixed={'learning_rate':.0001,'optimizer':'AdamW','weight_decay':.01,'sequence_targets':256,
           'tokens_per_update':4096,'microbatch':'benchmark:1,2,4','precision':'float32'}
    if any(c.get(key)!=value for key,value in fixed.items()):
        p.error('This trainer currently fixes optimizer/token-budget settings; unsupported config changes are rejected.')
    print('TRAINING CONFIGURATION',flush=True)
    print(json.dumps(c,indent=2),flush=True)
    if a.dashboard and not a.info:
        import subprocess,sys
        subprocess.Popen([sys.executable,'-u','-B','-m','Tantra.dashboard','--stage',str(Path(c['output']).parent)],cwd=ROOT,creationflags=subprocess.CREATE_NEW_CONSOLE if __import__('os').name=='nt' else 0)
    status=snapshot(Path(c['output']).parent)
    if status['worker_pids']:
        print('Existing worker owns this output. No duplicate training started.',flush=True);display(status)
        if not a.info:print('Live view: python -m Tantra.train watch',flush=True)
        return
    if a.microbatch is not None:print(f'Requested microbatch: {a.microbatch}; fixed effective update: 4096 targets',flush=True)
    if a.info:return
    try:archive_stale_locks(c['output'])
    except RuntimeError as exc:p.error(str(exc))
    override=a.microbatch
    if a.benchmark_report:
        from Tantra.data.public_sources import sha256
        report=json.loads(a.benchmark_report.read_text(encoding='utf-8'))
        if not report.get('checkpoint_unchanged') or report.get('threads')!=c['threads'] or report['checkpoint_sha256']!=sha256(Path(c['output'])/'latest.pt'):
            p.error('Benchmark report must match the unchanged checkpoint and CPU threads')
        override=report['selected_microbatch']
    from Tantra.data.train_base_pilot import run
    result=run(c['corpus'],c['output'],c['model_dir'],max_updates=c['max_updates'],resume=a.resume,
               threads=c['threads'],val_windows=c['validation_windows'],checkpoint_updates=c['checkpoint_updates'],early_stopping_patience=c.get('early_stopping_patience',0),microbatch_override=override,probe_suite=(ROOT/c['probe_suite']) if c.get('probe_suite') else None)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
