"""Resumable one-pass CPU base pilot on frozen, disk-backed bilingual tokens.

Exports a separate candidate base. Existing expert adapters are NOT transferred.
"""
from __future__ import annotations
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import psutil
import torch
from Tantra.config import ROOT, ModelConfig
from Tantra.model.tantra_model import TantraModel
from .public_sources import sha256

GIB=1024**3


def memory_ok():
    return psutil.Process().memory_info().rss < 8*GIB and psutil.virtual_memory().available >= 2*GIB


def window_order(files, sequence=256):
    queues={lang:deque((lang,i*sequence) for i in range((len(files[lang])-1)//sequence)) for lang in ('hi','en')}
    result=[]
    while any(queues.values()):
        for lang in ('hi','en'):
            if queues[lang]: result.append(queues[lang].popleft())
    return result


def tensor_batch(files, positions, sequence):
    return torch.from_numpy(np.stack([np.array(files[lang][offset:offset+sequence+1],dtype=np.int64)
                                     for lang,offset in positions]))


def atomic_checkpoint(path, payload):
    if Path(path).name=='latest.pt':
        from Tantra.training.checkpoints import save_resumable
        save_resumable(path,payload);return
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as writer:
        torch.save(payload,writer); writer.flush(); os.fsync(writer.fileno())
    os.replace(temp,path)


def run(corpus, output, model_dir=ROOT/'Model', max_updates=0, resume=False, threads=4, val_windows=16, checkpoint_updates=100, early_stopping_patience=0, microbatch_override=None, probe_suite=None):
    corpus,output,model_dir=Path(corpus).resolve(),Path(output).resolve(),Path(model_dir).resolve()
    if output==model_dir or output.is_relative_to(model_dir) or model_dir.is_relative_to(output):
        raise ValueError('Output must be separate from source model')
    if min(threads,val_windows,checkpoint_updates)<1 or max_updates<0 or early_stopping_patience<0: raise ValueError('Invalid training limits')
    if microbatch_override is not None and microbatch_override not in (1,2,4,8,16):raise ValueError('Unsupported microbatch override')
    manifest_path=corpus/'manifest.json'; manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest['ready']: raise ValueError('Corpus pilot quotas not met')
    identities={'tokenizer_sha256':sha256(model_dir/'tokenizer.json'),
                'source_base_sha256':sha256(model_dir/'base.pt'),'corpus_manifest_sha256':sha256(manifest_path)}
    if identities['tokenizer_sha256']!=manifest['tokenizer_sha256']: raise ValueError('Tokenizer mismatch')
    maps={}
    for name,record in manifest['files'].items():
        path=corpus/record['path']
        if sha256(path)!=record['sha256']: raise ValueError('Corpus shard checksum mismatch')
        maps[name]=np.memmap(path,mode='r',dtype='<u4')
    train={lang:maps[lang+'-train'] for lang in ('hi','en')}
    sequence=256; order=window_order(train,sequence)
    if not order: raise ValueError('No full training windows')
    import hashlib
    identities['window_order_sha256']=hashlib.sha256(json.dumps(order).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    if not resume and any(output.iterdir()): raise ValueError('Use a new output or --resume')
    if shutil.disk_usage(output).free<2*GIB: raise OSError('Need at least 2 GiB free disk')
    lock=output/'training.lock'
    with lock.open('x',encoding='utf-8') as handle: handle.write(str(os.getpid()))
    try:
        return _train(output,model_dir,identities,maps,train,order,sequence,max_updates,resume,threads,val_windows,checkpoint_updates,early_stopping_patience,microbatch_override,probe_suite)
    finally:
        lock.unlink()


def _train(output,model_dir,identities,maps,train,order,sequence,max_updates,resume,threads,val_windows,checkpoint_updates=100, early_stopping_patience=0, microbatch_override=None, probe_suite=None):
    # TANTRA_DEVICE=cuda lets the same resumable loop run on a hosted GPU; CPU stays the default.
    device=torch.device(os.environ.get('TANTRA_DEVICE','cpu'))
    torch.set_num_threads(threads); torch.manual_seed(42); random.seed(42); np.random.seed(42)
    source=torch.load(model_dir/'base.pt',map_location='cpu',weights_only=False)
    cfg=ModelConfig.from_dict(source['config'])
    if cfg.max_seq_len<sequence+1: raise ValueError('Base context too short')
    print(f'Model: width={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} vocab={cfg.vocab_size} context={cfg.max_seq_len}',flush=True)
    model=TantraModel(cfg)
    print(f'Base parameters: {model.num_params():,} | AdamW lr=0.0001 | FP32 | threads={threads}',flush=True)
    state={k:v for k,v in source['state'].items() if not k.startswith('router.') and 'lora_' not in k}
    missing,unexpected=model.load_state_dict(state,strict=False)
    if unexpected or any(not k.startswith('router.') for k in missing): raise ValueError('Incompatible base')
    del source,state
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.95))
    cursor=updates=tokens=0; peak=0; benchmarks=[]; best=float('inf'); base_eval={}
    chosen=1
    last=output/'latest.pt'
    bad_evals=0;saved_protocol=None
    if resume:
        from Tantra.training.checkpoints import load_resumable
        saved,recovery=load_resumable(last,identity_key='identities',expected_identity=identities,
            expected_config=cfg.to_dict(),expected_fields={'threads':threads,'sequence':sequence},
            required=('state','optimizer','torch_rng','python_rng','numpy_rng','cursor','updates','tokens','microbatch',
                      'benchmarks','baseline_validation','best_validation'),state_shapes=model.state_dict())
        if recovery['fallback_used'] or recovery['legacy_without_checksum']:
            print('Checkpoint resume evidence: '+json.dumps(recovery),flush=True)
        with (output/'recovery_history.jsonl').open('a',encoding='utf-8') as history:
            history.write(json.dumps(recovery)+'\n')
        model.load_state_dict(saved['state']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']); random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        cursor,updates,tokens=saved['cursor'],saved['updates'],saved['tokens']
        chosen=saved['microbatch']; benchmarks=saved['benchmarks']; base_eval=saved['baseline_validation']; best=saved['best_validation']
        bad_evals=saved.get('bad_evals',0);saved_protocol=saved.get('evaluation_protocol')
        del saved
    else:
        # Backward-only benchmarks leave weights unchanged; clear all gradients.
        model.train()
        for batch_size in (1,2,4):
            if not memory_ok(): break
            x=tensor_batch(train,order[:batch_size],sequence).to(device)
            start=time.perf_counter()
            try:
                for _ in range(2):
                    model.zero_grad(set_to_none=True); model.loss(x,x).backward()
                seconds=time.perf_counter()-start
                rss=psutil.Process().memory_info().rss; peak=max(peak,rss)
                benchmarks.append({'microbatch':batch_size,'tokens_per_second':2*len(x)*sequence/seconds,
                                   'rss_bytes':rss,'within_memory':memory_ok()})
            except (MemoryError,RuntimeError) as exc:
                if isinstance(exc,RuntimeError) and 'memory' not in str(exc).lower(): raise
                benchmarks.append({'microbatch':batch_size,'error':str(exc),'within_memory':False})
            finally: model.zero_grad(set_to_none=True)
        viable=[b for b in benchmarks if b.get('within_memory')]
        if not viable: raise MemoryError('No safe microbatch configuration')
        chosen=max(viable,key=lambda b:b['tokens_per_second'])['microbatch']
    if microbatch_override is not None:
        if microbatch_override not in (1,2,4,8,16):raise ValueError('Unsupported microbatch override')
        print(f'Explicit microbatch: {chosen} -> {microbatch_override}; effective targets remain 4096',flush=True)
        chosen=microbatch_override
    def evaluate():
        result={}; model.eval()
        with torch.no_grad():
            for lang in ('hi','en'):
                data=maps[lang+'-val']; available=(len(data)-1)//sequence
                n=min(val_windows,available)
                if n<1: raise ValueError('Not enough validation windows')
                total=0.;correct1=correct5=0
                for i in range(n):
                    if not memory_ok(): raise MemoryError('Validation memory gate failed')
                    offset=(i*(available-1)//max(1,n-1))*sequence
                    x=tensor_batch({lang:data},[(lang,offset)],sequence).to(device)
                    logits=model(x)[:,:-1].float();targets=x[:,1:]
                    value=float(torch.nn.functional.cross_entropy(logits.reshape(-1,cfg.vocab_size),targets.reshape(-1)))
                    top=logits.topk(5,dim=-1).indices
                    correct1+=int((top[:,:,0]==targets).sum())
                    correct5+=int((top==targets.unsqueeze(-1)).any(-1).sum())
                    if not math.isfinite(value): raise FloatingPointError('Non-finite validation')
                    total+=value
                result[lang]={'loss':total/n,'perplexity':math.exp(min(80,total/n)),'top1_percent':100*correct1/(n*sequence),'top5_percent':100*correct5/(n*sequence),'target_tokens':n*sequence,'windows':n,'sampling':'evenly spaced across held-out token shard'}
        return result
    protocol={'sampling':'spread-v1','windows':val_windows}
    if resume and saved_protocol!=protocol:
        print('Evaluation protocol changed; resetting baseline/best/patience for comparable scores.',flush=True)
        base_eval={};best=float('inf');bad_evals=0
    if not base_eval: base_eval=evaluate()
    # The starting weights are the bar to beat: a worse first evaluation must not become best-base.pt.
    if best==float('inf'): best=sum(v['loss'] for v in base_eval.values())/2
    metadata={'model':cfg.to_dict(),'base_parameters':model.num_params(),'optimizer':'AdamW','precision':'float32',
              'threads':threads,'microbatch':chosen,'tokens_per_update':4096,'sequence_targets':sequence,
              'evaluation_every':50,'validation_windows':val_windows,'checkpoint_updates':checkpoint_updates,
              'pid':os.getpid(),'evaluation_protocol':protocol,'device':str(device)}
    metadata_temp=output/'runtime.tmp'
    metadata_temp.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    os.replace(metadata_temp,output/'runtime.json')
    print('Baseline validation: '+json.dumps(base_eval),flush=True)
    started=time.perf_counter(); last_save=started; run_start_tokens=tokens
    def checkpoint():
        if shutil.disk_usage(output).free<1024**3: raise OSError('Checkpoint disk gate failed')
        payload={'config':cfg.to_dict(),'state':model.state_dict(),'optimizer':optimizer.state_dict(),
                 'identities':identities,'cursor':cursor,'updates':updates,'tokens':tokens,'sequence':sequence,
                 'microbatch':chosen,'threads':threads,'benchmarks':benchmarks,'baseline_validation':base_eval,
                 'best_validation':best,'bad_evals':bad_evals,'evaluation_protocol':protocol,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
                 'numpy_rng':np.random.get_state(),'scheduler':{'kind':'constant','lr':1e-4},'router_names':[]}
        atomic_checkpoint(last,payload)
        print(f'Checkpoint saved: {last} | update={updates} | tokens={tokens:,}',flush=True)
    history=output/'metrics.jsonl'
    checkpoint()
    state='running'; validation=base_eval
    try:
        while cursor<len(order) and (not max_updates or updates<max_updates):
            if (output/'stop.request').exists():
                (output/'stop.request').unlink();state='stopped_requested';break
            if not memory_ok(): state='paused_memory'; break
            if shutil.disk_usage(output).free<2*GIB: state='paused_disk'; break
            # One update sees up to 4096 real next-token targets. No padding or duplicate epochs.
            positions=order[cursor:cursor+16]; model.train(); optimizer.zero_grad(set_to_none=True)
            weighted=0.
            for start in range(0,len(positions),chosen):
                if not memory_ok():raise MemoryError('Microbatch memory gate failed; latest healthy checkpoint preserved')
                chunk=positions[start:start+chosen]; x=tensor_batch(train,chunk,sequence).to(device)
                loss=model.loss(x,x)
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite training loss')
                (loss*(len(chunk)/len(positions))).backward(); weighted+=float(loss.detach())*len(chunk)/len(positions)
                peak=max(peak,psutil.Process().memory_info().rss)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not bool(torch.isfinite(norm)): raise FloatingPointError('Non-finite gradient')
            optimizer.step(); cursor+=len(positions); updates+=1; tokens+=len(positions)*sequence
            event={'update':updates,'loss':weighted,'target_tokens':tokens,'rss_bytes':peak,'learning_rate':optimizer.param_groups[0]['lr'],'gradient_norm':float(norm),'timestamp':time.time()}
            if updates%50==0 or cursor==len(order):
                validation=evaluate(); event['validation']=validation
                print(f'Evaluation at update {updates}: '+json.dumps(validation),flush=True)
                mean=sum(v['loss'] for v in validation.values())/2
                if mean<best:
                    best=mean;bad_evals=0
                    atomic_checkpoint(output/'best-base.pt',{'config':cfg.to_dict(),'state':model.state_dict(),
                        'router_names':[],'identities':identities,'validation':validation,'updates':updates})
                else:bad_evals+=1
                if probe_suite:
                    from Tantra.training_probes import run_probes
                    from Tantra.tokenizer import TantraTokenizer
                    run_probes(model,TantraTokenizer.load(model_dir/'tokenizer.json'),probe_suite,updates,generate=(updates%200==0 or cursor==len(order)),output=output)
            with history.open('a',encoding='utf-8') as f: f.write(json.dumps(event)+'\n')
            if updates%10==0:
                elapsed_now=time.perf_counter()-started;rate=(tokens-run_start_tokens)/max(elapsed_now,1e-9)
                eta=(len(order)-cursor)*sequence/max(rate,1e-9)/3600
                print(f'Update {updates}/{math.ceil(len(order)/16)} | loss {weighted:.4f} | lr {optimizer.param_groups[0]['lr']:.2e} | grad {float(norm):.2f} | tokens {tokens:,} | {rate:.1f} tok/s | ETA {eta:.2f}h | RSS {peak/GIB:.2f} GiB',flush=True)
            if updates%checkpoint_updates==0 or time.perf_counter()-last_save>=600:
                checkpoint(); last_save=time.perf_counter()
            if early_stopping_patience and bad_evals>=early_stopping_patience:
                state='stopped_early';print('Early stopping: validation stopped improving.',flush=True);break
        if state=='running': state='pilot_complete' if cursor==len(order) else 'paused_update_limit'
        validation=evaluate(); checkpoint()
        final={'config':cfg.to_dict(),'state':model.state_dict(),'router_names':[],
               'identities':identities,'validation':validation,'updates':updates}
        if state=='stopped_early' and (output/'best-base.pt').exists():
            # Early stopping means later weights were worse; promote the best evaluated weights.
            final=torch.load(output/'best-base.pt',map_location='cpu',weights_only=False); final['promoted_from']='best-base.pt'
        atomic_checkpoint(output/'base.pt',final)
        shutil.copy2(model_dir/'tokenizer.json',output/'tokenizer.json')
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        # Discard partial accumulation; cursor still marks the last complete optimizer update.
        checkpoint(); state='interrupted'
    except Exception as exc:
        # Preserve last healthy checkpoint rather than saving potentially unhealthy weights.
        (output/'failure.json').write_text(json.dumps({'error':type(exc).__name__+': '+str(exc),
              'updates':updates,'cursor':cursor,'recovery':'Inspect diagnostics; resume latest healthy checkpoint explicitly.'},indent=2),encoding='utf-8')
        raise
    elapsed=time.perf_counter()-started
    report={'state':state,'updates':updates,'trained_target_tokens':tokens,'total_train_windows':len(order),
            'remaining_windows':len(order)-cursor,'baseline_validation':base_eval,'latest_validation':validation,'evaluation_protocol':protocol,'bad_evals':bad_evals,
            'benchmarks':benchmarks,'microbatch':chosen,'tokens_per_update':4096,'sampled_peak_rss_bytes':peak,
            'run_seconds':elapsed,'run_tokens_per_second':(tokens-run_start_tokens)/elapsed if elapsed else 0,
            'identities':identities,'source_model_changed':False,'expert_adapters_transferred':False,
            'exported_base_sha256':sha256(output/'base.pt') if (output/'base.pt').exists() else None,
            'exported_tokenizer_sha256':sha256(output/'tokenizer.json') if (output/'tokenizer.json').exists() else None,
            'limitations':['One-pass 1M-token pilot, not full pretraining','Validation loss is not conversation or factual accuracy',
                           'Checksum-checked corruption fallback retains three generations; resume remains explicit and never falls back across identity/configuration mismatch',
                           'Base updates invalidate adapter compatibility; re-evaluate/retrain experts against this exact base']}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,default=ROOT/'Model'); p.add_argument('--max-updates',type=int,default=0)
    p.add_argument('--resume',action='store_true'); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--val-windows',type=int,default=16)
    p.add_argument('--checkpoint-updates',type=int,default=100)
    a=p.parse_args(); print(json.dumps(run(a.corpus,a.output,a.model_dir,a.max_updates,a.resume,a.threads,a.val_windows,a.checkpoint_updates),indent=2),flush=True)

if __name__=='__main__': main()
