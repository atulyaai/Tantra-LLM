"""Configured CPU base training. Uses the shared resumable trainer, not a second loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import psutil
from Tantra import ROOT
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
    p.add_argument('--config',type=Path,default=ROOT/'Tantra/pretrain_64k.json')
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
        import subprocess, sys
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
        from Tantra import sha256
        report=json.loads(a.benchmark_report.read_text(encoding='utf-8'))
        if not report.get('checkpoint_unchanged') or report.get('threads')!=c['threads'] or report['checkpoint_sha256']!=sha256(Path(c['output'])/'latest.pt'):
            p.error('Benchmark report must match the unchanged checkpoint and CPU threads')
        override=report['selected_microbatch']
    result=pilot_run(c['corpus'],c['output'],c['model_dir'],max_updates=c['max_updates'],resume=a.resume,
               threads=c['threads'],val_windows=c['validation_windows'],checkpoint_updates=c['checkpoint_updates'],early_stopping_patience=c.get('early_stopping_patience',0),microbatch_override=override,probe_suite=(ROOT/c['probe_suite']) if c.get('probe_suite') else None)
    print(json.dumps(result,indent=2),flush=True)


# ──  data/public_sources.py  ──
"""Revision-pinned, checksum-verified public data acquisition (no paid services)."""
import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from Tantra import ROOT

LOCK = ROOT/'docs'/'data_sources.lock.json'
RAW = ROOT/'Datasets'/'raw'/'public-v1'




def sources():
    return json.loads(LOCK.read_text(encoding='utf-8'))['sources']


def location(source, root=RAW):
    return Path(root)/source['id']/Path(source['path']).name


def fetch(source, root=RAW):
    path = location(source, root)
    if path.exists():
        if path.stat().st_size != source['size'] or sha256(path) != source['sha256']:
            raise ValueError('Existing source checksum mismatch: '+str(path))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < source['size'] + 2*1024**3:
        raise OSError('Need source size plus 2 GiB free disk')
    url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['path']}?download=true"
    temp = path.with_suffix(path.suffix+'.part')
    # Resume only complete downloaded ranges; final full-file hash remains mandatory.
    chunk_size=4*1024*1024
    size=temp.stat().st_size if temp.exists() else 0
    if size>source['size'] or (size!=source['size'] and size%chunk_size):
        raise ValueError('Partial download has an invalid range boundary')
    h=hashlib.sha256()
    if size:
        with temp.open('rb') as reader:
            for chunk in iter(lambda:reader.read(1024*1024),b''): h.update(chunk)
    import time
    with temp.open('ab') as writer:
        for start in range(size,source['size'],chunk_size):
            end=min(source['size']-1,start+chunk_size-1)
            expected=f"bytes {start}-{end}/{source['size']}"
            for attempt in range(3):
                try:
                    request=urllib.request.Request(url+f'&range_start={start}',headers={
                        'User-Agent':'Tantra-dataset-pipeline/1','Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request,timeout=45) as response:
                        if response.status!=206 or response.headers.get('Content-Range')!=expected:
                            raise ValueError('Server did not honor exact byte range')
                        block=response.read(end-start+2)
                        if len(block)!=end-start+1: raise ValueError('Truncated source range')
                    break
                except (OSError,ValueError):
                    if attempt==2: raise
                    time.sleep(2*(attempt+1))
            writer.write(block);writer.flush();h.update(block);size+=len(block)
            if start%(32*1024*1024)==0:
                print(json.dumps({'source':source['id'],'downloaded_bytes':size,'total_bytes':source['size']}),flush=True)
    if size!=source['size'] or h.hexdigest()!=source['sha256']:
        raise ValueError('Downloaded source checksum mismatch; partial retained for diagnosis')
    os.replace(temp,path)
    # Retain attribution and dataset card at the same revision, beside source data.
    for filename in ('README.md',):
        card = f"https://huggingface.co/datasets/{source['repo']}/raw/{source['revision']}/{filename}"
        with urllib.request.urlopen(card, timeout=60) as response:
            (path.parent/filename).write_bytes(response.read(2*1024*1024))
    return path


def sources_cli():
    from concurrent.futures import ThreadPoolExecutor
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ids', nargs='+')
    a = p.parse_args()
    known={s['id']:s for s in sources()}
    if a.ids and set(a.ids)-set(known): p.error('Unknown source ID')
    items=[known[name] for name in dict.fromkeys(a.ids)] if a.ids else list(known.values())
    if sum(s['size'] for s in items)>50*1024**3: p.error('Initial source budget exceeds 50 GiB')
    if not items: p.error('No matching source IDs')
    # At most two downloads; never a corpus-sized in-memory buffer.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for source, path in zip(items, pool.map(fetch, items)):
            print(json.dumps({'source':source['id'],'path':str(path),'sha256':source['sha256']}), flush=True)



# ──  data/prepare_corpus.py  ──
"""Stream pinned articles into immutable disk-backed Hindi/English token splits."""
import argparse
from collections import Counter
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
import numpy as np
from Tantra import ROOT
from Tantra.tokenizer import TantraTokenizer
from Tantra import sha256


def split_for(family):
    value = int(hashlib.sha256(family.encode()).hexdigest(), 16) % 100
    return 'train' if value < 90 else 'val' if value < 95 else 'test'


def prepare(output, model_dir=ROOT/'Model', train_tokens=1_000_000, exclude_corpora=()):
    import pyarrow.parquet as pq
    output, model_dir = Path(output), Path(model_dir)
    if output.exists(): raise ValueError('Corpus output must be a new immutable directory')
    if train_tokens < 1000: raise ValueError('At least 1000 pilot tokens required')
    excluded_families=set();excluded_texts=set();exclusions=[]
    for prior in exclude_corpora:
        prior=Path(prior);meta=json.loads((prior/'manifest.json').read_text(encoding='utf-8'))
        attribution=prior/'attribution.jsonl'
        if sha256(attribution)!=meta['attribution_sha256']: raise ValueError('Prior attribution checksum mismatch')
        with attribution.open(encoding='utf-8') as reader:
            for line in reader:
                record=json.loads(line);excluded_families.add(record['family_id']);excluded_texts.add(record['text_sha256'])
        exclusions.append({'manifest':str(prior/'manifest.json'),'sha256':sha256(prior/'manifest.json')})
    output.mkdir(parents=True)
    tok = TantraTokenizer.load(model_dir/'tokenizer.json')
    scope=json.loads((ROOT/'Tantra/training_scope.json').read_text(encoding='utf-8'))
    shares=scope['base_token_shares']
    if set(shares)!={'hi','en'} or any(v<=0 for v in shares.values()) or abs(sum(shares.values())-1)>1e-9:
        raise ValueError('Invalid bilingual token shares')
    budgets = {(lang, split): int(train_tokens * share * (1 if split=='train' else 1/18))
               for lang, share in shares.items() for split in ('train','val','test')}
    counts, filters, documents = Counter(), Counter(), Counter()
    writers = {key:(output/f'{key[0]}-{key[1]}.bin').open('wb') for key in budgets}
    db = sqlite3.connect(output/'dedup.sqlite')
    db.execute('CREATE TABLE documents (hash TEXT PRIMARY KEY, family TEXT, split TEXT)')
    try:
        with (output/'attribution.jsonl').open('w', encoding='utf-8') as attribution:
            for source in sources():
                if source['kind'] != 'encyclopedia': continue
                path = location(source)
                if sha256(path) != source['sha256']: raise ValueError('Source checksum mismatch')
                lang = source['language']
                parquet = pq.ParquetFile(path)
                for batch in parquet.iter_batches(batch_size=64, columns=['id','url','title','text']):
                    for row in batch.to_pylist():
                        filters[f'{lang}:seen'] += 1
                        text = row['text'].strip()
                        if len(text)<200 or len(text)>200_000 or '\x00' in text:
                            filters[f'{lang}:length_or_malformed'] += 1; continue
                        family = f"wikipedia:{lang}:{row['id']}"
                        if family in excluded_families:
                            filters[f'{lang}:prior_family']+=1;continue
                        split = split_for(family); key = (lang, split)
                        if counts[key]>=budgets[key]: continue
                        h = hashlib.sha256(text.encode()).hexdigest()
                        if h in excluded_texts or db.execute('SELECT 1 FROM documents WHERE hash=?', (h,)).fetchone():
                            filters[f'{lang}:duplicate'] += 1; continue
                        ids = tok.encode(text)
                        # Reserve one EOS token; count actual text tokens separately below.
                        ids = ids[:max(0, budgets[key]-counts[key]-1)]
                        # A final EOS-only boundary can consume the remaining slot.
                        if not ids and budgets[key]-counts[key]!=1: continue
                        db.execute('INSERT INTO documents VALUES (?,?,?)', (h, family, split))
                        values = np.asarray(ids+[tok.end_id], dtype='<u4')
                        writers[key].write(values.tobytes())
                        counts[key] += len(values); documents[key] += 1
                        attribution.write(json.dumps({'family_id':family,'language':lang,'split':split,
                            'url':row['url'],'title':row['title'],'text_sha256':h,'text_tokens':len(ids),
                            'source_id':source['id'],'revision':source['revision'],'license':source['license']},ensure_ascii=False)+'\n')
                    if all(counts[(lang,s)]>=budgets[(lang,s)] for s in ('train','val','test')): break
                db.commit()
    finally:
        db.close()
        for writer in writers.values(): writer.close()
    files = {f'{lang}-{split}': {'path':f'{lang}-{split}.bin','sha256':sha256(output/f'{lang}-{split}.bin'),
              'tokens':counts[(lang,split)],'text_tokens':counts[(lang,split)]-documents[(lang,split)],
              'documents':documents[(lang,split)],'target':target}
             for (lang,split),target in budgets.items()}
    manifest = {'excluded_corpora':exclusions,'language_token_shares':shares,'training_scope_sha256':sha256(ROOT/'Tantra/training_scope.json'),'version':1,'purpose':'Base language-model pilot, not expert answer supervision',
                'tokenizer_sha256':sha256(model_dir/'tokenizer.json'),'token_dtype':'<u4',
                'split_policy':'article family SHA256 modulo 100, 90/5/5; exact text dedup before acceptance',
                'source_lock_sha256':sha256(ROOT/'docs'/'data_sources.lock.json'),
                'files':files,'filters':dict(filters),'ready':all(v['tokens']>=v['target'] for v in files.values()),
                'limitations':['Pilot uses one source shard per language; not a representative complete corpus',
                              'Wikipedia text is not verified expert QA or current factual truth',
                              'Only exact text deduplication; cross-language/semantic duplicates need further review'],
                'attribution_sha256':sha256(output/'attribution.jsonl')}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


def prepare_corpus_cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-tokens',type=int,default=1_000_000)
    p.add_argument('--model-dir',type=Path,default=ROOT/'Model')
    p.add_argument('--exclude-corpus',type=Path,action='append',default=[])
    a=p.parse_args(); print(json.dumps(prepare(a.output,a.model_dir,a.train_tokens,a.exclude_corpus),indent=2))



# ──  data/train_base_pilot.py  ──
"""Resumable one-pass CPU base pilot on frozen, disk-backed bilingual tokens.

Exports a separate candidate base. Existing expert adapters are NOT transferred.
"""
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
from Tantra import ROOT, ModelConfig
from Tantra.train import TantraModel
from Tantra import sha256

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
        from Tantra.train import save_resumable
        save_resumable(path,payload);return
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as writer:
        torch.save(payload,writer); writer.flush(); os.fsync(writer.fileno())
    os.replace(temp,path)


def pilot_run(corpus, output, model_dir=ROOT/'Model', max_updates=0, resume=False, threads=4, val_windows=16, checkpoint_updates=100, early_stopping_patience=0, microbatch_override=None, probe_suite=None):
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
    # TANTRA_DEVICE=cuda lets the same resumable loop pilot_run on a hosted GPU; CPU stays the default.
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
        from Tantra.train import load_resumable
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
        for batch_size in (1,2,4,8,16):
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
                    from Tantra.diagnostics import run_probes
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


def pilot_cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,default=ROOT/'Model'); p.add_argument('--max-updates',type=int,default=0)
    p.add_argument('--resume',action='store_true'); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--val-windows',type=int,default=16)
    p.add_argument('--checkpoint-updates',type=int,default=100)
    p.add_argument('--microbatch',type=int,choices=(1,2,4,8,16),default=None,
                   help='Explicit accumulation microbatch; one update still sees 4096 targets')
    a=p.parse_args(); print(json.dumps(pilot_run(a.corpus,a.output,a.model_dir,a.max_updates,a.resume,a.threads,a.val_windows,a.checkpoint_updates,microbatch_override=a.microbatch),indent=2),flush=True)



# ──  data/continue_pretraining.py  ──
"""Prepare new non-overlapping data, then continue pretraining after pilot gates."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time
import psutil
from Tantra import sha256


def pilot_passed(report):
    if report.get('state')!='pilot_complete': return False
    try:
        return all(math.isfinite(report['baseline_validation'][lang]['loss']) and math.isfinite(report['latest_validation'][lang]['loss']) and
                   report['latest_validation'][lang]['loss']<report['baseline_validation'][lang]['loss']
                   for lang in ('hi','en'))
    except (KeyError,TypeError): return False


def continue_run(output,pilot,initial_model,new_tokens=19_000_000):
    train = continue_run
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


def continue_cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--initial-model',type=Path,required=True)
    p.add_argument('--new-tokens',type=int,default=19_000_000);a=p.parse_args()
    if not 1000<=a.new_tokens<=19_000_000:p.error('New-stage token budget must be 1000..19000000')
    a.output.mkdir(parents=True,exist_ok=True);lock=a.output/'worker.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:continue_run(a.output,a.pilot,a.initial_model,a.new_tokens)
    except Exception as exc:
        write_status(a.output,'failed',error=type(exc).__name__+': '+str(exc));raise
    finally:lock.unlink()



# ──  data/public_training_job.py  ──
"""Run the public-data pilot sequentially, stopping at missing data/quality gates."""
import argparse
import json
import os
from pathlib import Path
import time
from Tantra import ROOT


def write_status(output, stage, **details):
    temp=output/'status.tmp'
    temp.write_text(json.dumps({'stage':stage,**details},indent=2)+'\n',encoding='utf-8')
    os.replace(temp,output/'status.json')
    print(json.dumps({'stage':stage,**details}),flush=True)


def job_run(output, wait_minutes=90, fetch_sources=False):
    prepare_corpus = prepare
    from Tantra.conversation_train import cs_prepare as prepare_conversations
    train_base = job_run
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    lock=output/'job.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        if fetch_sources:
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


def job_cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'runs'/'public-campaign-v1')
    p.add_argument('--wait-minutes',type=float,default=90)
    p.add_argument('--fetch',action='store_true',help='Acquire/resume sources inside this job before preparation')
    a=p.parse_args()
    if a.wait_minutes<0: p.error('wait-minutes must be nonnegative')
    job_run(a.output,a.wait_minutes,a.fetch)


if __name__ == '__main__':
    main()
