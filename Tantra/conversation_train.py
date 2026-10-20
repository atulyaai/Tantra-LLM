"""Prepare and train balanced reviewed conversation data on a finished base."""
from __future__ import annotations
import argparse
from collections import Counter
import json,math,os,random,shutil,time
from pathlib import Path
from Tantra.config import ROOT
from Tantra.data.public_sources import sha256
from Tantra.data.train_prepared import load_split

LANGUAGES=('hi','en','hinglish')
EXPERTS=('greeting','identity_self','identity_capabilities','identity_limitations','clarification','empathy','farewell','smalltalk')


def inspect(bundle,experts=EXPERTS,english_pilot=False):
    """english_pilot: English rows only, family minimums reported but not enforced.
    It never yields data_ready; its state is always english_pilot_undersized."""
    bundle=Path(bundle);meta=json.loads((bundle/'prepared/manifest.json').read_text(encoding='utf-8'))
    languages=('en',) if english_pilot else LANGUAGES
    rows={split:[] for split in ('train','val','test')};coverage={};rejected=[]
    for expert in experts:
        if english_pilot and expert not in meta['experts']:continue
        item=meta['experts'][expert];counts={lang:Counter() for lang in languages};families={lang:{s:set() for s in rows} for lang in languages}
        for split,record in item['files'].items():
            for row in load_split(bundle,record,expert):
                lang=row['language']
                if lang not in languages:continue
                if lang!='en' and not row['review'].get('fluent_language_review'):
                    rejected.append({'record_id':row.get('record_id'),'reason':'fluent_language_review_missing'});continue
                rows[split].append(row);counts[lang][split]+=1;families[lang][split].add(row['split_group'])
        coverage[expert]={lang:{s:{'rows':counts[lang][s],'families':len(families[lang][s]),'minimum':meta['experts'][expert]['minimum'][s]} for s in rows} for lang in languages}
    groups={s:{r['split_group'] for r in records} for s,records in rows.items()}
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Cross-split family leakage')
    # Families come from authoring scripts; the same opener can still sit in two splits. Report it.
    opener=lambda r:' '.join(next(m['content'] for m in r['messages'] if m['role']=='user').lower().split())
    openers={s:{opener(r) for r in records} for s,records in rows.items()}
    overlap={f'{a}/{b}':sorted(openers[a]&openers[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    shortages=[]
    for expert,languages in coverage.items():
        for lang,splits in languages.items():
            for split,c in splits.items():
                if c['families']<c['minimum']:shortages.append({'expert':expert,'language':lang,'split':split,'have_families':c['families'],'need':c['minimum']})
    if english_pilot:
        if not rows['train'] or not rows['val']:raise ValueError('English pilot needs reviewed train and validation rows')
        state='english_pilot_undersized'
    else:state='data_ready' if not shortages and not rejected else 'needs_reviewed_data'
    return rows,{'state':state,'coverage':coverage,'shortages':shortages,
                 'rejected':rejected,'bundle_manifest_sha256':sha256(bundle/'prepared/manifest.json'),
                 'policy':'No fabricated review or oversampling; require independent minimums per expert and language',
                 'opener_overlap':{k:len(v) for k,v in overlap.items()},'opener_overlap_examples':{k:v[:5] for k,v in overlap.items()}}


def balanced_order(rows,seed=42,languages=LANGUAGES):
    pools={lang:[i for i,row in enumerate(rows) if row['language']==lang] for lang in languages}
    if any(not pool for pool in pools.values()):raise ValueError('Missing reviewed conversation language')
    rng=random.Random(seed)
    for pool in pools.values():rng.shuffle(pool)
    n=min(map(len,pools.values()))
    return [pools[lang][i] for i in range(n) for lang in languages]


def pilot_order(rows,epochs,seed=42):
    """English pilot: each epoch is one shuffled pass; no row repeats within an epoch."""
    order=[]
    for epoch in range(epochs):
        order+=balanced_order(rows,seed+epoch,('en',))
    return order


def prepare(bundle,output):
    output=Path(output)
    if output.exists():raise ValueError('Use a new readiness report path')
    rows,report=inspect(bundle)
    report['accepted_rows_by_split']={s:dict(Counter(r['language'] for r in records)) for s,records in rows.items()}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def validation_progress(best, loss):
    """Update persisted validation patience; equal/worse loss is not improvement."""
    if not math.isfinite(loss):
        raise FloatingPointError('Non-finite conversation validation loss')
    improved = loss < best['loss']
    best['bad_evals'] = 0 if improved else best.get('bad_evals', 0) + 1
    return improved


def train(bundle,source,output,resume=False,english_pilot=False,epochs=1,rows_per_update=3,early_stopping_patience=5):
    import numpy as np
    import torch
    from Tantra.config import ModelConfig
    from Tantra.model.tantra_model import TantraModel
    from Tantra.tokenizer import TantraTokenizer
    from Tantra.data.expert_dataset import encode_row,collate
    from Tantra.data.train_base_pilot import memory_ok,atomic_checkpoint
    if early_stopping_patience<1:raise ValueError('Early stopping patience must be positive')
    if epochs<1 or (epochs>1 and not english_pilot):raise ValueError('Multiple epochs are only allowed for the English pilot')
    if rows_per_update<1 or (rows_per_update!=3 and not english_pilot):raise ValueError('Balanced training uses one row per language per update')
    if english_pilot:
        # Explicitly undersized English-only pilot: never data_ready, never deployable.
        rows,readiness=inspect(bundle,english_pilot=True);languages=('en',)
    else:
        rows,readiness=inspect(bundle);languages=LANGUAGES
        if readiness['state']!='data_ready':raise ValueError('Balanced reviewed conversation gate failed; run prepare for exact shortages')
    source,output=Path(source).resolve(),Path(output).resolve()
    if output==source or output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('Output must be separate from source')
    source_report=json.loads((source/'report.json').read_text(encoding='utf-8'))
    if source_report.get('state') not in ('pilot_complete','stopped_early'):raise ValueError('Finish and evaluate the base stage first')
    try:
        healthy_source=all(math.isfinite(source_report['baseline_validation'][lang]['loss']) and
            math.isfinite(source_report['latest_validation'][lang]['loss']) and
            source_report['latest_validation'][lang]['loss']<source_report['baseline_validation'][lang]['loss']
            for lang in ('hi','en'))
    except (KeyError,TypeError,ValueError):healthy_source=False
    if not healthy_source:raise ValueError('Base language gate failed: finite improving Hindi/English validation required')
    source_binding={}
    for field,filename in (('exported_base_sha256','base.pt'),('exported_tokenizer_sha256','tokenizer.json')):
        expected=source_report.get(field)
        if expected is not None:
            if sha256(source/filename)!=expected:raise ValueError('Source report artifact checksum mismatch: '+filename)
            source_binding[filename]='hash_verified'
        else:source_binding[filename]='legacy_report_unbound'
    if any(value=='legacy_report_unbound' for value in source_binding.values()):
        print('WARNING: legacy source report is not cryptographically bound to all exported assets; '
              'source hashes are recorded for this run, but this report is not release proof.',flush=True)
    if output.exists() and not resume:raise ValueError('Use new output or explicit resume')
    output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free<3*1024**3:raise OSError('Need 3 GiB free disk')
    lock=output/'training.lock'
    if lock.exists():
        # A lock left by a crashed run blocks resume; archive it only when its owner process is gone.
        import psutil
        try:pid=int(lock.read_text().strip())
        except ValueError as exc:raise RuntimeError('Cannot read lock owner: '+str(lock)) from exc
        if psutil.pid_exists(pid):raise RuntimeError(f'Training lock held by live PID {pid}: {lock}')
        lock.rename(lock.with_name(f'training.lock.stale-{pid}'))
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        torch.set_num_threads(4);torch.manual_seed(42)
        device=torch.device(os.environ.get('TANTRA_DEVICE','cpu'))
        tok=TantraTokenizer.load(source/'tokenizer.json');ck=torch.load(source/'base.pt',map_location='cpu',weights_only=False)
        model=TantraModel(ModelConfig.from_dict(ck['config']));model.load_state_dict(ck['state']);del ck;model.to(device)
        def batch(encoded_row):
            x,labels,mask=collate([encoded_row],tok.pad_id);return x.to(device),labels.to(device),mask
        if model.cfg.vocab_size!=tok.vocab_size:raise ValueError('Tokenizer mismatch')
        # Reject length overflow instead of silently dropping/truncating assistant supervision.
        for records in rows.values():
            for row in records:
                if len(tok.encode_chat(row['messages'])[0])>256:raise ValueError('Review/chunk long conversations before training')
        encoded={s:[encode_row(tok,row,256,assistant_only=True) for row in records] for s,records in rows.items()}
        order=pilot_order(rows['train'],epochs) if english_pilot else balanced_order(rows['train'])
        cursor=updates=tokens=0;optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
        identity={'bundle':readiness['bundle_manifest_sha256'],'source':sha256(source/'base.pt'),'tokenizer':sha256(source/'tokenizer.json'),
                  'mode':'english_pilot' if english_pilot else 'balanced','epochs':epochs}
        if rows_per_update!=3:identity['rows_per_update']=rows_per_update  # older runs stay resumable
        def evaluate():
            model.eval();result={}
            with torch.no_grad():
                for lang in languages:
                    total=count=0
                    for i,row in enumerate(rows['val']):
                        if row['language']!=lang:continue
                        if not memory_ok():raise MemoryError('SFT evaluation memory gate')
                        x,y,_=batch(encoded['val'][i]);n=int((y[:,1:]!=-100).sum())
                        value=float(model.loss(x,y))
                        if not math.isfinite(value):raise FloatingPointError('SFT validation loss')
                        total+=value*n;count+=n
                    if not count:raise ValueError('No validation assistant targets')
                    result[lang]={'loss':total/count,'assistant_targets':count}
            return result
        if resume:
            from Tantra.training.checkpoints import load_resumable
            state,recovery=load_resumable(output/'latest.pt',identity_key='identity',expected_identity=identity,
                expected_config=model.cfg.to_dict(),required=('state','optimizer','torch_rng','cursor','updates','tokens','baseline'),
                state_shapes=model.state_dict())
            if recovery['fallback_used'] or recovery['legacy_without_checksum']:
                print('Checkpoint resume evidence: '+json.dumps(recovery),flush=True)
            with (output/'recovery_history.jsonl').open('a',encoding='utf-8') as history:
                history.write(json.dumps(recovery)+'\n')
            model.load_state_dict(state['state']);optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'])
            cursor,updates,tokens=state['cursor'],state['updates'],state['tokens'];baseline=state['baseline']
            best=state.get('best',{'loss':float('inf'),'update':None});del state
        else:
            baseline=evaluate();best={'loss':sum(v['loss'] for v in baseline.values())/len(baseline),'update':0,'validation':baseline}
            # The untrained starting point must remain best if every trained candidate regresses.
            atomic_checkpoint(output/'best.pt',{'config':model.cfg.to_dict(),'state':model.state_dict(),
                'router_names':[],'validation':baseline,'updates':0,'identity':identity})
        def mean_loss(result):return sum(v['loss'] for v in result.values())/len(result)
        def save():
            atomic_checkpoint(output/'latest.pt',{'config':model.cfg.to_dict(),'state':model.state_dict(),'optimizer':optimizer.state_dict(),
                'cursor':cursor,'updates':updates,'tokens':tokens,'identity':identity,'baseline':baseline,'torch_rng':torch.get_rng_state(),'order_seed':42,'best':best})
        def keep_best(result):
            # Small datasets overfit; keep the weights with the lowest validation loss seen so far.
            if validation_progress(best,mean_loss(result)):
                best.update(loss=mean_loss(result),update=updates,validation=result)
                atomic_checkpoint(output/'best.pt',{'config':model.cfg.to_dict(),'state':model.state_dict(),'router_names':[],'validation':result,'updates':updates,'identity':identity})
        save();status='running'
        if best.get('bad_evals',0)>=early_stopping_patience:status='stopped_early'
        try:
            while cursor<len(order) and status=='running':
                if (output/'stop.request').exists():(output/'stop.request').unlink();status='stopped_requested';break
                if not memory_ok() or shutil.disk_usage(output).free<2*1024**3:status='paused_resources';break
                # One update = rows_per_update records (balanced mode: one per language), run as ONE right-padded
                # batch. Attention is causal and pads sit after real tokens, and loss() averages over every
                # supervised token, so this equals the old per-row token-weighted sum, with fewer passes.
                ids=order[cursor:cursor+rows_per_update];model.train();optimizer.zero_grad(set_to_none=True)
                x,labels,_=collate([encoded['train'][i] for i in ids],tok.pad_id);x,labels=x.to(device),labels.to(device)
                count=int((labels[:,1:]!=-100).sum());loss=model.loss(x,labels)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('SFT loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not bool(torch.isfinite(norm)):raise FloatingPointError('SFT gradients')
                optimizer.step();cursor+=len(ids);updates+=1;tokens+=count
                if updates%10==0:save();print(f'Chat update {updates} | assistant targets {tokens}',flush=True)
                if updates%50==0:
                    result=evaluate();keep_best(result);save();print(json.dumps({'update':updates,'validation':result,'bad_evals':best.get('bad_evals',0),'patience':early_stopping_patience}),flush=True)
                    if best.get('bad_evals',0)>=early_stopping_patience:status='stopped_early';break
        except KeyboardInterrupt:status='interrupted'
        validation=evaluate()
        # Final reporting is not another independent validation interval.
        if mean_loss(validation)<best['loss']:keep_best(validation)
        save()
        if cursor==len(order):status='english_pilot_complete_unvalidated' if english_pilot else 'sft_pass_complete_unvalidated'
        atomic_checkpoint(output/'base.pt',{'config':model.cfg.to_dict(),'state':model.state_dict(),'router_names':[]})
        shutil.copy2(output/'best.pt',output/'best-base.pt')
        shutil.copy2(source/'tokenizer.json',output/'tokenizer.json')
        report={'state':status,'updates':updates,'assistant_targets':tokens,'baseline':baseline,'validation':validation,
                'identity':identity,'deployed':False,'quality_gate':'Independent answer and router tests still required','data_readiness':readiness['state'],
                'mode':identity['mode'],'epochs':epochs,'languages':list(languages),
                'best_update':best['update'],'best_validation':best.get('validation'),'opener_overlap':readiness.get('opener_overlap'),
                'validation_rows':len(rows['val']),
                'source_report_binding':source_binding,
                'early_stopping':{'patience':early_stopping_patience,'bad_evals':best.get('bad_evals',0),'interval_updates':50},
                'selected_checkpoint':'best-base.pt','final_checkpoint':'base.pt',
                'pilot_warning':'English-only, undersized, AI-drafted/owner-approved data; not a release candidate' if english_pilot else None}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');return report
    finally:lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','train'])
    p.add_argument('--bundle',type=Path,default=ROOT/'Datasets/workbench-conversation-v2')
    p.add_argument('--source',type=Path,default=ROOT/'runs/pretraining-20m-v1/base-training')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--english-pilot',action='store_true',help='Explicit undersized English-only pilot; never data_ready or deployable')
    p.add_argument('--early-stopping-patience',type=int,default=5,help='Stop after this many validation intervals without lower loss')
    p.add_argument('--epochs',type=int,default=1,help='English pilot only: shuffled passes over the train split')
    p.add_argument('--rows-per-update',type=int,default=3,help='English pilot only: conversations per optimizer update');a=p.parse_args()
    report=prepare(a.bundle,a.output) if a.command=='prepare' else train(a.bundle,a.source,a.output,a.resume,a.english_pilot,a.epochs,a.rows_per_update,a.early_stopping_patience)
    print(json.dumps({'state':report['state'],'shortages':len(report.get('shortages',[]))},indent=2))

if __name__=='__main__':main()
