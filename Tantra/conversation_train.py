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


def train(bundle,source,output,resume=False,english_pilot=False,epochs=1):
    import numpy as np
    import torch
    from Tantra.config import ModelConfig
    from Tantra.model.tantra_model import TantraModel
    from Tantra.tokenizer import TantraTokenizer
    from Tantra.data.expert_dataset import encode_row,collate
    from Tantra.data.train_base_pilot import memory_ok,atomic_checkpoint
    if epochs<1 or (epochs>1 and not english_pilot):raise ValueError('Multiple epochs are only allowed for the English pilot')
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
    if any(source_report['latest_validation'][lang]['loss']>=source_report['baseline_validation'][lang]['loss'] for lang in ('hi','en')):raise ValueError('Base language gate failed')
    if output.exists() and not resume:raise ValueError('Use new output or explicit resume')
    output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free<3*1024**3:raise OSError('Need 3 GiB free disk')
    lock=output/'training.lock'
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
            state=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
            if state['identity']!=identity:raise ValueError('Resume identity mismatch')
            model.load_state_dict(state['state']);optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'])
            cursor,updates,tokens=state['cursor'],state['updates'],state['tokens'];baseline=state['baseline']
            best=state.get('best',{'loss':float('inf'),'update':None});del state
        else:baseline=evaluate();best={'loss':float('inf'),'update':None}
        def mean_loss(result):return sum(v['loss'] for v in result.values())/len(result)
        def save():
            atomic_checkpoint(output/'latest.pt',{'config':model.cfg.to_dict(),'state':model.state_dict(),'optimizer':optimizer.state_dict(),
                'cursor':cursor,'updates':updates,'tokens':tokens,'identity':identity,'baseline':baseline,'torch_rng':torch.get_rng_state(),'order_seed':42,'best':best})
        def keep_best(result):
            # Small datasets overfit; keep the weights with the lowest validation loss seen so far.
            if mean_loss(result)<best['loss']:
                best.update(loss=mean_loss(result),update=updates,validation=result)
                atomic_checkpoint(output/'best.pt',{'config':model.cfg.to_dict(),'state':model.state_dict(),'router_names':[],'validation':result,'updates':updates,'identity':identity})
        save();status='running'
        try:
            while cursor<len(order):
                if (output/'stop.request').exists():(output/'stop.request').unlink();status='stopped_requested';break
                if not memory_ok() or shutil.disk_usage(output).free<2*1024**3:status='paused_resources';break
                # Each three-record microbatch has one example per language (English pilot: three English rows).
                ids=order[cursor:cursor+3];model.train();optimizer.zero_grad(set_to_none=True)
                batches=[batch(encoded['train'][i]) for i in ids]
                count=sum(int((labels[:,1:]!=-100).sum()) for _,labels,_ in batches)
                for x,labels,_ in batches:
                    n=int((labels[:,1:]!=-100).sum());loss=model.loss(x,labels)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('SFT loss')
                    (loss*n/count).backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not bool(torch.isfinite(norm)):raise FloatingPointError('SFT gradients')
                optimizer.step();cursor+=len(ids);updates+=1;tokens+=count
                if updates%10==0:save();print(f'Chat update {updates} | assistant targets {tokens}',flush=True)
                if updates%50==0:
                    result=evaluate();keep_best(result);print(json.dumps({'update':updates,'validation':result}),flush=True)
        except KeyboardInterrupt:status='interrupted'
        validation=evaluate();keep_best(validation);save()
        if cursor==len(order):status='english_pilot_complete_unvalidated' if english_pilot else 'sft_pass_complete_unvalidated'
        atomic_checkpoint(output/'base.pt',{'config':model.cfg.to_dict(),'state':model.state_dict(),'router_names':[]})
        shutil.copy2(source/'tokenizer.json',output/'tokenizer.json')
        report={'state':status,'updates':updates,'assistant_targets':tokens,'baseline':baseline,'validation':validation,
                'identity':identity,'deployed':False,'quality_gate':'Independent answer and router tests still required','data_readiness':readiness['state'],
                'mode':identity['mode'],'epochs':epochs,'languages':list(languages),
                'best_update':best['update'],'best_validation':best.get('validation'),'opener_overlap':readiness.get('opener_overlap'),
                'validation_rows':len(rows['val']),
                'pilot_warning':'English-only, undersized, AI-drafted/owner-approved data; not a release candidate' if english_pilot else None}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');return report
    finally:lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','train'])
    p.add_argument('--bundle',type=Path,default=ROOT/'Datasets/workbench-conversation-v2')
    p.add_argument('--source',type=Path,default=ROOT/'runs/pretraining-20m-v1/base-training')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--english-pilot',action='store_true',help='Explicit undersized English-only pilot; never data_ready or deployable')
    p.add_argument('--epochs',type=int,default=1,help='English pilot only: shuffled passes over the train split');a=p.parse_args()
    report=prepare(a.bundle,a.output) if a.command=='prepare' else train(a.bundle,a.source,a.output,a.resume,a.english_pilot,a.epochs)
    print(json.dumps({'state':report['state'],'shortages':len(report.get('shortages',[]))},indent=2))

if __name__=='__main__':main()
