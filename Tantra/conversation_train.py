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


def inspect(bundle,experts=EXPERTS):
    bundle=Path(bundle);meta=json.loads((bundle/'prepared/manifest.json').read_text(encoding='utf-8'))
    rows={split:[] for split in ('train','val','test')};coverage={};rejected=[]
    for expert in experts:
        item=meta['experts'][expert];counts={lang:Counter() for lang in LANGUAGES};families={lang:{s:set() for s in rows} for lang in LANGUAGES}
        for split,record in item['files'].items():
            for row in load_split(bundle,record,expert):
                lang=row['language']
                if lang not in LANGUAGES:continue
                if lang!='en' and not row['review'].get('fluent_language_review'):
                    rejected.append({'record_id':row.get('record_id'),'reason':'fluent_language_review_missing'});continue
                rows[split].append(row);counts[lang][split]+=1;families[lang][split].add(row['split_group'])
        coverage[expert]={lang:{s:{'rows':counts[lang][s],'families':len(families[lang][s]),'minimum':meta['experts'][expert]['minimum'][s]} for s in rows} for lang in LANGUAGES}
    groups={s:{r['split_group'] for r in records} for s,records in rows.items()}
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Cross-split family leakage')
    shortages=[]
    for expert,languages in coverage.items():
        for lang,splits in languages.items():
            for split,c in splits.items():
                if c['families']<c['minimum']:shortages.append({'expert':expert,'language':lang,'split':split,'have_families':c['families'],'need':c['minimum']})
    return rows,{'state':'data_ready' if not shortages and not rejected else 'needs_reviewed_data','coverage':coverage,'shortages':shortages,
                 'rejected':rejected,'bundle_manifest_sha256':sha256(bundle/'prepared/manifest.json'),
                 'policy':'No fabricated review or oversampling; require independent minimums per expert and language'}


def balanced_order(rows,seed=42):
    pools={lang:[i for i,row in enumerate(rows) if row['language']==lang] for lang in LANGUAGES}
    if any(not pool for pool in pools.values()):raise ValueError('Missing reviewed conversation language')
    rng=random.Random(seed)
    for pool in pools.values():rng.shuffle(pool)
    n=min(map(len,pools.values()))
    return [pools[lang][i] for i in range(n) for lang in LANGUAGES]


def prepare(bundle,output):
    output=Path(output)
    if output.exists():raise ValueError('Use a new readiness report path')
    rows,report=inspect(bundle)
    report['accepted_rows_by_split']={s:dict(Counter(r['language'] for r in records)) for s,records in rows.items()}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def train(bundle,source,output,resume=False):
    import numpy as np
    import torch
    from Tantra.config import ModelConfig
    from Tantra.model.tantra_model import TantraModel
    from Tantra.tokenizer import TantraTokenizer
    from Tantra.data.expert_dataset import encode_row,collate
    from Tantra.data.train_base_pilot import memory_ok,atomic_checkpoint
    rows,readiness=inspect(bundle)
    if readiness['state']!='data_ready':raise ValueError('Balanced reviewed conversation gate failed; run prepare for exact shortages')
    source,output=Path(source).resolve(),Path(output).resolve()
    if output==source or output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('Output must be separate from source')
    source_report=json.loads((source/'report.json').read_text(encoding='utf-8'))
    if source_report.get('state')!='pilot_complete':raise ValueError('Finish and evaluate the base stage first')
    if any(source_report['latest_validation'][lang]['loss']>=source_report['baseline_validation'][lang]['loss'] for lang in ('hi','en')):raise ValueError('Base language gate failed')
    if output.exists() and not resume:raise ValueError('Use new output or explicit resume')
    output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free<3*1024**3:raise OSError('Need 3 GiB free disk')
    lock=output/'training.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        torch.set_num_threads(4);torch.manual_seed(42)
        tok=TantraTokenizer.load(source/'tokenizer.json');ck=torch.load(source/'base.pt',map_location='cpu',weights_only=False)
        model=TantraModel(ModelConfig.from_dict(ck['config']));model.load_state_dict(ck['state']);del ck
        if model.cfg.vocab_size!=tok.vocab_size:raise ValueError('Tokenizer mismatch')
        # Reject length overflow instead of silently dropping/truncating assistant supervision.
        for records in rows.values():
            for row in records:
                if len(tok.encode_chat(row['messages'])[0])>256:raise ValueError('Review/chunk long conversations before training')
        encoded={s:[encode_row(tok,row,256,assistant_only=True) for row in records] for s,records in rows.items()}
        order=balanced_order(rows['train']);cursor=updates=tokens=0;optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
        identity={'bundle':readiness['bundle_manifest_sha256'],'source':sha256(source/'base.pt'),'tokenizer':sha256(source/'tokenizer.json')}
        def evaluate():
            model.eval();result={}
            with torch.no_grad():
                for lang in LANGUAGES:
                    total=count=0
                    for i,row in enumerate(rows['val']):
                        if row['language']!=lang:continue
                        if not memory_ok():raise MemoryError('SFT evaluation memory gate')
                        x,y,_=collate([encoded['val'][i]],tok.pad_id);n=int((y[:,1:]!=-100).sum())
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
            cursor,updates,tokens=state['cursor'],state['updates'],state['tokens'];baseline=state['baseline'];del state
        else:baseline=evaluate()
        def save():
            atomic_checkpoint(output/'latest.pt',{'config':model.cfg.to_dict(),'state':model.state_dict(),'optimizer':optimizer.state_dict(),
                'cursor':cursor,'updates':updates,'tokens':tokens,'identity':identity,'baseline':baseline,'torch_rng':torch.get_rng_state(),'order_seed':42})
        save();status='running'
        try:
            while cursor<len(order):
                if (output/'stop.request').exists():(output/'stop.request').unlink();status='stopped_requested';break
                if not memory_ok() or shutil.disk_usage(output).free<2*1024**3:status='paused_resources';break
                # Each three-record microbatch contains exactly one example per language.
                ids=order[cursor:cursor+3];model.train();optimizer.zero_grad(set_to_none=True)
                batches=[collate([encoded['train'][i]],tok.pad_id) for i in ids]
                count=sum(int((labels[:,1:]!=-100).sum()) for _,labels,_ in batches)
                for x,labels,_ in batches:
                    n=int((labels[:,1:]!=-100).sum());loss=model.loss(x,labels)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('SFT loss')
                    (loss*n/count).backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not bool(torch.isfinite(norm)):raise FloatingPointError('SFT gradients')
                optimizer.step();cursor+=len(ids);updates+=1;tokens+=count
                if updates%10==0:save();print(f'Chat update {updates} | assistant targets {tokens}',flush=True)
                if updates%50==0:print(json.dumps({'update':updates,'validation':evaluate()}),flush=True)
        except KeyboardInterrupt:status='interrupted'
        validation=evaluate();save()
        if cursor==len(order):status='sft_pass_complete_unvalidated'
        atomic_checkpoint(output/'base.pt',{'config':model.cfg.to_dict(),'state':model.state_dict(),'router_names':[]})
        shutil.copy2(source/'tokenizer.json',output/'tokenizer.json')
        report={'state':status,'updates':updates,'assistant_targets':tokens,'baseline':baseline,'validation':validation,
                'identity':identity,'deployed':False,'quality_gate':'Independent answer and router tests still required','data_readiness':readiness['state']}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');return report
    finally:lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','train'])
    p.add_argument('--bundle',type=Path,default=ROOT/'Datasets/workbench-conversation-v2')
    p.add_argument('--source',type=Path,default=ROOT/'runs/pretraining-20m-v1/base-training')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    report=prepare(a.bundle,a.output) if a.command=='prepare' else train(a.bundle,a.source,a.output,a.resume)
    print(json.dumps({'state':report['state'],'shortages':len(report.get('shortages',[]))},indent=2))

if __name__=='__main__':main()
