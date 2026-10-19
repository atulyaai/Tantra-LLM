"""Train one prepared adapter without loading or rewriting all other experts."""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

from .expert_workbench import catalog, digest, validate_record, verify_auto, write_json


def load_split(root, item, expert):
    import hashlib
    # Manifests written on Windows store backslash paths; accept them on every OS.
    path=Path(root)/'prepared'/Path(*item['path'].replace('\\','/').split('/'))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        raise ValueError('Prepared split checksum mismatch: '+str(path))
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(validate_record(json.loads(line),expert))
    return rows


def train_one(out, expert, model_dir, output, steps=100, rank=8, threads=4, release_check=False, small_pilot=False):
    import hashlib
    import torch
    from Tantra.config import ModelConfig
    from Tantra.data.expert_dataset import EncodedDataset
    from Tantra.model.tantra_model import TantraModel
    from Tantra.tokenizer import TantraTokenizer
    out,model_dir,output=Path(out).resolve(),Path(model_dir).resolve(),Path(output).resolve()
    if output==model_dir or model_dir.is_relative_to(output) or output.is_relative_to(model_dir):
        raise ValueError('Output must be separate from the source Model directory')
    if steps<1 or rank<1 or threads<1: raise ValueError('steps, rank and threads must be positive')
    manifest_path=out/'prepared'/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if expert not in manifest['experts']: raise ValueError('Unknown expert: '+expert)
    record=manifest['experts'][expert]
    undersized=record['state']!='data_ready'
    if undersized and not small_pilot: raise ValueError('Expert data is not ready: '+expert)
    if undersized and release_check: raise ValueError('An undersized pilot cannot request release validation')
    if (output/f'{expert}.pt').exists(): raise ValueError('Candidate already exists; use a new output directory')
    rows={s:load_split(out,record['files'][s],expert) for s in ('train','val','test')}
    if any(not values for values in rows.values()): raise ValueError('Every pilot split must contain reviewed examples')
    groups={s:{row['split_group'] for row in values} for s,values in rows.items()}
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Cross-split family leakage')
    torch.set_num_threads(threads); torch.manual_seed(42)
    tok=TantraTokenizer.load(model_dir/'tokenizer.json')
    checkpoint=torch.load(model_dir/'base.pt',map_location='cpu',weights_only=False)
    cfg=ModelConfig.from_dict(checkpoint['config'])
    if cfg.vocab_size!=tok.vocab_size: raise ValueError('Base/tokenizer vocabulary mismatch')
    model=TantraModel(cfg)
    base={k:v for k,v in checkpoint['state'].items() if not k.startswith('router.') and 'lora_' not in k}
    missing,unexpected=model.load_state_dict(base,strict=False)
    if unexpected or any(not k.startswith('router.') for k in missing):
        raise ValueError('Incompatible base state')
    del checkpoint,base
    for p in model.parameters(): p.requires_grad=False
    model.add_expert(expert,rank)
    parameters=model.expert_parameters(expert)
    model.set_active([(expert,1.)])
    datasets={s:EncodedDataset(tok,values,max_len=min(256,cfg.max_seq_len)) for s,values in rows.items()}
    if any(not len(dataset) for dataset in datasets.values()): raise ValueError('No assistant labels after truncation')
    def loss_on(dataset):
        model.eval(); total=0.; count=0
        with torch.no_grad():
            for ids,labels,_ in dataset.batches(4,shuffle=False):
                n=int((labels[:,1:]!=-100).sum())
                value=float(model.loss(ids,labels))
                if not math.isfinite(value): raise FloatingPointError('Non-finite validation loss')
                total+=value*n; count+=n
        return total/count
    model.set_active([]); before=loss_on(datasets['val']); model.set_active([(expert,1.)])
    optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=.01)
    rng=random.Random(42); step=0; started=time.perf_counter(); peak=0
    import psutil
    while step<steps:
        for ids,labels,_ in datasets['train'].batches(4,rng=rng):
            memory=psutil.virtual_memory()
            rss=psutil.Process().memory_info().rss; peak=max(peak,rss)
            if rss>8*2**30 or memory.available<2*2**30:
                raise MemoryError('Resource limit reached; source weights remain unchanged')
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss=model.loss(ids,labels)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite training loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not bool(torch.isfinite(norm)): raise FloatingPointError('Non-finite gradient')
            optimizer.step(); step+=1
            if step>=steps: break
    after=loss_on(datasets['val'])
    # No test-set answers are generated until the optional release check.
    answer_accuracy={}
    if release_check and after<before:
        for split in ('val','test'):
            checked=correct=0
            for row in rows[split][:100]:
                ids,_=tok.encode_chat(row['messages'][:-1],add_generation_prompt=True)
                if len(ids)+64>cfg.max_seq_len: continue
                tokens=list(model.generate(torch.tensor([ids]),max_new_tokens=64,temperature=0,stop_ids=(tok.end_id,tok.user_id)))
                candidate={**row,'messages':row['messages'][:-1]+[{'role':'assistant','content':tok.decode(tokens)}]}
                if row.get('verification'):
                    checked+=1
                    try: correct+=bool(verify_auto(candidate))
                    except (ValueError,TypeError,KeyError): pass
            answer_accuracy[split]={'checked':checked,'correct':correct,'accuracy':correct/checked if checked else None}
    validated=(release_check and after<before and all(answer_accuracy.get(s,{}).get('checked',0)>=100 and
                (answer_accuracy[s]['accuracy'] or 0)>=.9 for s in ('val','test')))
    output.mkdir(parents=True,exist_ok=True)
    payload={'rank':rank,'state':model.expert_state(expert),'base_sha256':hashlib.sha256((model_dir/'base.pt').read_bytes()).hexdigest(),
             'tokenizer_sha256':hashlib.sha256((model_dir/'tokenizer.json').read_bytes()).hexdigest(),
             'dataset_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    temp=output/f'{expert}.tmp'; torch.save(payload,temp); os.replace(temp,output/f'{expert}.pt')
    report={'expert':expert,'state':'quality_validated' if validated else ('undersized_pilot_unvalidated' if undersized else 'pilot_unvalidated'),'steps':step,'rank':rank,
            'adapter_parameters':sum(p.numel() for p in parameters),'validation_loss_base':before,'validation_loss_adapter':after,
            'answer_accuracy':answer_accuracy,'seconds':time.perf_counter()-started,'peak_rss_bytes':peak,
            'source_model_changed':False,'router_retrained':False,
            'data_thresholds_met':not undersized,'split_examples':{s:len(v) for s,v in rows.items()},
            'split_families':{s:len(v) for s,v in groups.items()},
            'training_languages':sorted({row['language'] for row in rows['train']}),
            'note':'Candidate only. Keep paired with exact base/tokenizer hashes; router integration and regression checks are still required.'}
    write_json(output/f'{expert}.report.json',report)
    return report