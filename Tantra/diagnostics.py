"""Training diagnostics.

1. Microbatch benchmark: identical 4096-target backward passes at increasing
   microbatch sizes; preserves weights. Produces runs/<output>/report.json.
2. Recorded question/answer probes: held-out monitoring during base training,
   not automatic release certification.

  Microbatch benchmark:  python -m Tantra.diagnostics --stage STAGE --output OUT
  Question probes:       python -m Tantra.diagnostics --checkpoint CK --tokenizer TK \\
                                        --suite SUITE --output DIR
"""
import argparse, json, math, time
from pathlib import Path
import sys
import numpy as np
import psutil, torch
from Tantra import ModelConfig, sha256
from Tantra.train import TantraModel
from Tantra.pretrain import tensor_batch, window_order, memory_ok

# ─────────────────────────  microbatch benchmark  ─────────────────────────

def choose(results):
    safe=[r for r in results if r.get('safe') and r.get('tokens_per_second',0)>0]
    if not safe:raise MemoryError('No safe candidate')
    return max(safe,key=lambda r:r['tokens_per_second'])['microbatch']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new benchmark report')
    lock=a.stage/'base-training/training.lock'
    if lock.exists() and psutil.pid_exists(int(lock.read_text())):p.error('Pause live training before benchmark')
    torch.set_num_threads(4)
    source=a.stage/'base-training/latest.pt';before=sha256(source)
    ck=torch.load(source,map_location='cpu',weights_only=False)
    model=TantraModel(ModelConfig.from_dict(ck['config']));model.load_state_dict(ck['state']);del ck
    files={lang:np.memmap(a.stage/f'corpus-19m/{lang}-train.bin',mode='r',dtype='<u4') for lang in ('hi','en')}
    positions=window_order(files)[:16];results=[];model.train()
    for batch in (1,2,4,8,16):
        if not memory_ok():break
        rss=0
        try:
            elapsed=[]
            for repeat in range(3):
                started=time.perf_counter();model.zero_grad(set_to_none=True)
                for start in range(0,16,batch):
                    if not memory_ok():raise MemoryError('Memory gate before microbatch')
                    x=tensor_batch(files,positions[start:start+batch],256)
                    loss=model.loss(x,x)*(len(x)/16)
                    if not torch.isfinite(loss):raise FloatingPointError('Non-finite benchmark loss')
                    loss.backward();rss=max(rss,psutil.Process().memory_info().rss)
                elapsed.append(time.perf_counter()-started)
            result={'microbatch':batch,'tokens_per_second':8192/sum(elapsed[1:]),'sampled_rss_bytes':rss,'safe':memory_ok(),
                    'effective_targets':4096,'timed_repeats':2,'warmup_repeats':1}
        except (MemoryError,RuntimeError) as exc:
            if isinstance(exc,RuntimeError) and 'memory' not in str(exc).lower():raise
            result={'microbatch':batch,'safe':False,'error':str(exc)}
        finally:model.zero_grad(set_to_none=True)
        results.append(result);print(json.dumps(result),flush=True)
        if not result['safe']:break
    report={'results':results,'selected_microbatch':choose(results),'checkpoint_sha256':before,
        'checkpoint_unchanged':sha256(source)==before,'threads':4,
        'scope':'Steady-state accumulated backward throughput, excluding optimizer/evaluation; no speedup promise',
        'memory_limit_gib':8,'minimum_free_system_gib':2}
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

# ─────────────────────────  recorded question probes  ─────────────────────────

def console_print(text):
    import sys
    encoding=getattr(sys.stdout,'encoding',None) or 'utf-8'
    print(text.encode(encoding,errors='backslashreplace').decode(encoding),flush=True)


def run_probes(model,tokenizer,suite,step,generate=True,output=None):
    suite=Path(suite);items=json.loads(suite.read_text(encoding='utf-8'))['cases']
    if not items:raise ValueError('Empty probe suite')
    was_training=model.training;started=time.perf_counter();rows=[]
    torch_rng=torch.get_rng_state()
    try:
        model.eval()
        with torch.no_grad():
            for item in items:
                messages=[{'role':'user','content':item['question']},{'role':'assistant','content':item['reference']}]
                ids,mask=tokenizer.encode_chat(messages)
                if len(ids)>model.cfg.max_seq_len:raise ValueError('Probe exceeds model context')
                labels=[value if keep else -100 for value,keep in zip(ids,mask)]
                device=next(model.parameters()).device
                loss=float(model.loss(torch.tensor([ids],device=device),torch.tensor([labels],device=device)))
                if not math.isfinite(loss):raise FloatingPointError('Non-finite probe loss')
                row={**item,'answer_loss':loss,'answer_targets':sum(keep for keep in mask[1:])}
                if generate:
                    prompt,_=tokenizer.encode_chat(messages[:1],add_generation_prompt=True)
                    tokens=list(model.generate(torch.tensor([prompt],device=device),max_new_tokens=min(32,model.cfg.max_seq_len-len(prompt)),temperature=0,
                        stop_ids=(tokenizer.end_id,tokenizer.user_id)))
                    answer=tokenizer.decode(tokens);row['answer']=answer
                    row['exact_reference_match']=answer.strip().casefold()==item['reference'].strip().casefold()
                    row['correctness_review']='pending; exact reference mismatch can still be a valid paraphrase'
                rows.append(row)
    finally:
        model.train(was_training);torch.set_rng_state(torch_rng)
    per_language={}
    for language in sorted({row['language'] for row in rows}):
        group=[row for row in rows if row['language']==language];targets=sum(r['answer_targets'] for r in group)
        per_language[language]={'answer_loss':sum(r['answer_loss']*r['answer_targets'] for r in group)/targets,
            'questions':len(group),'exact_matches':sum(r.get('exact_reference_match',False) for r in group) if generate else None}
    result={'update':step,'suite_sha256':sha256(suite),'seconds':time.perf_counter()-started,'languages':per_language,'responses':rows,
            'warning':'Development probes, not final test; no routing/tools tested; reference matching is not semantic accuracy.'}
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        with (output/'probe_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
    console_print(f'QUESTION CHECK @ {step}: '+json.dumps(per_language))
    if generate:
        for row in rows:console_print(f"[{row['language']}] Q: {row['question']}\nA: {row['answer']}")
    return result


def main_probes():
    from Tantra.tokenizer import TantraTokenizer
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--suite',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    # Read-only local training checkpoint; no optimizer update or source rewrite.
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=TantraModel(ModelConfig.from_dict(ck['config']));model.load_state_dict(ck['state'])
    tok=TantraTokenizer.load(a.tokenizer)
    if tok.vocab_size!=model.cfg.vocab_size:raise ValueError('Tokenizer vocabulary mismatch')
    run_probes(model,tok,a.suite,ck.get('updates',0),True,a.output)


if __name__=='__main__':
    import sys
    main_probes() if '--checkpoint' in sys.argv else main()
