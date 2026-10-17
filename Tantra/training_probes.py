"""Question/answer probes for monitoring, not automatic release certification."""
from collections import defaultdict
import json,math,time
from pathlib import Path
import torch
import sys


def console_print(text):
    encoding=getattr(sys.stdout,'encoding',None) or 'utf-8'
    print(text.encode(encoding,errors='backslashreplace').decode(encoding),flush=True)
from Tantra.data.public_sources import sha256


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


def main():
    import argparse
    from Tantra.config import ModelConfig
    from Tantra.model.tantra_model import TantraModel
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

if __name__=='__main__':main()
