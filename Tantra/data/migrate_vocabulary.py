"""Expand a preserved-ID tokenizer's embeddings into an isolated model copy."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import torch
from Tantra.config import ROOT,ModelConfig
from Tantra.tokenizer import TantraTokenizer
from .public_sources import sha256


def expand_embedding(weight, old, new):
    old_vocab=old.tok.get_vocab(); new_vocab=new.tok.get_vocab()
    if len(old_vocab)!=weight.shape[0] or len(new_vocab)<=len(old_vocab):
        raise ValueError('Embedding/vocabulary sizes do not describe expansion')
    if any(new_vocab.get(token)!=index for token,index in old_vocab.items()):
        raise ValueError('Existing token IDs must be preserved')
    pairs={}
    for left,right in json.loads(new.tok.to_str())['model']['merges']:
        pairs.setdefault(left+right,(left,right))
    memo={token:(index,) for token,index in old_vocab.items()}
    def pieces(token):
        if token not in memo:
            if token not in pairs: raise ValueError('New token lacks an old-vocabulary decomposition')
            left,right=pairs[token]
            memo[token]=pieces(left)+pieces(right)
        return memo[token]
    result=weight.new_empty((len(new_vocab),weight.shape[1]));result[:len(old_vocab)].copy_(weight)
    with torch.no_grad():
        for token,index in sorted(new_vocab.items(),key=lambda item:item[1]):
            if index>=len(old_vocab):
                result[index].copy_(weight[list(pieces(token))].mean(dim=0))
    if not torch.equal(result[:len(old_vocab)],weight): raise AssertionError('Old embeddings changed')
    return result


def migrate(model_dir,tokenizer,output):
    model_dir,tokenizer,output=Path(model_dir).resolve(),Path(tokenizer).resolve(),Path(output).resolve()
    if output.exists() or output.is_relative_to(model_dir) or model_dir.is_relative_to(output):
        raise ValueError('Use a new output directory separate from source model')
    torch.set_num_threads(4)
    original_hash=sha256(model_dir/'base.pt'); old=TantraTokenizer.load(model_dir/'tokenizer.json')
    new=TantraTokenizer.load(tokenizer)
    source=torch.load(model_dir/'base.pt',map_location='cpu',weights_only=False)
    cfg=ModelConfig.from_dict(source['config'])
    if cfg.vocab_size!=old.vocab_size: raise ValueError('Source config/tokenizer mismatch')
    state={key:value for key,value in source['state'].items() if not key.startswith('router.') and 'lora_' not in key}
    state['embed.weight']=expand_embedding(state['embed.weight'],old,new)
    cfg.vocab_size=new.vocab_size
    if not all(bool(torch.isfinite(value).all()) for value in state.values()): raise ValueError('Non-finite expanded weights')
    output.mkdir(parents=True)
    payload={'config':cfg.to_dict(),'state':state,'router_names':[],
             'migration':{'source_base_sha256':original_hash,'source_tokenizer_sha256':sha256(model_dir/'tokenizer.json'),
                          'tokenizer_sha256':sha256(tokenizer),'old_vocab':old.vocab_size,
                          'initialization':'Mean of old token embeddings obtained from BPE merge decomposition'}}
    temporary=output/'base.tmp'
    with temporary.open('wb') as f: torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(temporary,output/'base.pt');shutil.copy2(tokenizer,output/'tokenizer.json')
    report={'old_vocab':old.vocab_size,'new_vocab':new.vocab_size,'base_parameters':sum(v.numel() for v in state.values()),
            'source_base_unchanged':sha256(model_dir/'base.pt')==original_hash,'old_embedding_rows_preserved':True,
            'tokenizer_sha256':sha256(output/'tokenizer.json'),'base_sha256':sha256(output/'base.pt'),
            'state':'initialized_requires_training','experts_copied':False,
            'note':'Old token logits are preserved mathematically before training; additional logits change softmax probabilities. No claim of preserved generated responses.'}
    (output/'migration.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',type=Path,default=ROOT/'Model')
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(migrate(a.model_dir,a.tokenizer,a.output),indent=2))

if __name__=='__main__':main()
