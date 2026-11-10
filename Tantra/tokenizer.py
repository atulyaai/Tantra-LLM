"""
Tantra Tokenizer
================
Small byte-level BPE (8k-32k vocab). Byte-level means no <unk>: any Hindi,
English or emoji text is encodable. A small vocab keeps the embedding table
(often the biggest part of a small model) cheap on CPU.
"""

from __future__ import annotations



from pathlib import Path
from typing import Dict, Iterable, List

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

PAD, BOS, END = "<|pad|>", "<|bos|>", "<|end|>"
SYSTEM, USER, ASSISTANT = "<|system|>", "<|user|>", "<|assistant|>"
SPECIAL_TOKENS = [PAD, BOS, END, SYSTEM, USER, ASSISTANT]


class TantraTokenizer:
    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.pad_id = tok.token_to_id(PAD)
        self.bos_id = tok.token_to_id(BOS)
        self.end_id = tok.token_to_id(END)
        self.user_id = tok.token_to_id(USER)
        self.assistant_id = tok.token_to_id(ASSISTANT)
        self.system_id = tok.token_to_id(SYSTEM)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 8192, min_frequency: int = 2) -> "TantraTokenizer":
        tok = Tokenizer(models.BPE())
        tok.normalizer = normalizers.NFC()
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok)

    @classmethod
    def load(cls, path: str | Path) -> "TantraTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(path))

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    # ── chat format ──
    #   <|bos|><|system|>..<|end|><|user|>..<|end|><|assistant|>..<|end|>
    def encode_chat(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False):
        """Returns (ids, labels_mask) where mask=1 on assistant tokens (incl. their <|end|>)."""
        role_id = {"system": self.system_id, "user": self.user_id, "assistant": self.assistant_id}
        ids, mask = [self.bos_id], [0]
        for m in messages:
            body = self.encode(m["content"].strip()) + [self.end_id]
            is_asst = m["role"] == "assistant"
            ids += [role_id[m["role"]]] + body
            mask += [0] + [1 if is_asst else 0] * len(body)
        if add_generation_prompt:
            ids.append(self.assistant_id)
            mask.append(0)
        return ids, mask


# ──  data/tokenizer_candidates.py  ──
"""Build larger tokenizer candidates without replacing a trained model's tokenizer."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time
import unicodedata
from Tantra import ROOT
from Tantra import sha256


def build_texts(output,chars_per_language=2_000_000):
    # Lazy cross-host imports keep the flattened package import graph acyclic.
    from Tantra.pretrain import sources, location, split_for
    from Tantra.conversation_train import accepted_node
    import pyarrow.parquet as pq
    counts=Counter(); docs=[]; seen=set()
    records={s['id']:s for s in sources()}
    def add(text,language,family,provenance):
        normalized=unicodedata.normalize('NFC',text.strip())
        h=hashlib.sha256(normalized.encode()).hexdigest()
        if h in seen or len(normalized)<80 or counts[language]>=chars_per_language: return
        seen.add(h); counts[language]+=len(normalized)
        docs.append({'text':normalized,'language':language,'family':family,'split':split_for(family),'source':provenance})
    hi=records['wikipedia-hi']; path=location(hi)
    if sha256(path)!=hi['sha256']: raise ValueError('Hindi source checksum mismatch')
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64,columns=['id','text','url']):
        for row in batch.to_pylist():
            add(row['text'][:12000],'hi','wikipedia:hi:'+row['id'],{'id':hi['id'],'revision':hi['revision'],'url':row['url']})
        if counts['hi']>=chars_per_language: break
    en=records['oasst1'];path=location(en)
    if sha256(path)!=en['sha256']: raise ValueError('English source checksum mismatch')
    with gzip.open(path,'rt',encoding='utf-8') as reader:
        for line in reader:
            tree=json.loads(line)
            pending=[tree['prompt']]
            while pending:
                node=pending.pop(); pending.extend(node.get('replies',[]))
                if node.get('lang')!='en' or not accepted_node(node): continue
                add(node['text'][:12000],'en','oasst1:'+tree['message_tree_id'],
                    {'id':en['id'],'revision':en['revision'],'record_id':node['message_id']})
            if counts['en']>=chars_per_language: break
    # Small original draft holdout only; never fit on these or claim fluent review.
    seed=json.loads((ROOT/'docs/data/conversation_seed.v2.json').read_text(encoding='utf-8'))
    for row in seed['records']:
        if row['language']=='hinglish':
            text='\n'.join(m['content'] for m in row['messages'])
            docs.append({'text':text,'language':'hinglish','family':'original-draft:'+row['family_id'],
                         'split':'val','source':{'id':row['source']['id'],'review':'pending fluent review'}})
    (output/'texts.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs),encoding='utf-8')
    return docs


def benchmark(output,chars_per_language=2_000_000):
    import psutil
    output=Path(output)
    if output.exists(): raise ValueError('Use a new tokenizer experiment directory')
    if psutil.virtual_memory().available<2*1024**3: raise MemoryError('At least 2 GiB free RAM required')
    output.mkdir(parents=True)
    docs=build_texts(output,chars_per_language)
    training={lang:[d['text'] for d in docs if d['split']=='train' and d['language']==lang] for lang in ('hi','en')}
    quota=min(sum(map(len,texts)) for texts in training.values())
    train=[]; training_counts={}
    for lang,texts in training.items():
        remaining=quota
        for text in texts:
            if remaining<=0: break
            part=text[:remaining]; train.append(part); remaining-=len(part)
        training_counts[lang]=quota-remaining
    held={lang:[d['text'] for d in docs if d['language']==lang and d['split']=='val'] for lang in ('hi','en','hinglish')}
    if not train or any(not rows for rows in held.values()): raise ValueError('Missing training or held-out category')
    original=ROOT/'Model/tokenizer.json'; before=sha256(original)
    def measure(tokenizer):
        return {lang:{'tokens':sum(len(tokenizer.encode(s)) for s in texts),'characters':sum(len(s) for s in texts),
                       'documents':len(texts),'roundtrip_ok':all(tokenizer.decode(tokenizer.encode(s))==unicodedata.normalize('NFC',s) for s in texts)}
                for lang,texts in held.items()}
    baseline=TantraTokenizer.load(original)
    results=[{'requested_vocab':baseline.vocab_size,'actual_vocab':baseline.vocab_size,'path':str(original),
              'sha256':before,'held_out':measure(baseline),'trained':False}]
    for requested in (16384,32768,65536):
        if psutil.virtual_memory().available<2*1024**3: raise MemoryError('Tokenizer memory gate failed')
        started=time.perf_counter(); tokenizer=TantraTokenizer.train(iter(train),vocab_size=requested)
        for i,special in enumerate(SPECIAL_TOKENS):
            if tokenizer.tok.token_to_id(special)!=i: raise ValueError('Special-token IDs changed')
        path=output/f'tokenizer-{requested}.json';tokenizer.save(path)
        result={'requested_vocab':requested,'actual_vocab':tokenizer.vocab_size,'path':str(path),
                'sha256':sha256(path),'held_out':measure(tokenizer),'trained':True,
                'seconds':time.perf_counter()-started,'embedding_parameters_width512':tokenizer.vocab_size*512}
        results.append(result);print(json.dumps(result),flush=True)
    best={lang:min(r['held_out'][lang]['tokens'] for r in results) for lang in held}
    eligible=[r for r in results if all(r['held_out'][lang]['roundtrip_ok'] and r['held_out'][lang]['tokens']<=best[lang]*1.05 for lang in held)]
    selected=min(eligible,key=lambda r:r['actual_vocab']) if eligible else None
    if sha256(original)!=before: raise ValueError('Source tokenizer changed during experiment')
    report={'results':results,'provisional_selected':selected['path'] if selected else None,
            'selection_rule':'Smallest actual vocabulary within 5% of best token count in every measured language category',
            'training_character_counts':training_counts,'source_tokenizer_unchanged':True,'training_documents':sum(d['split']=='train' for d in docs),
            'text_manifest_sha256':sha256(output/'texts.jsonl'),
            'limitations':['Hinglish holdout is a tiny unreviewed original draft sample; selection is provisional',
                          'CPU training throughput and quality are not inferred from tokenizer compression',
                          'New token IDs are incompatible with existing base/adapters despite stable special IDs',
                          'Existing base and background pilot keep their frozen tokenizer; migration/retraining is a separate stage']}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report



def extend_vocabulary(base, learned, target):
    """Append learned BPE merges, preserving every existing token ID/rank."""
    import copy
    from tokenizers import Tokenizer
    config=json.loads(base.tok.to_str()); candidate=json.loads(learned.tok.to_str())
    if config['model']['type']!='BPE' or candidate['model']['type']!='BPE': raise ValueError('BPE required')
    for field in ('normalizer','pre_tokenizer','decoder'):
        if config[field]!=candidate[field]: raise ValueError('Tokenizer text processing differs')
    vocab=config['model']['vocab']; original=dict(vocab)
    if target<=len(vocab): raise ValueError('Expansion target must exceed current vocabulary')
    if sorted(vocab.values())!=list(range(len(vocab))): raise ValueError('Non-contiguous original IDs')
    merges=config['model']['merges']; known={tuple(pair) for pair in merges}
    for pair in candidate['model']['merges']:
        left,right=pair; combined=left+right
        if (left,right) in known or left not in vocab or right not in vocab: continue
        if combined not in vocab:
            if len(vocab)>=target: continue
            vocab[combined]=len(vocab)
        merges.append([left,right]);known.add((left,right))
    result=TantraTokenizer(Tokenizer.from_str(json.dumps(config,ensure_ascii=False)))
    if any(result.tok.token_to_id(token)!=identity for token,identity in original.items()):
        raise ValueError('Existing token ID changed')
    return result


def extend_experiment(output, learned_path, texts_path, target=65536):
    output=Path(output)
    if output.exists(): raise ValueError('Use a new extension directory')
    output.mkdir(parents=True)
    base_path=ROOT/'Model/tokenizer.json'; original_hash=sha256(base_path)
    base=TantraTokenizer.load(base_path); learned=TantraTokenizer.load(learned_path)
    extended=extend_vocabulary(base,learned,target)
    path=output/'tokenizer.json';extended.save(path)
    counts={lang:{'base_tokens':0,'extended_tokens':0,'documents':0,'roundtrip_ok':True} for lang in ('hi','en','hinglish')}
    with Path(texts_path).open(encoding='utf-8') as reader:
        for line in reader:
            row=json.loads(line)
            if row['split']!='val':continue
            item=counts[row['language']];text=row['text']
            item['base_tokens']+=len(base.encode(text));ids=extended.encode(text)
            item['extended_tokens']+=len(ids);item['documents']+=1
            item['roundtrip_ok'] &= extended.decode(ids)==unicodedata.normalize('NFC',text)
    report={'actual_vocab':extended.vocab_size,'requested_vocab':target,'path':str(path),'sha256':sha256(path),
            'base_tokenizer_sha256':original_hash,'learned_tokenizer_sha256':sha256(learned_path),
            'text_manifest_sha256':sha256(texts_path),'existing_ids_preserved':True,'held_out':counts,
            'source_tokenizer_unchanged':sha256(base_path)==original_hash,
            'deployment':'Candidate only; embedding/output expansion and training required before model use',
            'limitations':['Hinglish evaluation is only eight unreviewed original draft texts',
                          'Compression does not establish model quality or CPU throughput']}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report

def candidates_cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chars-per-language',type=int,default=2_000_000)
    p.add_argument('--extend-from',type=Path);p.add_argument('--texts',type=Path)
    p.add_argument('--target-vocab',type=int,default=65536);a=p.parse_args()
    if a.extend_from:
        if not a.texts:p.error('--texts is required for extension evaluation')
        print(json.dumps(extend_experiment(a.output,a.extend_from,a.texts,a.target_vocab),indent=2));return
    if a.chars_per_language<10000:p.error('Need at least 10000 chars per language')
    r=benchmark(a.output,a.chars_per_language)
    print(json.dumps({'provisional_selected':r['provisional_selected'],'source_tokenizer_unchanged':r['source_tokenizer_unchanged']}))



# ──  data/migrate_vocabulary.py  ──
"""Expand a preserved-ID tokenizer's embeddings into an isolated model copy."""
import argparse
import json
import os
from pathlib import Path
import shutil
import torch
from Tantra import ROOT, ModelConfig
from Tantra import sha256


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


def migrate_cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',type=Path,default=ROOT/'Model')
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(migrate(a.model_dir,a.tokenizer,a.output),indent=2))



# ──  data/vocabulary_training_job.py  ──
"""Queue the isolated 64K pilot after the earlier CPU job and its quality gates."""
import argparse
import json
from pathlib import Path
import time
from Tantra import ROOT


def run(output,model_dir,previous=None):
    # Lazy cross-host import keeps the flattened package import graph acyclic.
    from Tantra.pretrain import write_status
    from Tantra.pretrain import prepare
    from Tantra.pretrain import pilot_run as train
    output,model_dir=Path(output),Path(model_dir)
    previous=Path(previous) if previous is not None else None
    output.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+8*3600
    while previous is not None and not (previous/'report.json').exists():
        write_status(output,'queued_after_existing_cpu_pilot',previous=str(previous))
        if (previous/'failure.json').exists():
            write_status(output,'blocked_previous_pilot_failed');return
        if time.monotonic()>deadline:
            write_status(output,'blocked_previous_pilot_timeout');return
        time.sleep(30)
    if previous is not None:
        prior=json.loads((previous/'report.json').read_text(encoding='utf-8'))
        if prior['state']!='pilot_complete' or any(prior['latest_validation'][lang]['loss']>=prior['baseline_validation'][lang]['loss'] for lang in ('hi','en')):
            write_status(output,'blocked_previous_language_quality_gate',previous_state=prior['state']);return
    try:
        corpus=output/'corpus-1m'
        write_status(output,'preparing_balanced_64k_corpus')
        if not (corpus/'manifest.json').exists(): prepare(corpus,model_dir,1_000_000)
        write_status(output,'training_64k_pilot',model_dir=str(model_dir),base_token_shares={'hi':.5,'en':.5})
        base=output/'base-pilot'
        result=train(corpus,base,model_dir,resume=(base/'latest.pt').exists(),threads=4,checkpoint_updates=10)
        write_status(output,result['state'],updates=result['updates'],trained_target_tokens=result['trained_target_tokens'],
                     validation=result['latest_validation'],deployed=False)
    except Exception as exc:
        write_status(output,'failed',error=type(exc).__name__+': '+str(exc));raise


def vjob_cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--previous',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import os
    lock=a.output/'worker.lock'
    with lock.open('x') as handle:handle.write(str(os.getpid()))
    try:run(a.output,a.model_dir,a.previous)
    finally:lock.unlink()
