"""Build larger tokenizer candidates without replacing a trained model's tokenizer."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time
import unicodedata
from Tantra.config import ROOT
from Tantra.tokenizer import TantraTokenizer,SPECIAL_TOKENS
from .public_sources import sources,location,sha256
from .conversation_sources import accepted_node
from .prepare_corpus import split_for


def build_texts(output,chars_per_language=2_000_000):
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

def main():
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

if __name__=='__main__': main()
