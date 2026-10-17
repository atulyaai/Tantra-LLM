"""Stream pinned articles into immutable disk-backed Hindi/English token splits."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
import numpy as np
from Tantra.config import ROOT
from Tantra.tokenizer import TantraTokenizer
from .public_sources import sources, location, sha256


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
    scope=json.loads((ROOT/'Tantra/config/training_scope.json').read_text(encoding='utf-8'))
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
    manifest = {'excluded_corpora':exclusions,'language_token_shares':shares,'training_scope_sha256':sha256(ROOT/'Tantra/config/training_scope.json'),'version':1,'purpose':'Base language-model pilot, not expert answer supervision',
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-tokens',type=int,default=1_000_000)
    p.add_argument('--model-dir',type=Path,default=ROOT/'Model')
    p.add_argument('--exclude-corpus',type=Path,action='append',default=[])
    a=p.parse_args(); print(json.dumps(prepare(a.output,a.model_dir,a.train_tokens,a.exclude_corpus),indent=2))

if __name__=='__main__': main()
