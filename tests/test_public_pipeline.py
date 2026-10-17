import json
from pathlib import Path
import numpy as np
import pytest
import torch
from Tantra.config import ModelConfig
from Tantra.model.tantra_model import TantraModel
from Tantra.data import conversation_sources as conversation
from Tantra.data import train_base_pilot as training
from Tantra.data.public_sources import sha256


def node(text='Hello', role='prompter', **changes):
    return dict(message_id=role+text, text=text, role=role, review_result=True,
                synthetic=False, deleted=False, lang='en', replies=[], **changes)


def test_review_queue_does_not_treat_source_ratings_as_expert_approval():
    root=node(); root['replies']=[node('Hello, how can I help?', 'assistant')]
    rows=list(conversation.paths(root))
    assert len(rows)==1 and rows[0][1][0]['role']=='user'
    root['replies'][0]['synthetic']=True
    assert not list(conversation.paths(root))
    assert conversation.suggest('hello!')=='greeting'
    assert conversation.suggest('Translate and summarize this') is None


def test_review_export_checks_queue_integrity_and_explicit_decisions(tmp_path):
    queue=tmp_path/'queue'; queue.mkdir()
    row={'schema_version':1,'record_id':'a','expert':'greeting','language':'en','family_id':'source:tree1',
         'template_id':'source-human','messages':[{'role':'user','content':'Hello'},{'role':'assistant','content':'Hello!'}],
         'source':{'id':'test','revision':'abc','license':'Apache-2.0','kind':'test'},
         'review':{'status':'pending','reviewer':None}}
    path=queue/'greeting.jsonl'; path.write_text(json.dumps(row)+'\n')
    (queue/'index.json').write_text(json.dumps({'files':{path.name:sha256(path)}}))
    decisions=tmp_path/'decisions.jsonl'
    decisions.write_text(json.dumps({'record_id':'a','expert':'greeting','status':'approved'}))
    with pytest.raises(ValueError, match='reviewer'):
        conversation.approve(queue,decisions,tmp_path/'bad')
    decisions.write_text(json.dumps({'record_id':'a','expert':'greeting','status':'approved',
                                     'reviewer':'test fixture reviewer','rationale':'test fixture approval'}))
    assert conversation.approve(queue,decisions,tmp_path/'good')['approved']==1
    path.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        conversation.approve(queue,decisions,tmp_path/'tampered')


def test_windows_do_not_repeat_and_follow_bilingual_ratio():
    files={'hi':np.arange(6*256+1),'en':np.arange(4*256+1)}
    order=training.window_order(files)
    assert [lang for lang,_ in order[:5]]==['hi','en','hi','en','hi']
    assert len(order)==len(set(order))==10


def test_interrupted_base_pilot_resumes_exactly(tmp_path,monkeypatch):
    # Small real transformer, fixed benchmark clock to select the same microbatch.
    monkeypatch.setattr(training,'memory_ok',lambda:True)
    tick=iter(range(10000)); monkeypatch.setattr(training.time,'perf_counter',lambda:next(tick))
    torch.manual_seed(1)
    model=TantraModel(ModelConfig(vocab_size=32,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,mlp_mult=4,max_seq_len=512))
    source=tmp_path/'source'; source.mkdir()
    torch.save({'config':model.cfg.to_dict(),'state':model.state_dict()},source/'base.pt')
    (source/'tokenizer.json').write_text('test fixture')
    maps={lang+'-'+split:np.asarray([i%31 for i in range(256*count+1)],dtype=np.uint32)
          for lang,count in [('hi',21),('en',14)] for split in ('train','val')}
    train={lang:maps[lang+'-train'] for lang in ('hi','en')}
    order=training.window_order(train); identities={'fixture':'same'}
    full=tmp_path/'full'; full.mkdir(); resumed=tmp_path/'resumed'; resumed.mkdir()
    training._train(full,source,identities,maps,train,order,256,0,False,1,1)
    training._train(resumed,source,identities,maps,train,order,256,1,False,1,1)
    result=training._train(resumed,source,identities,maps,train,order,256,0,True,1,1)
    a=torch.load(full/'latest.pt',weights_only=False); b=torch.load(resumed/'latest.pt',weights_only=False)
    assert a['cursor']==b['cursor']==len(order)
    assert a['tokens']==b['tokens'] and result['state']=='pilot_complete'
    assert all(torch.equal(value,b['state'][key]) for key,value in a['state'].items())
    assert torch.equal(a['torch_rng'],b['torch_rng'])


def test_source_download_requires_pinned_checksum(tmp_path,monkeypatch):
    import hashlib
    import io
    from Tantra.data import public_sources as public
    payload=b'known pinned dataset'
    source={'id':'fixture','repo':'test/fixture','revision':'immutable','path':'file.bin',
            'size':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    class Response(io.BytesIO):
        status=206
        headers={'Content-Range':f'bytes 0-{len(payload)-1}/{len(payload)}'}
    monkeypatch.setattr(public.urllib.request,'urlopen',lambda *a,**kw:Response(payload))
    path=public.fetch(source,tmp_path)
    assert path.read_bytes()==payload
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checksum'):
        public.fetch(source,tmp_path)


def test_corpus_freezes_disjoint_documents_and_real_tokens(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from Tantra.data import prepare_corpus as corpus
    # Use actual tokenizer with many distinct article IDs in each split.
    raw=tmp_path/'source.parquet'
    rows=[{'id':str(i),'url':f'https://example.test/wiki/{i}','title':str(i),
           'text':('A distinct article about learning language and history. '*8)+str(i)} for i in range(1000)]
    pq.write_table(pa.Table.from_pylist(rows),raw)
    records=[{'id':'fixture-'+lang,'kind':'encyclopedia','language':lang,'revision':'fixture',
              'license':'test fixture only','sha256':sha256(raw)} for lang in ('hi','en')]
    monkeypatch.setattr(corpus,'sources',lambda:records)
    monkeypatch.setattr(corpus,'location',lambda source:raw)
    # Prefix language-specific text to avoid duplicates across test source copies.
    # The same text across languages is deliberately rejected: don't inflate budgets.
    manifest=corpus.prepare(tmp_path/'corpus',train_tokens=10000)
    assert manifest['files']['hi-train']['target']==manifest['files']['en-train']['target']==5000
    assert manifest['language_token_shares']=={'hi':.5,'en':.5}
    assert manifest['files']['hi-train']['tokens']>0
    assert manifest['files']['en-train']['tokens']>0
    articles=[json.loads(line) for line in (tmp_path/'corpus'/'attribution.jsonl').read_text().splitlines()]
    mapping={}
    for row in articles:
        assert row['text_sha256'] not in mapping
        mapping[row['text_sha256']]=row['split']
    assert all(v['tokens']>=v['text_tokens'] for v in manifest['files'].values())
    next_manifest=corpus.prepare(tmp_path/'next-corpus',train_tokens=10000,exclude_corpora=[tmp_path/'corpus'])
    next_rows=[json.loads(line) for line in (tmp_path/'next-corpus'/'attribution.jsonl').read_text().splitlines()]
    assert {row['family_id'] for row in articles}.isdisjoint(row['family_id'] for row in next_rows)
    assert {row['text_sha256'] for row in articles}.isdisjoint(row['text_sha256'] for row in next_rows)
    assert next_manifest['excluded_corpora']
    with pytest.raises(ValueError,match='immutable'):
        corpus.prepare(tmp_path/'corpus',train_tokens=10000)
