"""
Serving integrity, reply/router evaluation and the conversation pipeline.

Merged: test_serving_integrity.py, test_reply_eval.py, test_router_eval.py, test_conversation_training.py, test_public_pipeline.py.
"""

# ──────────────────────────────────────────────────────────────────────  test_serving_integrity.py  ──────────────────────────────────────────────────────────────────────

import json
from pathlib import Path
import pytest
import torch
from Tantra import ModelConfig
from Tantra.train import TantraModel, save_expert
from Tantra.tokenizer import TantraTokenizer
from Tantra import sha256
from Tantra.main import TantraEngine
from Tantra.main import ServingIntegrityError
from Tantra.main import ExpertMismatch


def candidate(root,hashes=True):
    tok=TantraTokenizer.train(['Hello there I am Tantra']*3,vocab_size=280,min_frequency=1);tok.save(root/'tokenizer.json')
    model=TantraModel(ModelConfig(vocab_size=tok.vocab_size,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,mlp_mult=2,max_seq_len=64,lora_rank=2))
    model.add_expert('greeting');model.save(root/'base.pt')
    base=sha256(root/'base.pt');tokenizer=sha256(root/'tokenizer.json')
    save_expert(model,'greeting',root/'experts',**({'base_sha256':base,'tokenizer_sha256':tokenizer} if hashes else {}))
    manifest={'version':2,'base_sha256':base,'tokenizer_sha256':tokenizer,
              'artifacts':{'base.pt':base,'tokenizer.json':tokenizer},
              'experts':{'greeting':{'file':'experts/greeting.pt','rank':2,'sha256':sha256(root/'experts/greeting.pt')}}}
    (root/'manifest.json').write_text(json.dumps(manifest));return manifest


def test_declared_serving_adapters_require_training_identity_even_if_shapes_fit(tmp_path):
    candidate(tmp_path,hashes=False)
    eng=TantraEngine(tmp_path)
    with pytest.raises(ExpertMismatch,match='lacks required'):eng.cache.ensure(['greeting'])
    with pytest.raises(ExpertMismatch,match='lacks required'):TantraEngine(tmp_path,lazy=False)


def test_declared_adapter_tampering_is_refused_before_tensor_load(tmp_path):
    candidate(tmp_path);eng=TantraEngine(tmp_path)
    (tmp_path/'experts/greeting.pt').write_bytes(b'invalid replaced file')
    with pytest.raises(ExpertMismatch,match='checksum'):eng.cache.ensure(['greeting'])


def test_served_base_and_tokenizer_binding_and_eager_equivalence(tmp_path):
    candidate(tmp_path)
    lazy=TantraEngine(tmp_path);eager=TantraEngine(tmp_path,lazy=False)
    lazy.cache.activate([('greeting',1.)]);eager.model.set_active([('greeting',1.)])
    x=torch.tensor([[1,6,7]])
    with torch.no_grad():assert torch.equal(lazy.model(x),eager.model(x))
    with (tmp_path/'base.pt').open('ab') as f:f.write(b'tamper')
    with pytest.raises(ServingIntegrityError,match='checksum'):TantraEngine(tmp_path)


def test_old_unbound_manifest_refused_but_legacy_without_manifest_supported(tmp_path):
    candidate(tmp_path);(tmp_path/'manifest.json').write_text('{"base_sha256":"unbound"}')
    with pytest.raises(ServingIntegrityError,match='artifact binding'):TantraEngine(tmp_path)
    (tmp_path/'manifest.json').unlink();assert TantraEngine(tmp_path).model.cfg.vocab_size>0


def test_router_artifact_cannot_be_injected_without_manifest_binding(tmp_path):
    candidate(tmp_path);(tmp_path/'router_ngram.pt').write_bytes(b'not a router')
    with pytest.raises(ServingIntegrityError,match='router artifact binding'):TantraEngine(tmp_path)

# ──────────────────────────────────────────────────────────────────────  test_reply_eval.py  ──────────────────────────────────────────────────────────────────────

from Tantra.evaluate import aggregate, load_cases, repetitive, score_reply

CREATOR = {'id': 'c', 'prompt': 'Which person built this bot?', 'any': ['atul vij', 'atul']}
NOT_ATUL = {'id': 'n', 'prompt': 'So you are Atul Vij, right?', 'any': ['no', 'not'], 'none': ['i am atul vij', "i'm atul vij"]}
RAIN = {'id': 'r', 'prompt': "It's pouring rain outside.", 'any': ['rain', 'chai']}


def test_case_scoring_rewards_cues_and_punishes_violations():
    assert score_reply(CREATOR, 'Atul Vij built me!')['score'] == 1.0
    assert score_reply(CREATOR, "I'm Tantra! What's your name?")['score'] == 0.4
    assert score_reply(NOT_ATUL, "I'm Atul Vij.")['score'] == 0.0                      # forbidden
    assert score_reply(RAIN, "Rain! I'm an AI, so I can't feel it.")['score'] == 0.8   # unasked AI mention
    assert score_reply(RAIN, '')['score'] == 0.0


def test_repetition_detection():
    assert repetitive("I'm Tantra, and I'm Tantra, and I'm Tantra")
    assert repetitive('old old old')
    assert not repetitive("Hey! How's your day going?")


def test_one_line_for_everything_scores_badly():
    cases = [CREATOR, RAIN, {'id': 'h', 'prompt': 'Hey hey, hello!', 'any': ['hi', 'hello', 'hey']}]
    varied = aggregate({'c': 'Atul Vij made me.', 'r': 'Rain means chai!', 'h': 'Hey! Good to see you.'}, cases)
    same = aggregate({k: "I'm Tantra! What's your name?" for k in 'crh'}, cases)
    assert varied['reply_score'] == 1.0 and varied['passed'] == 3
    assert same['diversity'] < 0.4 and same['reply_score'] < 0.2


def test_shipped_cases_are_well_formed():
    cases = load_cases()
    assert len(cases) >= 30 and len({c['id'] for c in cases}) == len(cases)
    assert all(c['prompt'] and (c.get('any') or c.get('none')) for c in cases)

# ──────────────────────────────────────────────────────────────────────  test_router_eval.py  ──────────────────────────────────────────────────────────────────────

from Tantra.evaluate import NONE, evaluate, labelled
from Tantra.main import train_ngram_router


def rows(expert, texts, family):
    return [{'expert': expert, 'family_id': f'{expert}:{family}:{i // 2}',
             'messages': [{'role': 'user', 'content': t}, {'role': 'assistant', 'content': 'ok'}]} for i, t in enumerate(texts)]


IDENTITY = ['who made you', 'who created you', 'who built you', 'who is your creator', 'who designed you', 'who coded you']
WEATHER = ['it is raining today', 'so hot outside', 'it is very cold', 'the weather is nice', 'heavy rain here', 'foggy morning']


def test_none_class_and_metrics():
    names = ['identity_self', NONE]
    data = labelled(rows('identity_self', IDENTITY, 'f') + rows('smalltalk', WEATHER, 'g'), names, none=['smalltalk'])
    assert {names[y] for _, y, _ in data} == {'identity_self', NONE}
    router, _ = train_ngram_router([t for t, _, _ in data], [y for _, y, _ in data], names, epochs=30, holdout=0)
    report = evaluate(router, data, names)
    assert report['rows'] == 12 and report['top1_accuracy'] >= 0.9 and report['top2_recall'] == 1.0
    assert 0.3 <= report['routed_to_base_only'] <= 0.7


def test_grouped_holdout_never_splits_a_family():
    texts = IDENTITY + WEATHER
    groups = [f'g{i // 2}' for i in range(len(texts))]
    labels = [0] * 6 + [1] * 6
    import Tantra.main as module
    seen = {}
    original = module.NgramRouter.forward
    def spy(self, batch):
        seen.setdefault('sizes', []).append(len(batch))
        return original(self, batch)
    module.NgramRouter.forward = spy
    try:
        _, acc = train_ngram_router(texts, labels, ['a', 'b'], epochs=1, holdout=0.34, groups=groups, batch_size=100)
    finally:
        module.NgramRouter.forward = original
    # 6 families, 2 held out -> 4 held-out rows and 8 training rows, never 1 row of a pair on each side
    assert seen['sizes'][0] == 8 and seen['sizes'][-1] == 4 and 0.0 <= acc <= 1.0


def test_router_uses_last_user_request_in_multiturn_conversation():
    row={'expert':'identity_self','family_id':'one-conversation','messages':[
         {'role':'user','content':'Hello'}, {'role':'assistant','content':'Hi!'},
         {'role':'user','content':'Who created you?'}, {'role':'assistant','content':'Atul Vij.'}]}
    assert labelled([row],['greeting','identity_self'],[])[0][0]=='Who created you?'


def test_macro_f1_reports_missing_class_instead_of_hiding_it():
    import torch
    class Stub:
        def __call__(self,features):return torch.tensor([[9.,0.],[9.,0.]])
    result=evaluate(Stub(),[('a',0,'f1'),('b',1,'f2')],['greeting','identity_self'])
    assert result['top1_accuracy']==.5
    assert abs(result['macro_f1']-1/3)<1e-6
    assert result['per_class']['identity_self']['recall']==0

# ──────────────────────────────────────────────────────────────────────  test_conversation_training.py  ──────────────────────────────────────────────────────────────────────

import json
from pathlib import Path
import pytest
from Tantra.conversation_train import balanced_order,inspect,train
from Tantra import ROOT


def test_balanced_order_does_not_duplicate_to_fill_language_shortages():
    rows=[{'language':'en'}]*5+[{'language':'hi'}]*3+[{'language':'hinglish'}]*2
    order=balanced_order(rows)
    assert len(order)==len(set(order))==6
    assert [rows[i]['language'] for i in order]==['hi','en','hinglish']*2
    with pytest.raises(ValueError,match='Missing'):balanced_order([{'language':'en'}])


def test_actual_seed_blocks_unreviewed_multilingual_chat_training(tmp_path):
    bundle=ROOT/'Datasets/workbench-conversation-v2'
    if not bundle.exists():pytest.skip('Local ignored bundle is not distributed')
    rows,report=inspect(bundle)
    assert report['state']=='needs_reviewed_data'
    assert all(row['language']=='en' for records in rows.values() for row in records)
    with pytest.raises(ValueError,match='gate failed'):train(bundle,tmp_path/'source',tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_real_supervised_training_masks_user_and_preserves_source(tmp_path,monkeypatch):
    import torch
    from Tantra import ModelConfig
    from Tantra.train import TantraModel
    from Tantra.tokenizer import TantraTokenizer
    from Tantra.train import encode_row
    from Tantra import sha256
    import Tantra.conversation_train as module
    tok=TantraTokenizer.train(['Hello hi welcome thanks I am Tantra.']*3,vocab_size=280,min_frequency=1)
    source=tmp_path/'source';source.mkdir();tok.save(source/'tokenizer.json')
    model=TantraModel(ModelConfig(vocab_size=tok.vocab_size,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,mlp_mult=4,max_seq_len=256))
    torch.save({'config':model.cfg.to_dict(),'state':model.state_dict()},source/'base.pt')
    (source/'report.json').write_text(json.dumps({'state':'pilot_complete','latest_validation':{'hi':{'loss':1},'en':{'loss':1}},'baseline_validation':{'hi':{'loss':2},'en':{'loss':2}}}))
    original=sha256(source/'base.pt')
    row={'messages':[{'role':'user','content':'Hello'},{'role':'assistant','content':'Hi!'}]}
    records={split:[{**row,'language':lang} for lang in ('hi','en','hinglish')] for split in ('train','val','test')}
    # Fixture-only synthetic readiness bypass tests the numerical loop, not real data approval.
    monkeypatch.setattr(module,'inspect',lambda bundle:(records,{'state':'data_ready','bundle_manifest_sha256':'test-fixture'}))
    import Tantra.pretrain as resources
    monkeypatch.setattr(resources,'memory_ok',lambda:True)
    result=train(tmp_path/'fixture',source,tmp_path/'candidate')
    assert result['state']=='sft_pass_complete_unvalidated' and result['updates']==1
    assert not result['deployed'] and sha256(source/'base.pt')==original
    ids,labels=encode_row(tok,row,256,True)
    assert labels[0]==-100 and sum(v!=-100 for v in labels)>0
    saved=torch.load(tmp_path/'candidate/latest.pt',weights_only=False)
    assert saved['cursor']==3 and 'optimizer' in saved and 'torch_rng' in saved


def test_english_pilot_is_labelled_and_never_data_ready(tmp_path,monkeypatch):
    import torch
    from Tantra import ModelConfig
    from Tantra.train import TantraModel
    from Tantra.tokenizer import TantraTokenizer
    from Tantra.conversation_train import pilot_order
    import Tantra.conversation_train as module
    import Tantra.pretrain as resources
    tok=TantraTokenizer.train(['Hello hi welcome thanks I am Tantra.']*3,vocab_size=280,min_frequency=1)
    source=tmp_path/'source';source.mkdir();tok.save(source/'tokenizer.json')
    model=TantraModel(ModelConfig(vocab_size=tok.vocab_size,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,mlp_mult=4,max_seq_len=256))
    torch.save({'config':model.cfg.to_dict(),'state':model.state_dict()},source/'base.pt')
    (source/'report.json').write_text(json.dumps({'state':'pilot_complete','latest_validation':{'hi':{'loss':1},'en':{'loss':1}},'baseline_validation':{'hi':{'loss':2},'en':{'loss':2}}}))
    row={'language':'en','messages':[{'role':'user','content':'Hello'},{'role':'assistant','content':'Hi!'}]}
    records={'train':[row]*3,'val':[row],'test':[row]}
    calls=[]
    def fake_inspect(bundle,english_pilot=False):
        calls.append(english_pilot)
        return records,{'state':'english_pilot_undersized','bundle_manifest_sha256':'test-fixture'}
    monkeypatch.setattr(module,'inspect',fake_inspect)
    monkeypatch.setattr(resources,'memory_ok',lambda:True)
    with pytest.raises(ValueError,match='epochs'):train(tmp_path/'fixture',source,tmp_path/'x',epochs=2)
    result=train(tmp_path/'fixture',source,tmp_path/'pilot',english_pilot=True,epochs=2)
    assert calls==[True]
    assert result['state']=='english_pilot_complete_unvalidated' and result['updates']==2
    assert result['languages']==['en'] and result['mode']=='english_pilot' and not result['deployed']
    assert result['pilot_warning'] and set(result['validation'])=={'en'}
    assert result['best_update']==2 and (tmp_path/'pilot/best.pt').exists()
    order=pilot_order(records['train'],2)
    assert len(order)==6 and sorted(order[:3])==[0,1,2]


def test_validation_patience_persists_and_resets_only_on_improvement():
    from Tantra.conversation_train import validation_progress
    best={'loss':4.19}
    for loss in (4.32,4.19,4.40,4.35,4.42):
        assert not validation_progress(best,loss)
    assert best['bad_evals']==5
    restored=json.loads(json.dumps(best))
    assert not validation_progress(restored,4.2)
    assert restored['bad_evals']==6
    assert validation_progress(restored,4.1)
    assert restored['bad_evals']==0
    with pytest.raises(FloatingPointError):validation_progress(restored,float('nan'))


def test_select_by_replies_keeps_reply_scored_checkpoint(tmp_path,monkeypatch):
    import torch
    from Tantra import ModelConfig
    from Tantra.train import TantraModel
    from Tantra.tokenizer import TantraTokenizer
    import Tantra.conversation_train as module
    import Tantra.pretrain as resources
    import Tantra.evaluate as reply_eval
    tok=TantraTokenizer.train(['Hello hi welcome thanks I am Tantra.']*3,vocab_size=280,min_frequency=1)
    source=tmp_path/'source';source.mkdir();tok.save(source/'tokenizer.json')
    model=TantraModel(ModelConfig(vocab_size=tok.vocab_size,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,mlp_mult=4,max_seq_len=256))
    torch.save({'config':model.cfg.to_dict(),'state':model.state_dict()},source/'base.pt')
    (source/'report.json').write_text(json.dumps({'state':'pilot_complete','latest_validation':{'hi':{'loss':1},'en':{'loss':1}},'baseline_validation':{'hi':{'loss':2},'en':{'loss':2}}}))
    row={'language':'en','messages':[{'role':'user','content':'Hello'},{'role':'assistant','content':'Hi!'}]}
    records={'train':[row]*3,'val':[row],'test':[row]}
    monkeypatch.setattr(module,'inspect',lambda bundle,english_pilot=False:(records,{'state':'english_pilot_undersized','bundle_manifest_sha256':'t'}))
    monkeypatch.setattr(resources,'memory_ok',lambda:True)
    scored=[]
    def fake_eval(model,tok,cases=None):
        scored.append(1)
        return {'reply_score':0.42,'mean_case_score':0.5,'diversity':0.9,'passed':3,'cases':32}
    monkeypatch.setattr(reply_eval,'evaluate_model',fake_eval)
    with pytest.raises(ValueError,match='select_by'):train(tmp_path/'f',source,tmp_path/'bad',english_pilot=True,select_by='vibes')
    result=train(tmp_path/'f',source,tmp_path/'pilot',english_pilot=True,select_by='replies')
    assert scored and result['selected_by']=='replies' and result['selected_from']=='best_reply.pt'
    assert result['best_reply']['score']==0.42 and (tmp_path/'pilot/best_reply.pt').exists()
    saved=torch.load(tmp_path/'pilot/best-base.pt',weights_only=False)
    assert saved['reply_eval']['reply_score']==0.42

# ──────────────────────────────────────────────────────────────────────  test_public_pipeline.py  ──────────────────────────────────────────────────────────────────────

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from Tantra import ModelConfig
from Tantra.train import TantraModel
import Tantra.conversation_train as conversation
import Tantra.pretrain as training
from Tantra import sha256


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
        conversation.cs_approve(queue,decisions,tmp_path/'bad')
    decisions.write_text(json.dumps({'record_id':'a','expert':'greeting','status':'approved',
                                     'reviewer':'test fixture reviewer','rationale':'test fixture approval'}))
    assert conversation.cs_approve(queue,decisions,tmp_path/'good')['approved']==1
    path.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        conversation.cs_approve(queue,decisions,tmp_path/'tampered')


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
    import Tantra.pretrain as public
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
    import Tantra.pretrain as corpus
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
