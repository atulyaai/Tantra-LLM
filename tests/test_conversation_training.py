import json
from pathlib import Path
import pytest
from Tantra.conversation_train import balanced_order,inspect,train
from Tantra.config import ROOT


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
    from Tantra.config import ModelConfig
    from Tantra.model.tantra_model import TantraModel
    from Tantra.tokenizer import TantraTokenizer
    from Tantra.data.expert_dataset import encode_row
    from Tantra.data.public_sources import sha256
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
    import Tantra.data.train_base_pilot as resources
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
    from Tantra.config import ModelConfig
    from Tantra.model.tantra_model import TantraModel
    from Tantra.tokenizer import TantraTokenizer
    from Tantra.conversation_train import pilot_order
    import Tantra.conversation_train as module
    import Tantra.data.train_base_pilot as resources
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
