"""
Trainer commands, checkpoints, probes, benchmarks, vocabulary and dashboard.

Merged: test_training_command.py, test_training_probes.py, test_training_benchmark.py, test_checkpoint_recovery.py, test_continued_pretraining.py, test_tokenizer_expansion.py, test_dashboard.py.
"""

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€  test_training_command.py  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import json
from Tantra.train import snapshot


def test_training_status_handles_unstarted_and_partial_metrics(tmp_path):
    assert snapshot(tmp_path)['worker_pids']==[]
    base=tmp_path/'base-training';base.mkdir()
    (base/'metrics.jsonl').write_text(json.dumps({'update':2,'loss':4,'target_tokens':8192})+'\n{"partial":')
    result=snapshot(tmp_path)
    assert result['latest']['update']==2
    assert result['checkpoint'] is None


def test_stale_locks_preserved_but_live_owner_refused(tmp_path,monkeypatch):
    import pytest
    from Tantra.pretrain import archive_stale_locks
    from Tantra import pretrain
    output=tmp_path/'base-training';output.mkdir()
    lock=output/'training.lock';lock.write_text('999999')
    monkeypatch.setattr(pretrain.psutil,'pid_exists',lambda pid:False)
    archive_stale_locks(output)
    assert not lock.exists() and len(list(output.glob('training.lock.stale-*')))==1
    lock.write_text('123')
    monkeypatch.setattr(pretrain.psutil,'pid_exists',lambda pid:True)
    with pytest.raises(RuntimeError,match='alive'):archive_stale_locks(output)
    assert lock.exists()


def test_explicit_microbatch_option_info_and_invalid_choice(monkeypatch,capsys):
    import sys, pytest
    from Tantra import pretrain
    monkeypatch.setattr(pretrain,'snapshot',lambda stage:{'worker_pids':[]})
    monkeypatch.setattr(sys,'argv',['pretrain','--info','--microbatch','16'])
    pretrain.main()
    assert 'Requested microbatch: 16' in capsys.readouterr().out
    monkeypatch.setattr(sys,'argv',['pretrain','--info','--microbatch','32'])
    with pytest.raises(SystemExit):pretrain.main()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€  test_training_probes.py  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import json
import torch
from Tantra import ModelConfig
from Tantra.train import TantraModel
from Tantra.tokenizer import TantraTokenizer
from Tantra.diagnostics import run_probes


def test_probes_preserve_weights_rng_and_training_mode(tmp_path):
    tok=TantraTokenizer.train(['Hello! Who are you? I am Tantra.']*3,vocab_size=280,min_frequency=1)
    model=TantraModel(ModelConfig(vocab_size=tok.vocab_size,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,mlp_mult=4,max_seq_len=128));model.train()
    before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    suite=tmp_path/'suite.json';suite.write_text(json.dumps({'cases':[{'language':'en','question':'Who are you?','reference':'I am Tantra.'}]}))
    result=run_probes(model,tok,suite,7,True,tmp_path/'out')
    assert model.training and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert result['languages']['en']['questions']==1
    assert 'answer' in result['responses'][0]
    assert (tmp_path/'out/probe_history.jsonl').exists()


def test_probe_console_handles_non_unicode_windows_output(monkeypatch):
    import io
    import Tantra.diagnostics as probes
    data=io.BytesIO();stream=io.TextIOWrapper(data,encoding='ascii')
    monkeypatch.setattr(probes.sys,'stdout',stream)
    probes.console_print('à¤¨à¤®à¤¸à¥à¤¤à¥‡')
    stream.flush()
    assert b'\\u' in data.getvalue()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€  test_training_benchmark.py  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import pytest
from Tantra.diagnostics import choose


def test_choose_fastest_safe_microbatch():
    assert choose([{'microbatch':2,'safe':True,'tokens_per_second':400},
                   {'microbatch':8,'safe':True,'tokens_per_second':350},
                   {'microbatch':16,'safe':False,'tokens_per_second':900}])==2
    with pytest.raises(MemoryError):choose([{'microbatch':16,'safe':False}])

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€  test_checkpoint_recovery.py  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import json
from pathlib import Path

import pytest
import torch

from Tantra.train import CheckpointCorruptError, CheckpointIdentityError, generations, load_resumable, save_resumable


def payload(step=0,identity=None):
    return {'config':{'width':2},'identity':identity or {'dataset':'same'},
            'state':{'weight':torch.tensor([[float(step),1.]])},
            'optimizer':{'state':{},'param_groups':[]},
            'torch_rng':torch.get_rng_state(),'cursor':step,'updates':step,'tokens':step*10}


def load(path,identity=None,config=None):
    return load_resumable(path,identity_key='identity',expected_identity=identity or {'dataset':'same'},
                          expected_config=config or {'width':2},
                          required=('state','optimizer','torch_rng','cursor','updates','tokens'))


def test_three_generations_and_corrupt_latest_fallback_preserve_files(tmp_path):
    path=tmp_path/'latest.pt'
    for step in range(5):save_resumable(path,payload(step))
    assert [torch.load(p,weights_only=False)['updates'] for p in generations(path)]==[4,3,2]
    assert len(list(tmp_path.glob('*.pt')))==3
    path.write_bytes(b'broken ZIP header')
    result,evidence=load(path)
    assert result['updates']==3
    assert evidence['fallback_used'] and evidence['integrity_verified']
    assert evidence['rejected'][0]['path']==str(path)
    assert path.read_bytes()==b'broken ZIP header'  # loading never destroys evidence


def test_identity_or_architecture_mismatch_never_uses_older_compatible_checkpoint(tmp_path):
    path=tmp_path/'latest.pt';save_resumable(path,payload(1))
    save_resumable(path,payload(2,{'dataset':'different'}))
    with pytest.raises(CheckpointIdentityError,match='identity'):load(path)
    with pytest.raises(CheckpointIdentityError,match='configuration'):
        load(path,identity={'dataset':'different'},config={'width':3})


def test_corrupt_checksum_cannot_be_blessed_during_rotation(tmp_path):
    path=tmp_path/'latest.pt';save_resumable(path,payload(1));save_resumable(path,payload(2))
    integrity=Path(str(path)+'.integrity.json');integrity.write_text('{malformed')
    result,evidence=load(path)
    assert result['updates']==1 and evidence['fallback_used']
    save_resumable(path,payload(3))
    assert load(path)[0]['updates']==3
    path.write_bytes(b'broken primary')
    # previous-1 still has the rejected generation; previous-2 is last known healthy.
    result,evidence=load(path)
    assert result['updates']==1 and len(evidence['rejected'])==2


def test_semantically_unhealthy_checkpoint_falls_back_even_with_valid_checksum(tmp_path):
    path=tmp_path/'latest.pt';save_resumable(path,payload(1))
    bad=payload(2);bad['optimizer']['state']={'step':{'exp_avg':torch.tensor([float('nan')])}}
    save_resumable(path,bad)
    result,evidence=load(path)
    assert result['updates']==1 and 'Non-finite' in evidence['rejected'][0]['error']


def test_legacy_checkpoint_is_explicit_and_all_corrupt_stops(tmp_path):
    path=tmp_path/'latest.pt';torch.save(payload(1),path)
    result,evidence=load(path)
    assert result['updates']==1 and evidence['legacy_without_checksum'] and not evidence['integrity_verified']
    path.write_bytes(b'invalid')
    with pytest.raises(CheckpointCorruptError,match='three generations'):load(path)


def test_failed_serialization_keeps_last_checkpoint_and_generations(tmp_path,monkeypatch):
    path=tmp_path/'latest.pt';save_resumable(path,payload(1));original=path.read_bytes()
    def fail(*args,**kwargs):raise OSError('disk full')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError,match='disk full'):save_resumable(path,payload(2))
    assert path.read_bytes()==original
    assert not list(tmp_path.glob('*.tmp-*'))


def test_byte_tampering_detected_before_deserialization(tmp_path,monkeypatch):
    path=tmp_path/'latest.pt';save_resumable(path,payload(1))
    data=bytearray(path.read_bytes());data[len(data)//2]^=1;path.write_bytes(data)
    calls=[];monkeypatch.setattr(torch,'load',lambda *a,**kw:calls.append(a))
    with pytest.raises(CheckpointCorruptError):load(path)
    assert calls==[]


def test_source_gate_rejects_nan_and_bound_artifact_mismatch_before_model_load(tmp_path,monkeypatch):
    import json
    import pytest
    import Tantra.conversation_train as module
    records={s:[] for s in ('train','val','test')}
    monkeypatch.setattr(module,'inspect',lambda bundle:(records,{'state':'data_ready','bundle_manifest_sha256':'fixture'}))
    source=tmp_path/'source';source.mkdir()
    report={'state':'pilot_complete','baseline_validation':{'hi':{'loss':2},'en':{'loss':2}},
            'latest_validation':{'hi':{'loss':float('nan')},'en':{'loss':1}}}
    (source/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='finite improving'):
        module.train(tmp_path/'bundle',source,tmp_path/'bad-nan')
    assert not (tmp_path/'bad-nan').exists()
    report['latest_validation']['hi']['loss']=1
    report['exported_base_sha256']='not-the-source-hash'
    (source/'base.pt').write_bytes(b'fixture source bytes')
    (source/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='checksum mismatch'):
        module.train(tmp_path/'bundle',source,tmp_path/'bad-hash')
    assert not (tmp_path/'bad-hash').exists()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€  test_continued_pretraining.py  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

from Tantra.pretrain import pilot_passed


def test_continuation_requires_finished_pilot_and_both_languages_improving():
    report={'state':'pilot_complete','baseline_validation':{'hi':{'loss':10},'en':{'loss':9}},
            'latest_validation':{'hi':{'loss':8},'en':{'loss':8}}}
    assert pilot_passed(report)
    report['state']='paused_update_limit'
    assert not pilot_passed(report)
    report['state']='pilot_complete';report['latest_validation']['hi']['loss']=11
    assert not pilot_passed(report)
    report['latest_validation']['hi']['loss']=float('nan')
    assert not pilot_passed(report)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€  test_tokenizer_expansion.py  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import pytest
from Tantra.tokenizer import TantraTokenizer
from Tantra.tokenizer import extend_vocabulary


def test_vocabulary_extension_preserves_ids_and_old_merge_priority():
    base=TantraTokenizer.train(['hello hello hi there good morning'],vocab_size=280,min_frequency=1)
    training=[' '.join('exampleword'+str(i) for i in range(1000))]*2
    learned=TantraTokenizer.train(training,vocab_size=500,min_frequency=1)
    expanded=extend_vocabulary(base,learned,base.vocab_size+40)
    assert base.vocab_size<expanded.vocab_size<=base.vocab_size+40
    for token,identity in base.tok.get_vocab().items():
        assert expanded.tok.token_to_id(token)==identity
    for text in ['hello hi there','exampleword123 exampleword99','à¤¨à¤®à¤¸à¥à¤¤à¥‡!','Namaste, kaise ho?']:
        assert expanded.decode(expanded.encode(text))==text
        assert len(expanded.encode(text))<=len(base.encode(text))
    with pytest.raises(ValueError,match='exceed'):
        extend_vocabulary(base,learned,base.vocab_size)


def test_embedding_migration_preserves_old_rows_and_rejects_renumbering():
    import torch
    from Tantra.tokenizer import expand_embedding
    base=TantraTokenizer.train(['hello hello hi there'],vocab_size=280,min_frequency=1)
    learned=TantraTokenizer.train([' '.join('word'+str(i) for i in range(1000))]*2,vocab_size=500,min_frequency=1)
    expanded=extend_vocabulary(base,learned,base.vocab_size+40)
    weight=torch.randn(base.vocab_size,16)
    result=expand_embedding(weight,base,expanded)
    assert result.shape==(expanded.vocab_size,16)
    assert torch.equal(result[:base.vocab_size],weight)
    assert torch.isfinite(result).all()
    with pytest.raises(ValueError,match='IDs'):
        expand_embedding(weight,base,learned)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€  test_dashboard.py  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

from Tantra.dashboard import speed,spark,render
from rich.console import Console
import io


def test_dashboard_speed_uses_real_timestamps():
    assert speed([]) is None
    assert speed([{'timestamp':10,'target_tokens':100},{'timestamp':12,'target_tokens':500}])==200
    assert spark([1,1,1])=='...'


def test_dashboard_renders_empty_stage_without_claiming_training(tmp_path):
    stream=io.StringIO();console=Console(file=stream,width=110,color_system=None)
    console.print(render(tmp_path))
    output=stream.getvalue()
    assert 'WORKER STOPPED' in output and 'pending' in output
    assert 'paused' in output and 'stopped' in output
    assert 'No live training worker' in output


def test_loss_only_probe_does_not_hide_latest_generated_answers(tmp_path):
    import json
    directory=tmp_path/'base-training';directory.mkdir()
    records=[{'update':200,'responses':[{'language':'en','question':'Who made you?','answer':'A recorded reply'}]},
             {'update':250,'responses':[{'language':'en','question':'Who made you?','answer_loss':7}]}]
    (directory/'probe_history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    stream=io.StringIO();console=Console(file=stream,width=110,color_system=None)
    console.print(render(tmp_path))
    assert 'A recorded reply' in stream.getvalue() and 'UPDATE 200' in stream.getvalue()
