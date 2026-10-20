import json
from pathlib import Path

import pytest
import torch

from Tantra.training.checkpoints import (
    CheckpointCorruptError, CheckpointIdentityError, generations,
    load_resumable, save_resumable,
)


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
