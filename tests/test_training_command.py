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
    import sys,pytest
    from Tantra import pretrain
    monkeypatch.setattr(pretrain,'snapshot',lambda stage:{'worker_pids':[]})
    monkeypatch.setattr(sys,'argv',['pretrain','--info','--microbatch','16'])
    pretrain.main()
    assert 'Requested microbatch: 16' in capsys.readouterr().out
    monkeypatch.setattr(sys,'argv',['pretrain','--info','--microbatch','32'])
    with pytest.raises(SystemExit):pretrain.main()
