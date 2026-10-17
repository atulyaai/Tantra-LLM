import json
import pyarrow as pa
import pyarrow.parquet as pq
from Tantra.data import instruction_candidates as module


def test_candidate_acquisition_does_not_auto_approve_or_train_on_source_test(tmp_path,monkeypatch):
    train=tmp_path/'train.parquet';test=tmp_path/'test.parquet'
    shared={'inputs':'Reserved question','targets':'Reserved answer','language_code':'eng','annotation_type':'human'}
    hi={'inputs':'हिंदी प्रश्न','targets':'हिंदी उत्तर','language_code':'hin','annotation_type':'human'}
    pq.write_table(pa.Table.from_pylist([shared,hi]),train);pq.write_table(pa.Table.from_pylist([shared]),test)
    lock=tmp_path/'lock.json'
    sources=[{'id':'fixture-train','repo':'fixture','revision':'test','license':'test-only','path':'train','kind':'candidate'},
             {'id':'fixture-test','repo':'fixture','revision':'test','license':'test-only','path':'test','kind':'reserved_source_test'}]
    lock.write_text(json.dumps({'sources':sources}))
    monkeypatch.setattr(module,'LOCK',lock)
    monkeypatch.setattr(module,'fetch',lambda source,root:train if source['path']=='train' else test)
    out=tmp_path/'out';module.run(out)
    report=json.loads((out/'index.json').read_text())
    assert report['approved_training_examples']==0
    assert report['source_test_duplicates_excluded_from_candidates']==1
    rows=[json.loads(line) for line in (out/'pending-review.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==1 and rows[0]['language']=='hi'
    assert rows[0]['review']=={'status':'pending','reviewer':None}
    assert rows[0]['expert'] is None
