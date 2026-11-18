import json
import pytest
from Tantra.pretrained_benchmark import load_cases, run_benchmark

class Spec:
    model_id="test/model"
    revision="a"*40
class FakeGateway:
    spec=Spec()
    def generate(self,messages,**kwargs):
        prompt=messages[-1]["content"]
        return "I am Tantra." if "Who are you" in prompt or "कौन हैं" in prompt or "kaun ho" in prompt else "sample answer"

def test_load_cases_rejects_empty_duplicate_and_bad_rows(tmp_path):
    p=tmp_path/"empty.jsonl"; p.write_text("")
    with pytest.raises(ValueError,match="empty"): load_cases(p)
    p.write_text('{"id":"x","language":"en","category":"identity","prompt":"Hi"}\n{"id":"x","language":"hi","category":"identity","prompt":"Namaste"}\n')
    with pytest.raises(ValueError,match="duplicate"): load_cases(p)
    p.write_text('{"id":"x","language":"en"}\n')
    with pytest.raises(ValueError,match="requires"): load_cases(p)

def test_benchmark_records_language_metrics_and_human_review(tmp_path):
    cases=[
        {"id":"identity-en","language":"en","category":"identity","prompt":"Who are you?","identity_check":True},
        {"id":"identity-hi","language":"hi","category":"identity","prompt":"आप कौन हैं?","identity_check":True},
        {"id":"math-en","language":"en","category":"reasoning","prompt":"2+2?"},
    ]
    output=tmp_path/"report.json"
    report=run_benchmark(FakeGateway(),cases,output)
    saved=json.loads(output.read_text())
    assert report["cases"]==3 and saved["model"]=="test/model"
    assert report["identity_checks_passed"]==2
    assert report["by_language"]["hi"]["cases"]==1
    assert report["semantic_quality_status"]=="not_scored_human_review_required"
    assert all(row["human_review_status"]=="pending" for row in report["results"])
