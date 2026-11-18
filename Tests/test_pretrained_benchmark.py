import json

import pytest

from Tantra.pretrained_benchmark import load_cases, run_benchmark


class Spec:
    model_id = "test/model"
    revision = "a" * 40


class FakeGateway:
    spec = Spec()

    def generate(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        return "I am Tantra." if "Who are you" in prompt or "कौन हैं" in prompt or "kaun ho" in prompt else "sample answer"


def test_load_cases_rejects_empty_duplicate_and_bad_rows(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_cases(path)
    path.write_text(
        '{"id":"x","language":"en","category":"identity","prompt":"Hi"}\n'
        '{"id":"x","language":"hi","category":"identity","prompt":"Namaste"}\n'
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)
    path.write_text('{"id":"x","language":"en"}\n')
    with pytest.raises(ValueError, match="requires"):
        load_cases(path)


def test_benchmark_records_metrics_review_and_candidate_provenance(tmp_path):
    cases = [
        {"id": "identity-en", "language": "en", "category": "identity", "prompt": "Who are you?", "identity_check": True},
        {"id": "identity-hi", "language": "hi", "category": "identity", "prompt": "आप कौन हैं?", "identity_check": True},
        {"id": "math-en", "language": "en", "category": "reasoning", "prompt": "2+2?"},
    ]
    output = tmp_path / "report.json"
    candidate = {
        "kind": "lora",
        "checkpoint_path": "./checkpoint",
        "adapter_path": "./adapter",
        "residual_candidate_path": None,
        "source_commit": "deadbeef",
    }
    report = run_benchmark(FakeGateway(), cases, output, candidate=candidate)
    raw = output.read_bytes()
    saved = json.loads(raw)
    assert raw.endswith(b"\n")
    assert report["schema_version"] == 2
    assert report["cases"] == 3 and saved["model"] == "test/model"
    assert saved["candidate"] == candidate
    assert report["identity_checks_passed"] == 2
    assert report["by_language"]["hi"]["cases"] == 1
    assert report["semantic_quality_status"] == "not_scored_human_review_required"
    assert all(row["human_review_status"] == "pending" for row in report["results"])


def test_base_candidate_provenance_is_explicit(tmp_path):
    report = run_benchmark(
        FakeGateway(),
        [{"id": "x", "language": "en", "category": "identity", "prompt": "Who are you?"}],
        tmp_path / "base.json",
    )
    assert report["candidate"]["kind"] == "base"
    assert report["candidate"]["adapter_path"] is None
