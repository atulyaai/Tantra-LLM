"""Contract tests for the versioned research architecture manifest.

These tests validate decisions and safety boundaries only. They do not run
training, compression, tokenizer migration, model growth or accelerator work.
"""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "model_family.yaml"


def load_manifest():
    with MANIFEST.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_final_architecture_decisions_are_pinned():
    manifest = load_manifest()
    assert manifest["strategy"]["growth"] == "family"
    assert set(manifest["strategy"]["dna_compression"]) == {
        "storage_codec",
        "architecture_parameter_sharing",
    }
    assert manifest["strategy"]["tokenizer"]["target_vocab_size"] == 65536
    assert manifest["strategy"]["tokenizer"]["status"] == "candidate_only"
    assert manifest["strategy"]["runtime"] == "balanced"


def test_existing_native_artifacts_are_protected():
    compatibility = load_manifest()["compatibility"]
    assert compatibility["preserve_native_tantra"] is True
    assert compatibility["preserve_existing_tokenizer"] is True
    assert compatibility["preserve_existing_checkpoints"] is True
    assert compatibility["preserve_existing_experts"] is True
    assert compatibility["deploy_automatically"] is False


def test_model_family_is_ordered_and_sparse_scale_is_explicit():
    family = load_manifest()["model_family"]
    ids = [entry["id"] for entry in family]
    assert ids == [
        "edge-0_8b",
        "compact-2b",
        "standard-4b",
        "quality-7b-9b",
        "large-14b-plus",
        "sparse-100b-1t",
    ]
    sparse = family[-1]
    assert sparse["report_total_and_active_parameters_separately"] is True
    assert sparse["target_parameters"]["max"] == 1_000_000_000_000


def test_tokenizer_comparison_keeps_existing_ids_compatible():
    tokenizer = load_manifest()["tokenizer_candidates"]
    assert tokenizer["compare_vocab_sizes"] == [16384, 32768, 65536]
    assert tokenizer["baseline"] == "existing_model_tokenizer"
    assert set(tokenizer["languages"]) == {"hi", "en", "hinglish"}


def test_balanced_runtime_has_cpu_reference_and_optional_accelerators():
    runtime = load_manifest()["runtime"]
    assert runtime["reference"] == "cpu"
    assert runtime["policy"] == "capability_and_benchmark_driven"
    assert {"cuda", "rocm"}.issubset(set(runtime["optional_backends"]))
    assert "output_agreement" in runtime["required_metrics"]
