from npdna.benchmark import benchmark_checkpoint


def test_benchmark_checkpoint_smoke():
    result = benchmark_checkpoint("model/npdna_v3/best", prompts=["Hello."], max_tokens=2)
    assert result["metadata"]["hidden_size"] == 128
    assert result["load_seconds"] >= 0
    assert result["generations"]
