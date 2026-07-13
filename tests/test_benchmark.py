import json
from pathlib import Path

import pytest

from npdna.brain import benchmark_checkpoint


def test_benchmark_checkpoint_smoke():
    checkpoint = Path("model/npdna/best")
    if not checkpoint.exists():
        pytest.skip("No bundled checkpoint in this checkout")
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))

    result = benchmark_checkpoint(checkpoint, max_tokens=2)
    assert result["metadata"]["hidden_size"] == metadata["hidden_size"]
    assert result["load_seconds"] >= 0
    assert result["generations"]
    assert "overall_score" in result
    assert "domain_scores" in result
