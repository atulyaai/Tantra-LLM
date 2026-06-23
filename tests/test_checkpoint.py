from pathlib import Path
import json

from npdna import NpDnaCore


def test_seed_checkpoint_loads_and_generates():
    checkpoint = Path("model/npdna/best")
    assert checkpoint.exists()
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))

    core = NpDnaCore.load(checkpoint)
    assert core.config.hidden_size == metadata["hidden_size"]
    assert core.tokenizer.size >= 8000
    assert core.tokenizer.capacity >= core.tokenizer.size

    text = core.generate("Hello.", max_tokens=3)
    assert isinstance(text, str)
