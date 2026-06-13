from pathlib import Path

from npdna import NpDnaCore


def test_seed_checkpoint_loads_and_generates():
    checkpoint = Path("model/npdna_v3/best")
    assert checkpoint.exists()

    core = NpDnaCore.load(checkpoint)
    assert core.config.hidden_size == 128
    assert core.tokenizer.size >= 8000
    assert core.tokenizer.capacity >= core.tokenizer.size

    text = core.generate("Hello.", max_tokens=3)
    assert isinstance(text, str)
