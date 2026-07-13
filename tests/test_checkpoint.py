from pathlib import Path
import json

import pytest

from npdna import NpDnaCore
from npdna.tokenizer import SPECIAL_TOKENS


def test_seed_checkpoint_loads_and_generates():
    checkpoint = Path("model/npdna/best")
    if not checkpoint.exists():
        pytest.skip("No bundled checkpoint in this checkout")
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))

    core = NpDnaCore.load(checkpoint)
    assert core.config.hidden_size == metadata["hidden_size"]
    assert core.tokenizer.size > len(SPECIAL_TOKENS)
    assert core.tokenizer.capacity >= core.tokenizer.size

    text = core.generate("Hello.", max_tokens=3)
    assert isinstance(text, str)
