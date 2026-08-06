"""Combined tests for npdna utilities: CLI, agent defaults, benchmarking, checkpoint loading, and response memory."""

import json
from pathlib import Path

import pytest

from npdna import NpDnaAgent, NpDnaCore
from npdna.brain import benchmark_checkpoint
from npdna.serving import DEFAULT_CHECKPOINTS, chat_main, infer_max_tokens, info_main
from npdna.cognition import FastResponseMemory
from npdna.tokenizer import SPECIAL_TOKENS


def test_info_main_prints_seed_summary(capsys):
    info_main()
    out = capsys.readouterr().out
    assert "NP-DNA seed configuration" in out
    assert "initial_vocab" in out


def test_chat_main_defaults_to_best_when_present(monkeypatch, tmp_path, capsys):
    latest = tmp_path / "latest"
    best = tmp_path / "best"
    latest.mkdir()
    best.mkdir()

    class FakeCore:
        @classmethod
        def load(cls, checkpoint: Path):
            assert checkpoint == best
            return cls()

        def generate(self, prompt: str, max_tokens: int, **kwargs):
            assert prompt == "Hello"
            assert max_tokens == 64
            assert kwargs["context_window"] == 256
            return "ok"

    monkeypatch.setattr("npdna.serving.DEFAULT_CHECKPOINTS", (best, latest))
    monkeypatch.setattr("npdna.serving.NpDnaCore", FakeCore)
    monkeypatch.setattr("sys.argv", ["npdna-chat", "Hello"])

    chat_main()

    out = capsys.readouterr().out
    assert f"Loaded checkpoint: {best}" in out
    assert out.rstrip().endswith("ok")


def test_default_checkpoint_order_prefers_best():
    assert DEFAULT_CHECKPOINTS[0].as_posix().endswith("model/latest")


def test_infer_max_tokens_scales_with_prompt_shape():
    assert infer_max_tokens("What is gravity?") == 40
    assert infer_max_tokens("Hello") == 64
    assert infer_max_tokens("Write a Python function.") == 120
    assert infer_max_tokens(" ".join(["word"] * 30)) == 96


def test_agent_defaults_are_local_only():
    agent = NpDnaAgent(NpDnaCore.from_config("seed"))
    assert "cortex_search" in agent.tools
    assert "cortex_store" in agent.tools
    assert "math_eval" in agent.tools
    assert "web_search" not in agent.tools
    assert "code_execute" not in agent.tools


def test_benchmark_checkpoint_smoke():
    checkpoint = Path("model/latest")
    if not checkpoint.exists():
        pytest.skip("No bundled checkpoint in this checkout")
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))

    result = benchmark_checkpoint(checkpoint, max_tokens=2)
    assert result["metadata"]["hidden_size"] == metadata["hidden_size"]
    assert result["load_seconds"] >= 0
    assert result["generations"]
    assert "overall_score" in result
    assert "domain_scores" in result


def test_seed_checkpoint_loads_and_generates():
    checkpoint = Path("model/latest")
    if not checkpoint.exists():
        pytest.skip("No bundled checkpoint in this checkout")
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))

    core = NpDnaCore.load(checkpoint)
    assert core.config.hidden_size == metadata["hidden_size"]
    assert core.tokenizer.size > len(SPECIAL_TOKENS)
    assert core.tokenizer.capacity >= core.tokenizer.size

    text = core.generate("Hello.", max_tokens=3)
    assert isinstance(text, str)


def test_written_response_is_returned_for_equivalent_question():
    memory = FastResponseMemory()
    memory.write("What is 17 plus 29?", "17 plus 29 is 46.")
    assert memory.match("what is 17 plus 29 ?") == ("17 plus 29 is 46.", 1.0)


def test_unrelated_question_does_not_match():
    memory = FastResponseMemory()
    memory.write("What is 17 plus 29?", "17 plus 29 is 46.")
    assert memory.match("Explain photosynthesis.") is None
