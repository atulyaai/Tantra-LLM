from pathlib import Path

from npdna.cli import DEFAULT_CHECKPOINTS, chat_main, infer_max_tokens, info_main


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

    monkeypatch.setattr("npdna.cli.DEFAULT_CHECKPOINTS", (best, latest))
    monkeypatch.setattr("npdna.cli.NpDnaCore", FakeCore)
    monkeypatch.setattr("sys.argv", ["npdna-chat", "Hello"])

    chat_main()

    out = capsys.readouterr().out
    assert f"Loaded checkpoint: {best}" in out
    assert out.rstrip().endswith("ok")


def test_default_checkpoint_order_prefers_best():
    assert DEFAULT_CHECKPOINTS[0].as_posix().endswith("model/npdna/best")
    assert DEFAULT_CHECKPOINTS[1].as_posix().endswith("model/npdna/latest")


def test_infer_max_tokens_scales_with_prompt_shape():
    assert infer_max_tokens("What is gravity?") == 40
    assert infer_max_tokens("Hello") == 64
    assert infer_max_tokens("Write a Python function.") == 120
    assert infer_max_tokens(" ".join(["word"] * 30)) == 96
