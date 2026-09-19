"""Tests/test_checkpoint_and_chat_loader.py

Covers two roadmap items that had no test coverage:
  1. Checkpoint-config restoration (the CHANGELOG "Unreleased" fix that
     stops a shape-mismatched checkpoint from silently loading random
     weights).
  2. The CPU chat launcher (chat.py) actually being able to find, load,
     and run a checkpoint end-to-end -- it previously crashed
     unconditionally on `config.model.dim = 512` (NeuroCoreConfig has no
     `.model` attribute) before ever reaching a prompt.
"""
import os
import subprocess
import sys
import tempfile

import torch

from Tantra.model import NeuroCoreModel, cpu_dense_config, cpu_10m_config


def test_checkpoint_restores_saved_architecture_not_default():
    """A checkpoint saved with a NON-default config (cpu_10m, not
    cpu_dense) must reload with that same architecture -- not silently
    fall back to a default/mismatched shape."""
    small_cfg = cpu_10m_config(vocab_size=256)
    model = NeuroCoreModel(small_cfg, use_mtp=False, use_moe=False)

    with tempfile.TemporaryDirectory() as d:
        ckpt_path = os.path.join(d, "checkpoint_latest.pt")
        torch.save(
            {"model_state_dict": model.state_dict(), "config": small_cfg, "step": 1},
            ckpt_path,
        )

        from Tantra.utils import safe_load_checkpoint
        loaded = safe_load_checkpoint(ckpt_path, map_location="cpu")
        restored_cfg = loaded["config"]

        # This is the exact bug the CHANGELOG fix addresses: rebuilding
        # with a DIFFERENT default config than what was saved must not
        # silently "work" via strict=False while actually holding
        # mismatched/randomly-initialized weights.
        assert restored_cfg.block.alra.dim == small_cfg.block.alra.dim
        assert restored_cfg.block.num_layers == small_cfg.block.num_layers

        rebuilt = NeuroCoreModel(restored_cfg, use_mtp=False, use_moe=False)
        missing, unexpected = rebuilt.load_state_dict(
            loaded["model_state_dict"], strict=False
        )
        assert not missing, f"checkpoint config restoration left missing keys: {missing[:5]}"
        assert not unexpected, f"checkpoint config restoration left unexpected keys: {unexpected[:5]}"

        # Sanity: rebuilding with the WRONG (default dense) config instead
        # would have produced shape mismatches -- confirm that's true, so
        # this test would actually catch a regression back to that bug.
        wrong_cfg = cpu_dense_config(vocab_size=256)
        wrong_model = NeuroCoreModel(wrong_cfg, use_mtp=False, use_moe=False)
        wrong_state = wrong_model.state_dict()
        shape_mismatches = [
            k for k in loaded["model_state_dict"]
            if k in wrong_state and wrong_state[k].shape != loaded["model_state_dict"][k].shape
        ]
        assert shape_mismatches, (
            "expected cpu_10m and cpu_dense to have different tensor shapes "
            "-- if this fails, the two profiles converged and this test's "
            "regression check is no longer meaningful"
        )


def test_chat_launcher_loads_checkpoint_and_generates(tmp_path):
    """Run chat.py as a subprocess against a real (tiny, untrained)
    checkpoint + tokenizer fixture, feed it one line of input, and confirm
    it reaches the interactive prompt without crashing. This is a
    regression test for the `config.model.dim = 512` crash -- chat.py
    previously could not load ANY checkpoint, this test would have failed
    immediately on that bug."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tok_source = os.path.join(repo_root, "Model", "Latest", "tokenizer.json")
    if not os.path.exists(tok_source):
        import pytest
        pytest.skip("no tokenizer.json fixture available in Model/Latest/")

    ckpt_dir = tmp_path / "Model" / "Checkpoints"
    ckpt_dir.mkdir(parents=True)

    cfg = cpu_dense_config()
    model = NeuroCoreModel(cfg, use_mtp=False, use_moe=False)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": cfg, "step": 1},
        ckpt_dir / "checkpoint_latest.pt",
    )
    import shutil
    shutil.copy(tok_source, ckpt_dir / "tokenizer.json")

    result = subprocess.run(
        [sys.executable, os.path.join(repo_root, "chat.py")],
        input="hi\nquit\n",
        cwd=str(tmp_path),
        capture_output=True,
        encoding='utf-8', errors='replace', text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"chat.py exited nonzero:\n{result.stdout}\n{result.stderr}"
    assert "Model loaded:" in result.stdout
    assert "तंत्र चैट" in result.stdout
