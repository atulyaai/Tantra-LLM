"""
Tests/test_export_benchmark.py — Unit & Integration tests for export.py, benchmark.py, and webui/server.py endpoints.
"""
import os
import sys
import tempfile
import pytest
import torch

from Tantra.config import NeuroCoreConfig
from Tantra.model import NeuroCoreModel
from Tantra.train import NeuroTrainer
from Tantra.export import export_clean_checkpoint
from Tantra.benchmark import run_benchmarks


def _create_dummy_checkpoint(path: str, step: int = 15000, best_loss: float = 3.14):
    cfg = NeuroCoreConfig(model_name="test-tiny")
    cfg.block.alra.dim = 64
    cfg.block.alra.num_heads = 4
    cfg.block.alra.head_dim = 16
    cfg.block.sgp.dim = 64
    cfg.block.num_layers = 2
    cfg.vocab.vocab_size = 32768

    model = NeuroCoreModel(cfg)
    trainer = NeuroTrainer(model, lr=1e-3, total_steps=100)
    trainer.step_count = step
    trainer.best_loss = best_loss
    trainer.total_tokens = 500_000

    trainer.save_checkpoint(path, save_optimizer=True)
    return cfg, model


def test_export_clean_checkpoint_end_to_end():
    """Verify export_clean_checkpoint properly extracts metadata and removes optimizer states."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_ckpt = os.path.join(tmpdir, "input_checkpoint.pt")
        output_ckpt = os.path.join(tmpdir, "output_clean.pt")

        _create_dummy_checkpoint(input_ckpt, step=42000, best_loss=2.85)

        exported_path = export_clean_checkpoint(input_ckpt, output_ckpt)
        assert os.path.exists(exported_path)

        data = torch.load(output_ckpt, map_location="cpu", weights_only=False)
        assert data["step_count"] == 42000
        assert data["step"] == 42000
        assert data["best_loss"] == 2.85
        assert data["total_tokens"] == 500_000
        assert data["format"] == "tantra-v1-production-clean"
        assert "model_state_dict" in data
        assert "optimizer_state_dict" not in data
        assert data["config"] is not None


def test_benchmark_script_execution():
    """Verify Tantra/benchmark.py executes without import or initialization errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "bench_ckpt.pt")
        _create_dummy_checkpoint(ckpt_path)

        # Run benchmark with cpu device
        run_benchmarks(checkpoint_path=ckpt_path, device="cpu")


def test_webui_server_api_endpoints():
    """Verify WebUI server API endpoints return valid JSON responses."""
    try:
        from fastapi.testclient import TestClient
        from webui.server import app
    except ImportError:
        pytest.skip("fastapi or testclient not available")

    client = TestClient(app)

    # 1. Status
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data or "model" in data

    # 2. Telemetry
    res = client.get("/api/telemetry")
    assert res.status_code == 200
    tel = res.json()
    assert "device" in tel

    # 3. Knowledge Graph
    res = client.get("/api/knowledge_graph")
    assert res.status_code == 200
    kg = res.json()
    assert "nodes" in kg
    assert "links" in kg
    assert len(kg["nodes"]) >= 4

    # 4. Experts
    res = client.get("/api/experts")
    assert res.status_code == 200
    exp = res.json()
    assert "experts" in exp
    assert len(exp["experts"]) > 0

    # 5. Live training status
    res = client.get("/api/training/live")
    assert res.status_code == 200
    t_status = res.json()
    assert "status" in t_status
