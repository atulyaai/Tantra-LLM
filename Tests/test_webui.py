"""WebUI server: status, chat (plain + streaming), saved chats, checkpoint safety."""
import json

import pytest
import torch
from fastapi.testclient import TestClient

import WebUI.server as server
from Tantra.config import NeuroCoreConfig
from Tantra.model import NeuroCoreModel
from Tantra.tokenizer import build_tokenizer, load_tokenizer


@pytest.fixture()
def client(tmp_path, monkeypatch):
    data = tmp_path / "d.jsonl"
    data.write_text("\n".join(json.dumps({"text": f"नमस्ते दुनिया {i} hello"}, ensure_ascii=False) for i in range(50)),
                    encoding="utf-8")
    model_dir = tmp_path / "Model"
    tok = load_tokenizer(build_tokenizer([str(data)], str(model_dir), vocab_size=300, max_mb=1))
    cfg = NeuroCoreConfig.tiny()
    cfg.vocab.vocab_size = tok.vocab_size
    torch.save({"model_state_dict": NeuroCoreModel(cfg, use_mtp=False).state_dict(), "config": cfg, "step_count": 7},
               model_dir / "best.pt")
    monkeypatch.setattr(server, "MODEL_DIR", str(model_dir))
    monkeypatch.setattr(server, "CHATS_FILE", str(model_dir / "chats.json"))
    monkeypatch.setattr(server, "STATUS_FILE", str(model_dir / "status.json"))
    monkeypatch.setattr(server, "PROBE_FILE", str(model_dir / "probe.jsonl"))
    server.state.update(model=None, tok=None, info={})
    return TestClient(server.app)


def test_chat_plain_and_stream(client):
    msgs = [{"role": "user", "content": "नमस्ते"}]
    r = client.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 5})
    assert r.status_code == 200 and r.json()["usage"]["completion_tokens"] <= 5
    s = client.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 5, "stream": True})
    assert s.text.strip().endswith("data: [DONE]")
    status = client.get("/api/status").json()
    assert status["model"]["step"] == 7 and status["hardware"]["device"]


def test_saved_chats_roundtrip(client):
    c = client.post("/api/chats", json={"messages": [{"role": "user", "content": "पहला सवाल"}]}).json()
    assert client.get("/api/chats").json()[c["id"]]["title"] == "पहला सवाल"
    client.delete(f"/api/chats/{c['id']}")
    assert c["id"] not in client.get("/api/chats").json()


def test_checkpoint_switch_refuses_paths_outside_model_dir(client):
    assert client.post("/api/checkpoints/switch", json={"path": "../main.py"}).status_code == 404


def test_last_message_must_be_user(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "assistant", "content": "hi"}]})
    assert r.status_code == 400
