import os
import sys
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from webui.server import app, TANTRA_API_KEY

client = TestClient(app)


def test_get_dashboard():
    """GET / should render index.html with status 200."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Tantra" in resp.text


def test_list_models():
    """GET /v1/models should return model list."""
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert data["data"][0]["id"] == "tantra-neurocore-v1"


def test_capabilities_reports_sandbox_availability():
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    assert resp.json()["sandbox_enabled"] is False


def test_telemetry_and_status():
    """GET /api/telemetry and /api/status should return dynamic hardware & model metrics."""
    resp = client.get("/api/telemetry")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert "hardware" in data
    assert "cpu_threads" in data["hardware"]
    assert "simd" in data["hardware"]

    resp_status = client.get("/api/status")
    assert resp_status.status_code == 200


def test_experts_endpoint():
    """GET /api/experts should return expert router items and top_k configuration."""
    resp = client.get("/api/experts")
    assert resp.status_code == 200
    data = resp.json()
    assert "experts" in data
    assert len(data["experts"]) > 0
    assert "top_k" in data


def test_knowledge_graph_endpoint():
    """GET /api/knowledge_graph should return dynamic nodes and links topology."""
    resp = client.get("/api/knowledge_graph")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "links" in data
    assert len(data["nodes"]) >= 4


def test_chats_crud_operations():
    """Test full CRUD lifecycle of chat sessions."""
    # 1. List chats
    resp_list = client.get("/api/chats")
    assert resp_list.status_code == 200

    # 2. Create new session
    test_id = "test-session-101"
    session_payload = {
        "id": test_id,
        "title": "Test Chat Session",
        "archived": False,
        "messages": [{"role": "user", "content": "Hello Tantra"}]
    }
    resp_create = client.post("/api/chats", json=session_payload)
    assert resp_create.status_code == 200

    # 3. Rename session
    resp_rename = client.post(f"/api/chats/{test_id}/rename", json={"title": "Renamed Session"})
    assert resp_rename.status_code == 200

    # 4. Archive session
    resp_archive = client.post(f"/api/chats/{test_id}/archive")
    assert resp_archive.status_code == 200
    assert resp_archive.json()["archived"] is True

    # 5. Auto-title
    resp_title = client.post("/api/chats/auto_title", json={"prompt": "Write a Python script to sort an array"})
    assert resp_title.status_code == 200
    assert "title" in resp_title.json()

    # 6. Delete session
    resp_del = client.delete(f"/api/chats/{test_id}")
    assert resp_del.status_code == 200


def test_memory_bank_crud():
    """Test memory bank addition, listing, and deletion."""
    resp_get = client.get("/api/memory")
    assert resp_get.status_code == 200

    resp_add = client.post("/api/memory", json={"category": "User Fact", "fact": "User is building Tantra LLM"})
    assert resp_add.status_code == 200
    mem_item = resp_add.json()["item"]

    resp_del = client.delete(f"/api/memory/{mem_item['id']}")
    assert resp_del.status_code == 200


def test_tokenize_endpoint():
    """Test text tokenization endpoint."""
    resp = client.post("/api/tokenize", json={"text": "Tantra Quantum MoE Engine"})
    assert resp.status_code == 200
    data = resp.json()
    assert "tokens" in data
    assert data["total_count"] > 0


def test_datasets_endpoint():
    """Test listing registered datasets."""
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_admin_security_and_disabled_sandbox():
    """Admin endpoints require the local API key and code execution stays off."""
    assert client.post("/api/sandbox/run", json={"code": "print('test')"}).status_code == 401
    assert client.post("/api/checkpoints", json={"checkpoint": "test.pt"}).status_code == 401
    assert client.post("/api/datasets/clean", json={}).status_code == 401

    headers = {"X-API-Key": TANTRA_API_KEY}
    clean = client.post("/api/datasets/clean", json={}, headers=headers)
    assert clean.status_code == 501
    assert "not available" in clean.json()["detail"]
    sandbox = client.post("/api/sandbox/run", json={"code": "while True: pass"}, headers=headers)
    assert sandbox.status_code == 403
    traversal = client.post("/api/checkpoints", json={"checkpoint": "../../etc/passwd"}, headers=headers)
    assert traversal.status_code == 404
