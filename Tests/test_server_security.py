import os
import sys
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from server import app, TANTRA_API_KEY

client = TestClient(app)


def test_unauthenticated_admin_routes():
    """Unauthenticated requests to admin/mutating endpoints should return 401 Unauthorized."""
    resp_sandbox = client.post("/api/sandbox/run", json={"code": "print('test')"})
    assert resp_sandbox.status_code == 401

    resp_ckpt = client.post("/api/checkpoints", json={"checkpoint": "test.pt"})
    assert resp_ckpt.status_code == 401

    resp_clean = client.post("/api/datasets/clean", json={})
    assert resp_clean.status_code == 401


def test_authenticated_admin_routes():
    """The code runner requires explicit opt-in, even with a valid API key."""
    headers = {"X-API-Key": TANTRA_API_KEY}
    
    resp_clean = client.post("/api/datasets/clean", json={}, headers=headers)
    assert resp_clean.status_code == 501
    assert "prepare_dataset.py" in resp_clean.json()["detail"]

    resp_sandbox = client.post("/api/sandbox/run", json={"code": "print('hello from sandbox')"}, headers=headers)
    assert resp_sandbox.status_code == 403
    assert "disabled" in resp_sandbox.json()["detail"]


def test_sandbox_timeout_isolation():
    """Disabled-by-default code execution cannot be invoked with an API key."""
    headers = {"X-API-Key": TANTRA_API_KEY}
    infinite_code = "while True: pass"
    
    resp = client.post("/api/sandbox/run", json={"code": infinite_code}, headers=headers)
    assert resp.status_code == 403


def test_checkpoint_path_traversal():
    """A traversal attempt cannot select a file outside Model/."""
    headers = {"X-API-Key": TANTRA_API_KEY}
    bad_ckpt = "../../etc/passwd"
    
    resp = client.post("/api/checkpoints", json={"checkpoint": bad_ckpt}, headers=headers)
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()
