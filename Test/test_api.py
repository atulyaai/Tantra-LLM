import json


def test_smoke_endpoints():
    # Simple import test to ensure FastAPI app loads
    from Training.serve_api import app  # noqa: F401
    assert True


