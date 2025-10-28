from __future__ import annotations

import os


def get_device() -> str:
    # TODO: expand with CUDA checks
    return "cuda" if os.environ.get("USE_CUDA") == "1" else "cpu"


