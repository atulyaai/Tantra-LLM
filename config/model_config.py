"""Model paths, dimensions, and device assignments."""

MODEL_CONFIG = {
    "model_dim": 4096,  # Base model dimension
    "wm_tokens": 8192,  # Working memory window
    "spikingbrain": {
        "max_seq": 32768,   # DESIGN QUESTION: confirm max seq
        "path": "Model/spikingbrain-7b",
    },
    "vision": {
        "embed_dim": 1024,
        "remote": True,
    },
    "audio": {
        "embed_dim": 1024,
        "remote": False,
    },
}


