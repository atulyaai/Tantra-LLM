"""Model paths, dimensions, and device assignments."""

MODEL_CONFIG = {
    "spikingbrain": {
        "model_dim": 4096,  # DESIGN QUESTION: confirm D
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


