"""Model paths, dimensions, and device assignments."""

MODEL_CONFIG = {
    "model_dim": 4096,  # SpikingBrain hidden size
    "wm_tokens": 8192,  # Working memory window
    "spikingbrain": {
        "max_seq": 32768,   # Max sequence length
        "path": "Model/spikingbrain-7b",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 24,
        "intermediate_size": 16384,
    },
    "vision": {
        "embed_dim": 1024,
        "remote": True,
    },
    "audio": {
        "embed_dim": 1024,
        "remote": False,
    },
    "memory": {
        "embedding_dim": 768,
        "max_episodic": 10000,
    },
}


