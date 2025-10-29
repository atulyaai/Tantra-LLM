"""Model paths, dimensions, and device assignments."""

MODEL_CONFIG = {
    "model_dim": 4096,  # SpikingBrain-7B hidden size
    "wm_tokens": 8192,  # Working memory window
    "spikingbrain": {
        "max_seq": 32768,   # Max sequence length (full SpikingBrain-7B)
        "path": "Model/spikingbrain-7b",
        "hidden_size": 4096,  # SpikingBrain-7B dimensions
        "num_attention_heads": 32,  # SpikingBrain-7B heads
        "num_hidden_layers": 24,  # SpikingBrain-7B layers
        "intermediate_size": 16384,  # SpikingBrain-7B intermediate
        "vocab_size": 50257,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "use_cache": True,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    },
    "vision": {
        "embed_dim": 4096,  # Aligned with model_dim (Long-VITA)
        "remote": True,
        "api_url": None,
        "local_path": None,
    },
    "audio": {
        "embed_dim": 4096,  # Aligned with model_dim
        "remote": False,
        "model_name": "base",  # Whisper base for efficiency
        "local_path": None,
    },
    "memory": {
        "embedding_dim": 4096,  # Aligned with model_dim
        "max_episodic": 10000,
        "consolidation_threshold": 0.7,
        "consolidation_frequency": 100,
    },
    "personality": {
        "default_mode": "DirectAssertive",
        "override_threshold": 0.8,
    },
    "compute": {
        "max_tokens": 200,
        "context_window": 4096,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    },
}


