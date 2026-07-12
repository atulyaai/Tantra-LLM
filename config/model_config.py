"""Model paths, dimensions, and device assignments.

Memory settings are owned by config/memory_config.py.
Personality settings are owned by config/personality_config.json.
Do NOT re-declare those settings here — import or reference them instead.
"""

from config.memory_config import MEMORY_CONFIG

MODEL_CONFIG = {
    "model_dim": 4096,  # SpikingBrain hidden size
    "wm_tokens": MEMORY_CONFIG["working_tokens"],  # Working memory window (single source)
    "spikingbrain": {
        "max_seq": 32768,   # Max sequence length
        "path": "Model/spikingbrain-7b",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 24,
        "intermediate_size": 16384,
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
        "embed_dim": 4096,  # Aligned with model_dim
        "remote": True,
        "api_url": None,
        "local_path": None,
    },
    "audio": {
        "embed_dim": 4096,  # Aligned with model_dim
        "remote": False,
        "model_name": "openai/whisper-large-v3",
        "local_path": None,
    },
    # Memory settings: see config/memory_config.py (MEMORY_CONFIG)
    "memory": MEMORY_CONFIG,
    # Personality settings: see config/personality_config.json
    # Loaded at runtime by PersonalityLayer via JSON — not duplicated here.
    "compute": {
        "max_tokens": 200,
        "context_window": 4096,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    },
}
