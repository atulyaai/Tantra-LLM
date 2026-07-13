"""Model paths, dimensions, and device assignments.

Memory settings are owned by npdna/fusion_config/memory_config.py.
Personality settings are owned by npdna/fusion_config/personality_config.json.
Do NOT re-declare those settings here — import or reference them instead.
"""

from npdna.fusion_config.memory_config import MEMORY_CONFIG

MODEL_CONFIG = {
    "model_dim": 4096,  # Fusion embedding dimension
    "wm_tokens": MEMORY_CONFIG["working_tokens"],  # Working memory window (single source)
    "npdna": {
        "checkpoint_path": "model/npdna/best",
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
