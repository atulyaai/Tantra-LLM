"""Memory thresholds and retention policies.

Single source of truth for all memory-related configuration.
Imported by model_config.py — do NOT re-declare these keys elsewhere.
"""

MEMORY_CONFIG = {
    # Working memory window (token count)
    "working_tokens": 8192,
    # Episodic memory consolidation threshold
    "episodic_threshold": 0.7,
    # Embedding dimensionality (aligned with model hidden size)
    "embedding_dim": 4096,
    # Maximum episodic memories before eviction
    "max_episodic": 10000,
    # How often (in steps) to consolidate similar memories
    "consolidation_frequency": 100,
}
