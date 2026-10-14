"""
Tantra Config
=============
Model size presets (tiny -> 1B -> 7B) and expert taxonomy helpers.

Design (CPU-first):
  * one small shared base transformer (tied embeddings, small vocab)
  * many tiny LoRA experts; the router picks top-k per request, so the
    cost of a forward pass does NOT grow with the number of experts
  * the base grows in depth (function-preserving) when a stage is done
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY = ROOT / "Tantra" / "config" / "taxonomy.yaml"


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    n_kv_heads: int = 2          # grouped-query attention: smaller KV cache on CPU
    mlp_mult: float = 2.75       # SwiGLU hidden = d_model * mlp_mult (rounded to 64)
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    dropout: float = 0.0
    lora_rank: int = 8           # default rank for a new expert
    lora_alpha: float = 16.0
    lora_targets: List[str] = field(default_factory=lambda: ["q", "v", "o", "down"])

    @property
    def mlp_hidden(self) -> int:
        return int(round(self.d_model * self.mlp_mult / 64) * 64)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Parameter counts are for the shared base (experts add ~0.1-1M each).
PRESETS: Dict[str, ModelConfig] = {
    "tiny":  ModelConfig(vocab_size=8192,  d_model=256,  n_layers=6,  n_heads=4,  n_kv_heads=2),   # ~6.5M - smoke tests
    "small": ModelConfig(vocab_size=16384, d_model=512,  n_layers=8,  n_heads=8,  n_kv_heads=4),   # ~30M - first real stage
    "base":  ModelConfig(vocab_size=32000, d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4),   # ~210M
    "1b":    ModelConfig(vocab_size=32000, d_model=2048, n_layers=21, n_heads=16, n_kv_heads=4),   # ~1.0B
    "7b":    ModelConfig(vocab_size=32000, d_model=4096, n_layers=36, n_heads=32, n_kv_heads=8),   # ~6.6B
}


def get_preset(name: str) -> ModelConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    return ModelConfig.from_dict(PRESETS[name].to_dict())


def estimate_params(cfg: ModelConfig) -> int:
    d, h = cfg.d_model, cfg.mlp_hidden
    kv = d // cfg.n_heads * cfg.n_kv_heads
    per_layer = d * d * 2 + d * kv * 2 + 3 * d * h + 2 * d
    return cfg.vocab_size * d + cfg.n_layers * per_layer + d


# ── Taxonomy ──────────────────────────────────────────────────────────────────

def load_taxonomy(path: str | Path = DEFAULT_TAXONOMY) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_taxonomy(taxonomy: Dict) -> List[str]:
    errors = []
    seen = set()
    for domain, info in taxonomy.items():
        if not isinstance(info, dict) or "experts" not in info:
            errors.append(f"{domain}: missing 'experts'")
            continue
        if "count" in info and len(info["experts"]) != info["count"]:
            errors.append(f"{domain}: count {info['count']} != {len(info['experts'])} experts")
        for e in info["experts"]:
            if not e.replace("_", "").isalnum():
                errors.append(f"{domain}.{e}: invalid name")
            if e in seen:
                errors.append(f"{domain}.{e}: duplicate expert name")
            seen.add(e)
    return errors


def expert_domains(taxonomy: Dict) -> Dict[str, str]:
    """expert name -> domain"""
    return {e: d for d, info in taxonomy.items() if isinstance(info, dict) for e in info.get("experts", [])}


def expert_rank(taxonomy: Dict, expert: str, default: int = 8) -> int:
    dom = expert_domains(taxonomy).get(expert)
    return int(taxonomy[dom].get("rank", default)) if dom else default


def print_taxonomy_summary(taxonomy: Dict) -> None:
    total = 0
    for domain, info in taxonomy.items():
        if isinstance(info, dict) and "experts" in info:
            total += len(info["experts"])
            print(f"{domain:18s} {len(info['experts']):3d} experts  rank {info.get('rank', '?')}")
    print(f"{'TOTAL':18s} {total:3d} experts")


# Unpickling shim for old checkpoints that stored custom *Config classes.
class _LegacyConfig:
    def __init__(self, *a, **kw):
        self.__dict__.update(kw)

    def __setstate__(self, state):
        self.__dict__.update(state)


def __getattr__(name):
    if name.endswith("Config") or name.endswith("State"):
        cls = type(name, (_LegacyConfig,), {})
        globals()[name] = cls
        return cls
    raise AttributeError(name)
