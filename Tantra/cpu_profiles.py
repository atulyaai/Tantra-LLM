"""Small, explicit CPU-first model profiles.

These profiles are intentionally separate from ``NeuroCoreConfig.small()`` so
they never alter an existing checkpoint's architecture during a resume.
"""
from __future__ import annotations

from Tantra.config import NeuroCoreConfig
from Tantra.model import NeuroCoreModel


def cpu_dense_config(vocab_size: int = 32768, attention_kind: str = "alra") -> NeuroCoreConfig:
    """~30M active-parameter CPU profile with tied 32K embeddings."""
    cfg = NeuroCoreConfig.small()
    cfg.model_name = "tantra-cpu-dense-32k"
    cfg.vocab.vocab_size = vocab_size
    cfg.vocab.byte_bpe_vocab = vocab_size
    cfg.vocab.text_range_end = vocab_size - 1
    cfg.block.alra.dim = 512
    cfg.block.alra.num_heads = 8
    cfg.block.alra.head_dim = 64
    cfg.block.alra.attention_kind = attention_kind
    cfg.block.sgp.dim = 512
    cfg.block.sgp.expansion = 2
    cfg.block.sgp.implementation = "swiglu"
    cfg.block.num_layers = 8
    cfg.moe.num_experts = 1
    cfg.moe.real_top1 = False
    return cfg


def cpu_top1_moe_config(vocab_size: int = 32768, experts: int = 2, attention_kind: str = "alra") -> NeuroCoreConfig:
    """CPU comparison profile: same active MLP size, but real top-1 experts."""
    cfg = cpu_dense_config(vocab_size=vocab_size, attention_kind=attention_kind)
    cfg.model_name = f"tantra-cpu-top1-moe-{experts}e-32k"
    cfg.moe.num_experts = max(2, experts)
    cfg.moe.top_k = 1
    cfg.moe.real_top1 = True
    return cfg


def cpu_10m_config(vocab_size: int = 32768, attention_kind: str = "alra") -> NeuroCoreConfig:
    """~10M parameter minimum useful CPU profile; intended for distillation."""
    cfg = cpu_dense_config(vocab_size=vocab_size, attention_kind=attention_kind)
    cfg.model_name = "tantra-cpu-10m-32k"
    cfg.block.alra.dim = 224
    cfg.block.alra.num_heads = 7
    cfg.block.alra.head_dim = 32
    cfg.block.sgp.dim = 224
    cfg.block.sgp.expansion = 2
    cfg.block.num_layers = 4
    return cfg


def build_cpu_model(profile: str = "dense", attention_kind: str = "alra", vocab_size: int = 32768) -> NeuroCoreModel:
    if profile == "dense":
        return NeuroCoreModel(cpu_dense_config(vocab_size=vocab_size, attention_kind=attention_kind), use_mtp=False, use_moe=False)
    if profile == "moe2":
        return NeuroCoreModel(cpu_top1_moe_config(vocab_size=vocab_size, experts=2, attention_kind=attention_kind), use_mtp=False, use_moe=True)
    if profile == "micro10":
        return NeuroCoreModel(cpu_10m_config(vocab_size=vocab_size, attention_kind=attention_kind), use_mtp=False, use_moe=False)
    raise ValueError(f"Unknown CPU profile: {profile}")
