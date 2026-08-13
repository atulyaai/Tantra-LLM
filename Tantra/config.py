"""
core/config.py — Single source of truth for all NeuroCore configuration.
Every module reads from this file. No hardcoded values anywhere else.
"""
from dataclasses import dataclass, field
from typing import Optional
import json
import os


@dataclass
class VocabConfig:
    vocab_size: int = 32000
    byte_bpe_vocab: int = 32000
    audio_codebook_size: int = 8192
    image_codebook_size: int = 8192
    video_codebook_size: int = 8192
    # Remapping ranges inside 32K unified space
    text_range_start: int = 0
    text_range_end: int = 24999
    audio_range_start: int = 25000
    audio_range_end: int = 27999
    image_range_start: int = 28000
    image_range_end: int = 30999
    video_range_start: int = 31000
    video_range_end: int = 31999
    special_tokens: dict = field(default_factory=lambda: {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "<audio>": 4,
        "<image>": 5,
        "<video>": 6,
    })
    megabyte_patch_size: int = 8  # bytes per megabyte patch


@dataclass
class ALRAConfig:
    """Adaptive Linear Resonance Attention config."""
    dim: int = 4096
    num_heads: int = 32
    head_dim: int = 128         # dim // num_heads
    kernel_type: str = "elu1"   # "elu1" | "relu" | "learned"
    dropout: float = 0.0
    use_forget_gate: bool = True
    attention_kind: str = "alra"  # "alra" | "causal"


@dataclass
class SGPConfig:
    """Sparse Gated Projection (FFN replacement) config."""
    dim: int = 4096
    expansion: int = 4          # hidden = dim * expansion
    sparsity: float = 0.10      # fraction of neurons active (brain-like)
    activation: str = "gelu"    # "gelu" | "silu" | "relu"
    implementation: str = "sparse"  # "sparse" | "swiglu"


@dataclass
class NeuroCoreBlockConfig:
    alra: ALRAConfig = field(default_factory=ALRAConfig)
    sgp: SGPConfig = field(default_factory=SGPConfig)
    num_layers: int = 32
    pre_norm: bool = True        # pre-normalization (more stable)


@dataclass
class MoEConfig:
    num_experts: int = 500
    top_k: int = 1               # Top-1 routing = most brain-like
    router_dim: int = 2048       # Router network hidden dim
    router_layers: int = 4
    load_balance_coeff: float = 0.01  # Weight of load balancing loss
    expert_cache_size: int = 8   # LRU cache: experts kept in RAM
    expert_dir: str = "Experts"  # Directory containing .dna expert files
    real_top1: bool = False       # True only for the explicit real-MoE profile


@dataclass
class AdapterConfig:
    """Per-domain specialist-layer configuration for CPU deployment."""
    mode: str = "layer"          # "layer" = one dedicated NeuroCoreBlock per category
    rank: int = 32               # bottleneck width (used only in "bottleneck" mode)
    clone_layer_index: int = -1  # base block index to clone specialist layer weights from (-1 = last)
    default_categories: int = 8  # how many routeable categories to seed



@dataclass
class BitNetConfig:
    enabled: bool = True
    quantize_mode: str = "ternary"  # "ternary" | "binary"
    scale_type: str = "absmax"      # "absmax" | "rms"
    pack_bits: int = 2              # bits per weight for packing
    use_shadow_weights: bool = True  # Keep FP32 during training


@dataclass
class CompressionConfig:
    method: str = "dna"             # "dna" | "zstd" | "none"
    zstd_level: int = 3             # Level 3 for fast dev/test; level 19 for production release
    zstd_dict_size: int = 131072    # 128KB ZSTD dictionary
    residual_window: int = 64       # AI predictor context window
    residual_hidden: int = 256      # AI predictor hidden size
    dna_parity_interval: int = 8    # parity check every N symbols
    target_compression_ratio: float = 12.0  # target: 12x


@dataclass
class InferenceConfig:
    max_seq_len: int = 131072       # 128K context window
    temperature: float = 0.65      # was 0.8 — sharper, less rambling text
    top_p: float = 0.90            # was 0.95 — narrower nucleus, better focus
    top_k: int = 30                # was 40 — fewer token candidates
    repetition_penalty: float = 1.25  # was 1.1 — stronger anti-repeat
    batch_size: int = 1             # overridden by hardware auto-detect
    prefetch_experts: int = 2       # pre-load next N likely experts



@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 1_000_000
    checkpoint_every: int = 5000
    eval_every: int = 1000
    batch_size: int = 4             # per-device, overridden by hardware
    grad_accumulation_steps: int = 8
    dtype: str = "float32"          # "float32" | "bfloat16"


@dataclass
class NeuroCoreConfig:
    """Master configuration — all modules read from this."""
    model_name: str = "tantra"
    vocab: VocabConfig = field(default_factory=VocabConfig)
    block: NeuroCoreBlockConfig = field(default_factory=NeuroCoreBlockConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    bitnet: BitNetConfig = field(default_factory=BitNetConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Paths
    checkpoint_dir: str = "checkpoints"
    expert_dir: str = "Experts"
    vocab_dir: str = "vocab_data"
    reports_dir: str = "reports"
    log_dir: str = "logs"

    def save(self, path: str) -> None:
        """Save config to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)

    def _to_dict(self) -> dict:
        """Recursively convert dataclasses to dict."""
        import dataclasses
        def convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        return convert(self)

    @classmethod
    def load(cls, path: str) -> "NeuroCoreConfig":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "NeuroCoreConfig":
        """Reconstruct config from dict."""
        cfg = cls()
        if "model_name" in data:
            cfg.model_name = data["model_name"]
        for key in ["checkpoint_dir", "expert_dir", "vocab_dir", "reports_dir", "log_dir"]:
            if key in data:
                setattr(cfg, key, data[key])

        if "vocab" in data and isinstance(data["vocab"], dict):
            for k, v in data["vocab"].items():
                if hasattr(cfg.vocab, k): setattr(cfg.vocab, k, v)
        if "block" in data and isinstance(data["block"], dict):
            if "alra" in data["block"] and isinstance(data["block"]["alra"], dict):
                for k, v in data["block"]["alra"].items():
                    if hasattr(cfg.block.alra, k): setattr(cfg.block.alra, k, v)
            if "sgp" in data["block"] and isinstance(data["block"]["sgp"], dict):
                for k, v in data["block"]["sgp"].items():
                    if hasattr(cfg.block.sgp, k): setattr(cfg.block.sgp, k, v)
            if "num_layers" in data["block"]: cfg.block.num_layers = data["block"]["num_layers"]
            if "pre_norm" in data["block"]: cfg.block.pre_norm = data["block"]["pre_norm"]
        if "moe" in data and isinstance(data["moe"], dict):
            for k, v in data["moe"].items():
                if hasattr(cfg.moe, k): setattr(cfg.moe, k, v)
        if "bitnet" in data and isinstance(data["bitnet"], dict):
            for k, v in data["bitnet"].items():
                if hasattr(cfg.bitnet, k): setattr(cfg.bitnet, k, v)
        if "compression" in data and isinstance(data["compression"], dict):
            for k, v in data["compression"].items():
                if hasattr(cfg.compression, k): setattr(cfg.compression, k, v)
        if "inference" in data and isinstance(data["inference"], dict):
            for k, v in data["inference"].items():
                if hasattr(cfg.inference, k): setattr(cfg.inference, k, v)
        if "training" in data and isinstance(data["training"], dict):
            for k, v in data["training"].items():
                if hasattr(cfg.training, k): setattr(cfg.training, k, v)
        return cfg

    @classmethod
    def small(cls) -> "NeuroCoreConfig":
        """NeuroCore small config — GPT-2 equivalent architecture (12L, 768 dim, 12H).

        Note: Actual param count is ~178M due to NeuroCore additions (MTP head,
        DSN, ALRA gate, LatentCoT) on top of GPT-2's 124M baseline architecture.
        """
        cfg = cls(model_name="neurocore-178m")
        cfg.block.alra.dim = 768
        cfg.block.alra.num_heads = 12
        cfg.block.alra.head_dim = 64
        cfg.block.sgp.dim = 768
        cfg.block.num_layers = 12
        cfg.moe.num_experts = 10
        cfg.moe.expert_cache_size = 8
        return cfg

    @classmethod
    def medium(cls) -> "NeuroCoreConfig":
        """1B param config."""
        cfg = cls(model_name="neurocore-1b")
        cfg.block.alra.dim = 2048
        cfg.block.alra.num_heads = 16
        cfg.block.alra.head_dim = 128
        cfg.block.sgp.dim = 2048
        cfg.block.num_layers = 24
        cfg.moe.num_experts = 64
        cfg.moe.expert_cache_size = 8
        return cfg

    @classmethod
    def large(cls) -> "NeuroCoreConfig":
        """7B param config."""
        cfg = cls(model_name="neurocore-7b")
        cfg.block.alra.dim = 4096
        cfg.block.alra.num_heads = 32
        cfg.block.alra.head_dim = 128
        cfg.block.sgp.dim = 4096
        cfg.block.num_layers = 32
        cfg.moe.num_experts = 128
        cfg.moe.expert_cache_size = 12
        return cfg

    @classmethod
    def trillion(cls) -> "NeuroCoreConfig":
        """1T param config — 500 experts × 2B each."""
        cfg = cls(model_name="neurocore-1t")
        cfg.block.alra.dim = 8192
        cfg.block.alra.num_heads = 64
        cfg.block.alra.head_dim = 128
        cfg.block.sgp.dim = 8192
        cfg.block.num_layers = 80
        cfg.moe.num_experts = 500
        cfg.moe.expert_cache_size = 4  # tight RAM budget
        return cfg
