"""
Tantra/config.py — All model settings in one place.

Presets:
  NeuroCoreConfig.tiny()      ~1M params, for tests
  NeuroCoreConfig.small()     default: ~70M params (dim 512 x 8 layers), trains on a laptop CPU
  NeuroCoreConfig.billion()   ~1B params (dim 1536 x 24 layers), needs GPUs to train

The vocabulary (64k) is the same for every size, so a small model can be grown
into a bigger one (more layers) without retraining or changing the tokenizer.

Class names are kept stable because checkpoints store the config object.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field


SPECIAL_TOKENS = {
    "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "<bos>": 4, "<eos>": 5,
    "<|system|>": 6, "<|user|>": 7, "<|assistant|>": 8,
    "<audio>": 9, "<image>": 10, "<video>": 11,
    "<tool_call>": 12, "<tool_result>": 13,
    "<thought>": 14, "</thought>": 15, "<code>": 16, "</code>": 17,
    "।": 18, "॥": 19, "ॐ": 20,
}
EOS_ID = 2  # "</s>" — end of every assistant reply / document


@dataclass
class VocabConfig:
    vocab_size: int = 64000          # fixed forever: Model/tokenizer.json (64k Hindi+English)
    byte_bpe_vocab: int = 64000
    # Reserved for future speech/vision tokens. 0 = text-only model.
    audio_codebook_size: int = 0
    image_codebook_size: int = 0
    video_codebook_size: int = 0
    special_tokens: dict = field(default_factory=lambda: dict(SPECIAL_TOKENS))

    def recompute_ranges(self) -> None:
        """Kept for old call sites; ranges are derived on demand."""

    @property
    def total_embedding_size(self) -> int:
        return (self.vocab_size + (self.audio_codebook_size or 0)
                + (self.image_codebook_size or 0) + (self.video_codebook_size or 0))


@dataclass
class ALRAConfig:
    """Attention settings. ALRA = linear-time recurrent attention (CPU friendly)."""
    dim: int = 512
    num_heads: int = 8
    head_dim: int = 64
    kernel_type: str = "elu1"
    dropout: float = 0.0
    use_forget_gate: bool = True
    attention_kind: str = "alra"     # "alra" | "causal"
    # Every Nth layer uses exact sliding-window softmax attention (good recall).
    # 0 = off (old checkpoints).
    local_attn_every: int = 0
    local_window: int = 512


@dataclass
class SGPConfig:
    """Feed-forward (MLP) settings."""
    dim: int = 512
    expansion: int = 4
    sparsity: float = 0.50
    activation: str = "gelu"
    implementation: str = "swiglu"   # "swiglu" (dense, fast on CPU) | "sparse"


@dataclass
class NeuroCoreBlockConfig:
    alra: ALRAConfig = field(default_factory=ALRAConfig)
    sgp: SGPConfig = field(default_factory=SGPConfig)
    num_layers: int = 8
    pre_norm: bool = True


@dataclass
class MoEConfig:
    """Optional Top-1 mixture-of-experts MLP (off by default)."""
    num_experts: int = 1
    top_k: int = 1
    router_dim: int = 512
    router_layers: int = 2
    load_balance_coeff: float = 0.01
    expert_cache_size: int = 8
    expert_dir: str = "Experts"
    real_top1: bool = False


@dataclass
class AdapterConfig:
    """Category layers: one extra specialist block per category (e.g. code, math)."""
    mode: str = "layer"
    rank: int = 32
    clone_layer_index: int = -1
    default_categories: int = 8


@dataclass
class BitNetConfig:
    """Ternary {-1,0,+1} weights. Train in float first; quantize after convergence."""
    enabled: bool = False
    quantize_mode: str = "ternary"
    scale_type: str = "absmax"
    pack_bits: int = 2
    use_shadow_weights: bool = True


@dataclass
class CompressionConfig:
    """Unused. Kept only so older checkpoints can still be opened."""
    method: str = "none"
    zstd_level: int = 3
    zstd_dict_size: int = 131072
    residual_window: int = 64
    residual_hidden: int = 256
    dna_parity_interval: int = 8
    target_compression_ratio: float = 1.0


@dataclass
class InferenceConfig:
    max_seq_len: int = 4096
    temperature: float = 0.65
    top_p: float = 0.90
    top_k: int = 30
    repetition_penalty: float = 1.15
    batch_size: int = 1
    prefetch_experts: int = 2


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    max_steps: int = 20000
    checkpoint_every: int = 1000
    eval_every: int = 250
    batch_size: int = 4
    grad_accumulation_steps: int = 8
    dtype: str = "float32"
    latent_cot: bool = False
    auto_growth: bool = False
    gradient_checkpointing: bool = False


@dataclass
class NeuroCoreConfig:
    model_name: str = "tantra"
    vocab: VocabConfig = field(default_factory=VocabConfig)
    block: NeuroCoreBlockConfig = field(default_factory=NeuroCoreBlockConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    bitnet: BitNetConfig = field(default_factory=BitNetConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # ── helpers ──
    def set_shape(self, dim: int, layers: int, heads: int, vocab_size: int | None = None) -> "NeuroCoreConfig":
        self.block.alra.dim = self.block.sgp.dim = dim
        self.block.alra.num_heads = heads
        self.block.alra.head_dim = dim // heads
        self.block.num_layers = layers
        if vocab_size:
            self.vocab.vocab_size = self.vocab.byte_bpe_vocab = vocab_size
        return self

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "NeuroCoreConfig":
        with open(path, encoding="utf-8") as f:
            return cls._from_dict(json.load(f))

    @classmethod
    def _from_dict(cls, data: dict) -> "NeuroCoreConfig":
        cfg = cls()

        def fill(obj, d):
            for k, v in (d or {}).items():
                cur = getattr(obj, k, None)
                if dataclasses.is_dataclass(cur) and isinstance(v, dict):
                    fill(cur, v)
                elif hasattr(obj, k):
                    setattr(obj, k, v)
        fill(cfg, data)
        return cfg

    # ── presets ──
    @classmethod
    def tiny(cls) -> "NeuroCoreConfig":
        cfg = cls(model_name="tantra-tiny").set_shape(64, 4, 4, vocab_size=512)
        cfg.block.alra.local_attn_every, cfg.block.alra.local_window = 2, 16
        return cfg

    @classmethod
    def small(cls, vocab_size: int = 64000) -> "NeuroCoreConfig":
        """Default: dim 512, 8 layers, 64k vocab, every 4th layer local softmax. ~70M params."""
        cfg = cls(model_name="tantra-small").set_shape(512, 8, 8, vocab_size=vocab_size)
        cfg.block.alra.local_attn_every, cfg.block.alra.local_window = 4, 512
        return cfg

    @classmethod
    def billion(cls, vocab_size: int = 64000) -> "NeuroCoreConfig":
        """~1.0B params: dim 1536 x 24 layers + 64k vocab. Train on GPU, run quantized on CPU."""
        cfg = cls(model_name="tantra-1b").set_shape(1536, 24, 24, vocab_size=vocab_size)
        cfg.block.alra.local_attn_every, cfg.block.alra.local_window = 4, 1024
        cfg.training.gradient_checkpointing = True
        cfg.training.dtype = "bfloat16"
        return cfg
