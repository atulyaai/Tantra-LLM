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
    # vocab_size / byte_bpe_vocab = TEXT vocab size (the real trained BPE
    # tokenizer size). Kept as plain mutable fields because ~15 call sites
    # across main.py / webui/server.py / model.py assign to these directly
    # (e.g. `cfg.vocab.vocab_size = 65536`) and a read-only property would
    # break all of them.
    #
    # BUG FIXED (2026-09): audio_range_start/end and image_range_start/end
    # and video_range_start/end were ALL hardcoded to the single point 32768
    # (start == end == 32768), i.e. zero-width ranges that silently collide
    # on one token id. That's why the README correctly lists multimodal
    # fusion as "implemented, current-checkpoint result unverified" --
    # there was no real id space reserved for audio/image/video tokens.
    # Call `recompute_ranges()` after changing vocab_size or the codebook
    # sizes so the four ranges stay non-overlapping and correctly sized.
    vocab_size: int = 24609  # Real BPE tokenizer size
    byte_bpe_vocab: int = 24609  # Real BPE tokenizer size
    audio_codebook_size: int = 8192
    image_codebook_size: int = 8192
    video_codebook_size: int = 111583  # 152576 - 24609 - 8192 - 8192 = 111583
    text_range_start: int = 0
    text_range_end: int = 24608
    audio_range_start: int = 24609
    audio_range_end: int = 68191
    image_range_start: int = 68192
    image_range_end: int = 76383
    video_range_start: int = 76384
    video_range_end: int = 152575

    special_tokens: dict = field(default_factory=lambda: {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<bos>": 4,
        "<eos>": 5,
        "<|system|>": 6,
        "<|user|>": 7,
        "<|assistant|>": 8,
        "<audio>": 9,
        "<image>": 10,
        "<video>": 11,
        "<tool_call>": 12,
        "<tool_result>": 13,
    })
    megabyte_patch_size: int = 8  # bytes per megabyte patch

    def recompute_ranges(self) -> None:
        """Re-derive non-overlapping audio/image/video ranges from vocab_size
        and the codebook sizes. Called automatically (see __post_init__ /
        __setattr__ below) whenever vocab_size, byte_bpe_vocab, or any
        *_codebook_size field changes, so callers no longer need to remember
        to call this by hand."""
        self.text_range_start = 0
        self.text_range_end = self.vocab_size - 1
        self.audio_range_start = self.vocab_size
        self.audio_range_end = self.audio_range_start + self.audio_codebook_size - 1
        self.image_range_start = self.audio_range_end + 1
        self.image_range_end = self.image_range_start + self.image_codebook_size - 1
        self.video_range_start = self.image_range_end + 1
        self.video_range_end = self.video_range_start + self.video_codebook_size - 1

    # ── 0.926B Model Configuration ──────────────
    model_0926b_layers = 20
    model_0926b_dim = 1472
    model_0926b_heads = 23
    model_0926b_head_dim = 64
    model_0926b_total_output_rows = 152576
    model_0926b_text_logits_rows = 60000

    # ── Code & Markdown Support ─────────────────
    code_special_tokens = {
        "<code>": 12, "</code>": 13, "<python>": 14, "</python>": 15,
        "<javascript>": 16, "</javascript>": 17,
        "<markdown>": 18, "</markdown>": 19,
    }

    # ── FSDP Configuration ──────────────────────
    fsdp_strategy = "FULL_SHARD"
    fsdp_backward_prefetch = "BACKWARD_PRE"
    fsdp_forward_prefetch = "FORWARD_PRE"
    fsdp_limit_all_gathers = True
    fsdp_use_orig_params = False

    # ── Precision ───────────────────────────────
    precision = "fp16"
    grad_scaler = True

    # ── Range Trigger Fields ────────────────────
    _RANGE_TRIGGER_FIELDS = (
        "vocab_size", "byte_bpe_vocab",
        "audio_codebook_size", "image_codebook_size", "video_codebook_size",
    )
    
    def recompute_ranges(self) -> None:
        """Re-derive non-overlapping audio/image/video ranges from vocab_size
        and the codebook sizes. Called automatically (see __post_init__ /
        __setattr__ below) whenever vocab_size, byte_bpe_vocab, or any
        *_codebook_size field changes, so callers no longer need to remember
        to call this by hand."""
        self.text_range_start = 0
        self.text_range_end = self.vocab_size - 1
        self.audio_range_start = self.vocab_size
        self.audio_range_end = self.audio_range_start + self.audio_codebook_size - 1
        self.image_range_start = self.audio_range_end + 1
        self.image_range_end = self.image_range_start + self.image_codebook_size - 1
        self.video_range_start = self.image_range_end + 1
        self.video_range_end = self.video_range_start + self.video_codebook_size - 1
        object.__setattr__(self, "_ranges_initialized", True)

    def __setattr__(self, name, value) -> None:
        object.__setattr__(self, name, value)
        if name in VocabConfig._RANGE_TRIGGER_FIELDS and getattr(self, "_ranges_initialized", False):
            # A range field is about to be overwritten by recompute_ranges()
            # itself (e.g. text_range_end) -- those names aren't in
            # _RANGE_TRIGGER_FIELDS, so this doesn't recurse.
            self.recompute_ranges()

    @property
    def total_embedding_size(self) -> int:
        """Real number of rows the embedding table needs if audio/image/video
        codebooks share the same embedding matrix as text (video_range_end + 1).
        model.py currently only allocates `vocab_size` (text-only) rows --
        see NOTE in NeuroCoreModel.__init__."""
        return self.video_range_end + 1

    @property
    def byte_vocab_size(self) -> int:
        return self.byte_bpe_vocab


@dataclass
class ALRAConfig:
    """Adaptive Linear Resonance Attention config."""
    dim: int = 512
    num_heads: int = 8
    head_dim: int = 64          # dim // num_heads
    kernel_type: str = "elu1"   # "elu1" | "relu" | "learned"
    dropout: float = 0.0
    use_forget_gate: bool = True
    attention_kind: str = "alra"  # "alra" | "causal"


@dataclass
class SGPConfig:
    """Sparse Gated Projection (FFN replacement) config."""
    dim: int = 512
    expansion: int = 4          # hidden = dim * expansion
    sparsity: float = 0.50      # fraction of neurons active (50% active for rich gradient flow)
    activation: str = "gelu"    # "gelu" | "silu" | "relu"
    implementation: str = "sparse"  # "sparse" | "swiglu"


@dataclass
class NeuroCoreBlockConfig:
    alra: ALRAConfig = field(default_factory=ALRAConfig)
    sgp: SGPConfig = field(default_factory=SGPConfig)
    num_layers: int = 8
    pre_norm: bool = True        # pre-normalization (more stable)


@dataclass
class MoEConfig:
    num_experts: int = 10
    top_k: int = 1               # Top-1 routing = most brain-like
    router_dim: int = 512        # Router network hidden dim
    router_layers: int = 2
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
    enabled: bool = False  # DISABLED for training - ternary weights kill learning capacity
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
    latent_cot: bool = False        # DISABLED for initial training - causes instability
    auto_growth: bool = False       # DISABLED for initial training - destabilizes learning
    gradient_checkpointing: bool = False


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
        """1B param config.
        NOTE: there are two ~1B-param profiles in this file (`medium()` and
        `billion()`) that disagree on vocab_size and modality ranges.
        `billion()` is the maintained/documented one (has the param-count
        math worked out and the vocab fix). Prefer `billion()`; `medium()`
        is kept only for backward compatibility with anything that already
        calls it.
        """
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

    @classmethod
    def billion(cls) -> "NeuroCoreConfig":
        """1B param knowledge-dense config — Hindi-English bilingual, ALRA attention, auto-growth up to 1B.

        Architecture math:
          Embedding:    65536 × 2048 = 134M (13% of model — healthy budget)
          Per layer:    ALRA(2048,16h,128hd) + SwiGLU(2048→8192→2048) ≈ 33.5M
          24 layers:    24 × 33.5M = 804M
          Norms+heads:  ~20M
          Total:        ~958M ≈ 1B params

        Vocab: 60,000 real trained BPE tokens (see Model/vocab.json) supports
        Hindi (2-3 tok/word) + English (1-2 tok/word) + Code efficiently.
        FIXED (2026-09): this used to hardcode vocab_size=65536, a number that
        was never actually trained (the real corpus saturates BPE merges at
        ~59.4K -- pushing past that just adds frequency-1 junk tokens). It
        also hand-typed audio/image/video ranges that didn't match
        audio_codebook_size/image_codebook_size/video_codebook_size (8192
        each) -- e.g. audio was only 3,999 ids wide, not 8,192. Both are
        fixed below by using the real tokenizer size and recompute_ranges().
        """
        cfg = cls(model_name="tantra-1b")
        cfg.vocab.vocab_size = cfg.vocab.byte_bpe_vocab = 24609
        cfg.vocab.audio_codebook_size = 0
        cfg.vocab.image_codebook_size = 0
        cfg.vocab.video_codebook_size = 0
        cfg.vocab.recompute_ranges()

        cfg.block.alra.dim = 1536
        cfg.block.alra.num_heads = 24
        cfg.block.alra.head_dim = 64
        cfg.block.alra.attention_kind = "alra"

        cfg.block.sgp.dim = 1536
        cfg.block.sgp.expansion = 4          # 1536 → 6144 → 1536
        cfg.block.sgp.implementation = "swiglu"
        cfg.block.num_layers = 20

        cfg.moe.num_experts = 1              # Start dense; switch to MoE later
        cfg.moe.real_top1 = False

        cfg.bitnet.enabled = False           # Train in BF16; quantize after convergence
        cfg.bitnet.use_shadow_weights = True

        cfg.training.batch_size = 8
        cfg.training.grad_accumulation_steps = 16
        cfg.training.learning_rate = 3e-4
        cfg.training.warmup_steps = 2000
        cfg.training.max_steps = 1_000_000
        cfg.training.dtype = "bfloat16"
        cfg.training.auto_growth = True      # Grow from 24 → 32 layers as loss plateaus

        cfg.inference.max_seq_len = 131072
        return cfg
