"""NP-DNA configuration — single dynamic config that auto-scales.

No more 6 presets. One config with a complexity parameter.
hidden_size, layers, strands, vocab all grow on demand when fill > 95%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class GenomeConfig:
    latent_dim: int | None = None
    rank: int = 32
    max_strands: int | None = None
    encoder_hidden: int = 512

    @property
    def param_estimate(self) -> int:
        L = self.latent_dim or 0
        M = self.max_strands or 0
        encoder = L * self.encoder_hidden + self.encoder_hidden * L
        seeds = M * L
        return encoder + seeds


@dataclass
class StrandConfig:
    hidden_size: int = 128
    state_size: int = 64
    strand_type: str = "ssm"  # "ssm" or "attention"


@dataclass
class MeshConfig:
    num_strands: int = 8
    top_k: int = 2
    balance_weight: float = 0.05
    strand: StrandConfig = field(default_factory=StrandConfig)


@dataclass
class LayerSpec:
    name: str = "main"
    num_strands: int = 8
    top_k: int = 2
    categories: list[tuple[str, int]] | None = None
    dense: bool = False
    strand: StrandConfig = field(default_factory=StrandConfig)

    @property
    def total_strands(self) -> int:
        if self.categories:
            return sum(c[1] for c in self.categories)
        return self.num_strands

    def make_mesh_config(self, hidden_size: int, state_size: int) -> MeshConfig:
        self.strand.hidden_size = hidden_size
        self.strand.state_size = state_size
        return MeshConfig(num_strands=self.total_strands, top_k=self.top_k, strand=self.strand)

    def is_dense(self) -> bool:
        return self.dense

    def is_category(self) -> bool:
        return self.categories is not None


@dataclass
class CortexConfig:
    dim: int = 256
    max_entries: int = 100_000
    top_k: int = 8


@dataclass
class CodecConfig:
    enabled: bool = False
    audio_codec: str | None = None
    image_codec: str | None = None
    video_codec: str | None = None


# Growth thresholds
_GROW_FILL_RATIO = 0.95
_GROW_MULTIPLIER = 1.5


@dataclass
class NpDnaConfig:
    """Single auto-scaling config. Start small, grow on demand.

    The only essential choice is ``complexity`` (0.25–4.0+), which determines
    the base size.  Everything else scales relative to that and auto-grows
    when layer/vocab/strand fill exceeds 95%.
    """

    # ── Core geometry (derived from complexity) ───────────────────────────
    complexity: float = 1.0          # 0.5=tiny .. 4.0=large .. 12.0+=massive
    hidden_size: int = 64            # auto-computed from complexity
    state_size: int = 32             # auto-computed = hidden_size // 2
    num_layers: int = 5              # auto-computed from complexity

    # ── Tokenizer ─────────────────────────────────────────────────────────
    initial_vocab: int = 4096
    max_vocab: int = 256_000

    # ── Strands per layer ─────────────────────────────────────────────────
    strands_per_layer: int = 8       # auto-computed from complexity
    top_k: int = 3                   # auto-computed = max(2, strands_per_layer // 3)

    # ── Mesh specs (auto-built in __post_init__) ──────────────────────────
    mesh_specs: list[LayerSpec] = field(default_factory=list)

    # ── Genome ────────────────────────────────────────────────────────────
    genome: GenomeConfig = field(default_factory=GenomeConfig)

    # ── Mesh defaults (used when mesh_specs is empty) ─────────────────────
    mesh: MeshConfig = field(default_factory=MeshConfig)

    # ── Cortex ────────────────────────────────────────────────────────────
    cortex: CortexConfig = field(default_factory=CortexConfig)

    # ── Embeddings ────────────────────────────────────────────────────────
    tie_embeddings: bool = True

    # ── Dynamic growth state (live counters, not saved in checkpoints) ────
    growth_state: dict = field(default_factory=lambda: {
        "vocab_grows": 0,
        "strand_grows": 0,
        "layer_grows": 0,
        "last_vocab_grow_step": 0,
        "last_strand_grow_step": 0,
        "last_layer_grow_step": 0,
    })

    def __post_init__(self):
        c = self.complexity
        self.hidden_size = max(32, int(64 * c))
        self.state_size = max(16, self.hidden_size // 2)
        self.strands_per_layer = max(4, int(6 + 1.5 * c))
        self.top_k = max(2, self.strands_per_layer // 3)
        self.initial_vocab = max(2048, int(4096 * c))

        if not self.mesh_specs:
            self.mesh_specs = [
                LayerSpec(name="conversation", num_strands=self.strands_per_layer, top_k=self.top_k),
                LayerSpec(name="code", num_strands=self.strands_per_layer, top_k=self.top_k),
                LayerSpec(name="math", num_strands=self.strands_per_layer, top_k=self.top_k),
                LayerSpec(name="science", num_strands=max(4, self.strands_per_layer - 2), top_k=max(2, self.top_k - 1)),
                LayerSpec(name="writing", num_strands=max(4, self.strands_per_layer - 2), top_k=max(2, self.top_k - 1)),
            ]
            self.num_layers = len(self.mesh_specs)

        for spec in self.mesh_specs:
            spec.strand.hidden_size = self.hidden_size
            spec.strand.state_size = self.state_size

        self.mesh.strand.hidden_size = self.hidden_size
        self.mesh.strand.state_size = self.state_size
        self.mesh.num_strands = self.total_strands
        self.mesh.top_k = self.top_k

        self.cortex.dim = self.hidden_size

        if self.genome.latent_dim is None:
            self.genome.latent_dim = min(512, self.hidden_size * 2)
        if self.genome.max_strands is None:
            self.genome.max_strands = self.total_strands * 4  # room to grow 4x

    @property
    def total_strands(self) -> int:
        if self.mesh_specs:
            return sum(spec.total_strands for spec in self.mesh_specs)
        return self.mesh.num_strands * self.num_layers

    # ── Dynamic growth triggers ───────────────────────────────────────────────

    def should_grow_vocab(self, vocab_fill_ratio: float) -> bool:
        return vocab_fill_ratio >= _GROW_FILL_RATIO and self.initial_vocab < self.max_vocab

    def next_vocab_size(self) -> int:
        return min(int(self.initial_vocab * _GROW_MULTIPLIER), self.max_vocab)

    def grow_vocab(self) -> int:
        old = self.initial_vocab
        self.initial_vocab = self.next_vocab_size()
        self.growth_state["vocab_grows"] += 1
        return old

    def should_grow_strands(self, max_usage_ratio: float) -> bool:
        return max_usage_ratio >= _GROW_FILL_RATIO

    def grow_strands(self, layer_idx: int | None = None) -> int:
        """Add strands to one or all layers. Returns new count per layer."""
        add_count = max(1, self.strands_per_layer // 4)
        if layer_idx is not None and layer_idx < len(self.mesh_specs):
            self.mesh_specs[layer_idx].num_strands += add_count
            self.mesh_specs[layer_idx].top_k = max(2, self.mesh_specs[layer_idx].num_strands // 3)
        else:
            for spec in self.mesh_specs:
                spec.num_strands += add_count
                spec.top_k = max(2, spec.num_strands // 3)
        self.strands_per_layer += add_count
        self.top_k = max(2, self.strands_per_layer // 3)
        self.growth_state["strand_grows"] += 1
        return self.strands_per_layer

    def should_grow_layers(self, all_layers_high_usage: bool) -> bool:
        return all_layers_high_usage

    def grow_layers(self) -> list[LayerSpec]:
        """Add a new layer. Returns the new layer specs."""
        new_spec = LayerSpec(
            name=f"layer_{len(self.mesh_specs) + 1}",
            num_strands=self.strands_per_layer,
            top_k=self.top_k,
        )
        self.mesh_specs.append(new_spec)
        self.num_layers = len(self.mesh_specs)
        self.growth_state["layer_grows"] += 1
        return self.mesh_specs


def auto_config(complexity: float = 0.5) -> NpDnaConfig:
    return NpDnaConfig(complexity=complexity)


# ── Backward-compatible config presets (all dynamic now) ─────────────────────
# Each name maps to a complexity level. "seed" = 0.25, "nano" = 0.5,
# "micro" = 1.0, "small" = 2.0, "medium" = 4.0, "large" = 8.0.
_CONFIG_COMPLEXITY: dict[str, float] = {
    "seed": 1.0,    # hidden=64,  8 strands/layer,  5 layers
    "nano": 2.0,    # hidden=128, 9 strands/layer,  5 layers
    "micro": 4.0,   # hidden=256, 12 strands/layer, 5 layers
    "small": 6.0,   # hidden=384, 15 strands/layer, 5 layers
    "medium": 8.0,  # hidden=512, 18 strands/layer, 5 layers
    "large": 12.0,  # hidden=768, 24 strands/layer, 5 layers
}

CONFIGS: dict[str, NpDnaConfig] = {
    name: NpDnaConfig(complexity=c)
    for name, c in _CONFIG_COMPLEXITY.items()
}

PREFERRED_CONFIG_NAMES = tuple(_CONFIG_COMPLEXITY.keys())
