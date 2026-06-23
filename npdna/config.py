"""NP-DNA configuration.

The public release exposes one named configuration, ``seed``. It starts small
and grows vocabulary, strands, and layers at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenomeConfig:
    latent_dim: int | None = 128
    rank: int = 8
    max_strands: int | None = None
    encoder_hidden: int = 128

    @property
    def param_estimate(self) -> int:
        latent = self.latent_dim or 0
        max_strands = self.max_strands or 0
        encoder = latent * self.encoder_hidden + self.encoder_hidden * latent
        seeds = max_strands * latent
        return encoder + seeds


@dataclass
class StrandConfig:
    hidden_size: int = 128
    state_size: int = 64
    strand_type: str = "ssm"
    # SwiGLU feed-forward layer inside every SSM Strand (enabled by default for new models)
    use_swiglu: bool = True
    # ffn_expansion × hidden_size = inner FFN dimension (standard LLM ratio)
    ffn_expansion: float = 4.0
    # GQA: num_kv_heads < num_heads to speed up Attention strands on CPU.
    # 0 = auto (num_heads // 4, minimum 1)
    num_kv_heads: int = 0


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
            return sum(count for _, count in self.categories)
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
    min_relevance: float = 0.3


@dataclass
class CodecConfig:
    enabled: bool = False
    audio_codec: str | None = None
    image_codec: str | None = None
    video_codec: str | None = None


GROW_FILL_RATIO = 0.95
MAX_VOCAB_GROWTH_PER_STAGE = 10000


@dataclass
class NpDnaConfig:
    """Single auto-scaling config. Start small, grow on demand."""

    complexity: float = 1.0
    hidden_size: int = 256
    state_size: int = 256
    num_layers: int = 15

    # Adaptive Compute Depth (Early Exit)
    adaptive_depth: bool = True
    exit_threshold: float = 0.85

    initial_vocab: int = 4096
    max_vocab: int = 256_000

    strands_per_layer: int = 8
    top_k: int = 3

    mesh_specs: list[LayerSpec] = field(default_factory=list)
    genome: GenomeConfig = field(default_factory=GenomeConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    cortex: CortexConfig = field(default_factory=CortexConfig)
    tie_embeddings: bool = True

    # List of lists, e.g. [[0, 1, 2], [3, 4]] means layers 0,1,2 share a genome, layers 3,4 share another.
    # If None, all layers share a single genome (default).
    weight_sharing_groups: list[list[int]] | None = None

    growth_state: dict = field(default_factory=lambda: {
        "vocab_grows": 0,
        "strand_grows": 0,
        "layer_grows": 0,
        "last_vocab_grow_step": 0,
        "last_strand_grow_step": 0,
        "last_layer_grow_step": 0,
    })

    def __post_init__(self) -> None:
        complexity = self.complexity
        # Base scale: complexity 1.0 -> hidden=128, layers=15.
        self.hidden_size = max(64, int(128 * complexity))
        self.state_size = self.hidden_size
        self.strands_per_layer = max(4, int(8 + 2 * complexity))
        self.top_k = max(2, self.strands_per_layer // 3)
        self.initial_vocab = max(2048, int(4096 * complexity))

        if not self.mesh_specs:
            self.mesh_specs = [
                LayerSpec(name="conversation", num_strands=self.strands_per_layer, top_k=self.top_k),
                LayerSpec(name="code", num_strands=self.strands_per_layer, top_k=self.top_k),
                LayerSpec(name="math", num_strands=self.strands_per_layer, top_k=self.top_k),
                LayerSpec(
                    name="science",
                    num_strands=max(4, self.strands_per_layer - 2),
                    top_k=max(2, self.top_k - 1),
                ),
                LayerSpec(
                    name="writing",
                    num_strands=max(4, self.strands_per_layer - 2),
                    top_k=max(2, self.top_k - 1),
                ),
            ]

            # Fill remaining layers with named training domains up to num_layers.
            extra_layer_names = [
                "instruction",
                "experts",
                "reasoning",
                "factual",
                "general",
                "chat",
                "emotion",
                "spatial",
                "action",
                "synthesis",
            ]
            extra_i = 0
            while len(self.mesh_specs) < self.num_layers:
                name = (
                    extra_layer_names[extra_i]
                    if extra_i < len(extra_layer_names)
                    else f"layer_{len(self.mesh_specs)}"
                )
                self.mesh_specs.append(
                    LayerSpec(
                        name=name,
                        num_strands=self.strands_per_layer,
                        top_k=self.top_k,
                    )
                )
                extra_i += 1

            self.num_layers = len(self.mesh_specs)

        for spec in self.mesh_specs:
            spec.strand.hidden_size = self.hidden_size
            spec.strand.state_size = self.state_size
            # Propagate new arch fields to every spec strand
            if spec.strand.use_swiglu is True:  # only override if still at default
                pass  # keep as-is (default True)


        self.mesh.strand.hidden_size = self.hidden_size
        self.mesh.strand.state_size = self.state_size
        self.mesh.num_strands = self.total_strands
        self.mesh.top_k = self.top_k
        self.cortex.dim = self.hidden_size

        if self.genome.latent_dim is None:
            self.genome.latent_dim = min(128, self.hidden_size)
        if self.genome.max_strands is None:
            self.genome.max_strands = self.total_strands * 4

    @property
    def total_strands(self) -> int:
        if self.mesh_specs:
            return sum(spec.total_strands for spec in self.mesh_specs)
        return self.mesh.num_strands * self.num_layers

    def should_grow_vocab(self, vocab_fill_ratio: float) -> bool:
        return vocab_fill_ratio >= GROW_FILL_RATIO and self.initial_vocab < self.max_vocab

    def next_vocab_size(self) -> int:
        return min(self.initial_vocab + MAX_VOCAB_GROWTH_PER_STAGE, self.max_vocab)

    def grow_vocab(self) -> int:
        old = self.initial_vocab
        self.initial_vocab = self.next_vocab_size()
        self.growth_state["vocab_grows"] += 1
        return old

    def should_grow_strands(self, max_usage_ratio: float) -> bool:
        return max_usage_ratio >= GROW_FILL_RATIO

    def grow_strands(self, layer_idx: int | None = None) -> int:
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
        new_spec = LayerSpec(
            name=f"layer_{len(self.mesh_specs) + 1}",
            num_strands=self.strands_per_layer,
            top_k=self.top_k,
        )
        self.mesh_specs.append(new_spec)
        self.num_layers = len(self.mesh_specs)
        self.growth_state["layer_grows"] += 1
        return self.mesh_specs


def auto_config(complexity: float = 1.0) -> NpDnaConfig:
    return NpDnaConfig(complexity=complexity)


DEFAULT_CONFIG_NAME = "seed"
DEFAULT_COMPLEXITY = 1.0

CONFIGS: dict[str, NpDnaConfig] = {
    DEFAULT_CONFIG_NAME: NpDnaConfig(complexity=DEFAULT_COMPLEXITY)
}

PREFERRED_CONFIG_NAMES = (DEFAULT_CONFIG_NAME,)
