"""NP-DNA: NeuroPlastic DNA Network — Tantra LLM core package.

Structure:
    config.py    — NpDnaConfig, LayerSpec, GenomeConfig, CortexConfig
    genome.py    — Genome: low-rank weight generator
    mesh.py       — NeuralMesh, AttentionStrand, sparse routing
    cortex.py     — MemoryCortex: external vector knowledge store
    model.py      — NpDnaModel, NpDnaCore (training + inference)
    tokenizer.py  — AtulyaTokenizer: BPE with dynamic growth
    generation.py — GenerationMixin: sampling, streaming, repetition penalty
    train.py      — Training loop, curriculum, dataset, checkpointing
    cli.py        — Command-line chat + info interface
    brain.py      — Everything else in one place:
                     PlasticityEngine, tag_text, NpDnaAgent,
                     build_multimodal_prompt, encode_image_clip,
                     benchmark_checkpoint, quantize_model_for_cpu
"""

from .config import CONFIGS, PREFERRED_CONFIG_NAMES, NpDnaConfig, auto_config
from .cortex import CortexAutoStore, MemoryCortex
from .genome import Genome
from .mesh import CategoryMesh, NeuralMesh, Strand
from .model import NpDnaCore, NpDnaModel
from .tokenizer import AtulyaTokenizer

from .brain import (
    # Plasticity
    PlasticityAutoScaler,
    PlasticityEngine,
    PlasticityMetrics,
    # Classifier
    NpDnaTopicClassifier,
    tag_text,
    # Agent
    NpDnaAgent,
    # Multimodal
    build_multimodal_prompt,
    encode_image_clip,
    describe_image,
    describe_audio,
    # Optimise
    quantize_model_for_cpu,
    apply_torch_compile,
    model_size_mb,
    freeze_for_partial_training,
    # Benchmark
    benchmark_checkpoint,
    write_benchmark,
    # Codec registry
    FrozenCodecRef,
    FrozenCodecRegistry,
)

__all__ = [
    "CONFIGS", "PREFERRED_CONFIG_NAMES", "NpDnaConfig", "auto_config",
    "Genome", "Strand", "NeuralMesh", "CategoryMesh",
    "MemoryCortex", "CortexAutoStore",
    "NpDnaModel", "NpDnaCore",
    "AtulyaTokenizer",
    "PlasticityEngine", "PlasticityMetrics", "PlasticityAutoScaler",
    "NpDnaTopicClassifier", "tag_text",
    "NpDnaAgent",
    "build_multimodal_prompt", "encode_image_clip", "describe_image", "describe_audio",
    "quantize_model_for_cpu", "apply_torch_compile", "model_size_mb",
    "freeze_for_partial_training",
    "benchmark_checkpoint", "write_benchmark",
    "FrozenCodecRef", "FrozenCodecRegistry",
]
