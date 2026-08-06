"""NP-DNA: NeuroPlastic DNA Network — Tantra LLM core package.

Flat single-level layout (no sub-packages):
    architecture.py    NpDnaConfig, LayerSpec, MeshConfig, Strand, NeuralMesh
    schema.py          Shared request/response/settings types, identity, middleware/memory protocols
    model.py           NpDnaModel, NpDnaCore, GenerationMixin, Genome, LoRA
    tokenizer.py       AtulyaTokenizer: BPE with dynamic growth
    train.py           Training loop, curriculum, checkpointing, DynamicGrowthController, optimization
    serving.py         CLI, Gradio Studio, and FastAPI API server
    brain.py           PlasticityEngine, tag_text, NpDnaAgent, multimodal prompts, CPU optimization
    fusion.py          MultimodalDataset (projectors are built-in in model.py)
    sensory.py         Vision/Audio/TTS encoders + VisionOrgan, VoiceOrgan, SentimentCore
    inference.py       UnifiedInferenceHub + adapters + middleware
    cognition.py       ComputeRouter, DynamicContext, EventBus, MemoryStore, FastResponseMemory, MemoryCortex
"""

from .architecture import CONFIGS, PREFERRED_CONFIG_NAMES, NpDnaConfig, auto_config, CategoryMesh, NeuralMesh, Strand
from .model import NpDnaCore, NpDnaModel, Genome, LoRALinear, inject_lora, load_lora_adapter, mark_only_lora_trainable, save_lora_adapter
from .tokenizer import AtulyaTokenizer
from .cognition import CortexAutoStore, MemoryCortex

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
    configure_cpu_runtime,
    compile_readiness,
    optimize_with_ipex,
    prepare_cpu_inference,
    model_size_mb,
    freeze_for_partial_training,
    # Benchmark
    benchmark_checkpoint,
    write_benchmark,
)

__all__ = [
    "CONFIGS", "PREFERRED_CONFIG_NAMES", "NpDnaConfig", "auto_config",
    "Genome", "Strand", "NeuralMesh", "CategoryMesh",
    "MemoryCortex", "CortexAutoStore",
    "NpDnaModel", "NpDnaCore",
    "LoRALinear", "inject_lora", "mark_only_lora_trainable", "save_lora_adapter", "load_lora_adapter",
    "AtulyaTokenizer",
    "PlasticityEngine", "PlasticityMetrics", "PlasticityAutoScaler",
    "NpDnaTopicClassifier", "tag_text",
    "NpDnaAgent",
    "build_multimodal_prompt", "encode_image_clip", "describe_image", "describe_audio",
    "quantize_model_for_cpu", "apply_torch_compile", "configure_cpu_runtime", "compile_readiness", "optimize_with_ipex", "prepare_cpu_inference", "model_size_mb",
    "freeze_for_partial_training",
    "benchmark_checkpoint", "write_benchmark",
]
