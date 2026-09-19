"""Tantra NeuroCore — Hindi-first compact foundation model."""

from Tantra.config import NeuroCoreConfig, VocabConfig, ALRAConfig, SGPConfig, TrainingConfig, InferenceConfig, BitNetConfig, MoEConfig, AdapterConfig
from Tantra.model import NeuroCoreModel, cpu_dense_config, build_cpu_model
from Tantra.train import NeuroTrainer, build_optimizer, create_lr_scheduler
from Tantra.dataset import JSONLDataset, PretokenizedBinDataset
from Tantra.tokenizer import ByteBPETokenizer, UnifiedTokenizer
from Tantra.hardware import HardwareDetector, RuntimeConfigBuilder, AdaptiveScheduler
from Tantra.ui import print_banner, print_status_dashboard, print_expert_panel
from Tantra.cli_hardware_dispatch import detect_hardware
from Tantra.codec import DNACodec
from Tantra.moe import ExpertRegistry
from Tantra.utils import get_logger, safe_load_checkpoint
from Tantra.eval_suite import EvaluationEngine
from Tantra.adapters import AdapterRegistry

__version__ = "0.5.0"
__all__ = [
    "NeuroCoreConfig", "VocabConfig", "ALRAConfig", "SGPConfig",
    "TrainingConfig", "InferenceConfig", "BitNetConfig", "MoEConfig", "AdapterConfig",
    "NeuroCoreModel", "cpu_dense_config", "build_cpu_model",
    "NeuroTrainer", "build_optimizer", "create_lr_scheduler",
    "JSONLDataset", "PretokenizedBinDataset",
    "ByteBPETokenizer", "UnifiedTokenizer",
    "HardwareDetector", "RuntimeConfigBuilder", "AdaptiveScheduler",
    "print_banner", "print_status_dashboard", "print_expert_panel",
    "detect_hardware", "DNACodec", "ExpertRegistry",
    "get_logger", "safe_load_checkpoint", "EvaluationEngine", "AdapterRegistry",
    "__version__",
]