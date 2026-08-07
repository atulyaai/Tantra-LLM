"""
tantra package init
"""
from tantra.config import NeuroCoreConfig, VocabConfig, MoEConfig, CompressionConfig, BitNetConfig
from tantra.utils import get_logger, set_seed
from tantra.model import NeuroCoreModel, LatentCoTHeader
from tantra.bitnet import BitLinear, TernaryQuantizer
from tantra.moe import ExpertRegistry, MoERouter, LazyExpertLoader
from tantra.tokenizer import UnifiedTokenizer, ByteBPETokenizer, ModalityRouter
from tantra.codec import DNACodec, CompressionBenchmark
from tantra.hardware import HardwareDetector, Profiler, RuntimeConfigBuilder
from tantra.train import NeuroTrainer
from tantra.dataset import JSONLDataset, extract_corpus_sample
from tantra.evolution import AutoGrowthController, SelfRepairEngine
from tantra.eval import EvaluationEngine

__version__ = "1.0.0"

__all__ = [
    "NeuroCoreConfig", "VocabConfig", "MoEConfig", "CompressionConfig", "BitNetConfig",
    "get_logger", "set_seed",
    "NeuroCoreModel", "LatentCoTHeader", "BitLinear", "TernaryQuantizer",
    "ExpertRegistry", "MoERouter", "LazyExpertLoader",
    "UnifiedTokenizer", "ByteBPETokenizer", "ModalityRouter",
    "DNACodec", "CompressionBenchmark",
    "HardwareDetector", "Profiler", "RuntimeConfigBuilder",
    "NeuroTrainer", "JSONLDataset", "extract_corpus_sample",
    "AutoGrowthController", "SelfRepairEngine", "EvaluationEngine",
]
