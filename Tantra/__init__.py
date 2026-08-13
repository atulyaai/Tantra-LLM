"""
tantra package init
"""
from Tantra.config import NeuroCoreConfig, VocabConfig, MoEConfig, CompressionConfig, BitNetConfig
from Tantra.utils import get_logger, set_seed
from Tantra.model import NeuroCoreModel, LatentCoTHeader
from Tantra.bitnet import BitLinear, TernaryQuantizer
from Tantra.moe import ExpertRegistry, MoERouter, LazyExpertLoader
from Tantra.tokenizer import UnifiedTokenizer, ByteBPETokenizer, ModalityRouter
from Tantra.codec import DNACodec, CompressionBenchmark
# hardware module is imported lazily to avoid psutil blocking on Colab/containers
from Tantra.train import NeuroTrainer
from Tantra.dataset import JSONLDataset, extract_corpus_sample
from Tantra.evolution import AutoGrowthController, SelfRepairEngine
from Tantra.eval import EvaluationEngine

__version__ = "1.0.0"

__all__ = [
    "NeuroCoreConfig", "VocabConfig", "MoEConfig", "CompressionConfig", "BitNetConfig",
    "get_logger", "set_seed",
    "NeuroCoreModel", "LatentCoTHeader", "BitLinear", "TernaryQuantizer",
    "ExpertRegistry", "MoERouter", "LazyExpertLoader",
    "UnifiedTokenizer", "ByteBPETokenizer", "ModalityRouter",
    "DNACodec", "CompressionBenchmark",
    "NeuroTrainer", "JSONLDataset", "TopicMixedDataset", "extract_corpus_sample",
    "AutoGrowthController", "SelfRepairEngine", "EvaluationEngine",
]
