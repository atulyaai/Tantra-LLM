"""Shared API contracts, identity, and settings for NP-DNA.

Everything here is transport- and model-agnostic: request/response types,
adapter/middleware/memory protocols, identity/behavioral profiles, and the
validated settings singleton. No torch tensors and no neural-net code live
here — that stays in architecture.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field, model_validator


class ModelProvider(str, Enum):
    LOCAL = "local"
    GEMINI = "gemini"
    OPENAI = "openai"


@dataclass
class Message:
    role: str
    content: str


@dataclass
class TantraRequest:
    messages: List[Message]
    provider: Optional[ModelProvider] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    trace_id: Optional[str] = None


@dataclass
class TantraResponse:
    content: str
    model: str
    provider: ModelProvider
    usage: Dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    trace_id: Optional[str] = None
    entropy_score: float = 0.0
    confidence_level: Optional[str] = None


@dataclass
class RequestContext:
    """Shared execution context threaded through the middleware chain."""
    trace_id: str
    user_id: Optional[str] = None
    budget_remaining: float = 10.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskStep:
    id: str
    instruction: str
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TaskPlan:
    plan_id: str
    goal: str
    nodes: List[TaskStep] = field(default_factory=list)


@dataclass
class SystemPulse:
    """20Hz heartbeat payload broadcast by the Event Bus."""
    cpu_load: float = 0.0
    mem_usage: float = 0.0
    active_modules: List[str] = field(default_factory=list)

    def model_dump(self) -> dict:
        return {
            "cpu_load": self.cpu_load,
            "mem_usage": self.mem_usage,
            "active_modules": self.active_modules,
        }


class BaseTantraAdapter:
    """Base class for all Tantra model adapters."""
    async def generate(self, request: TantraRequest) -> TantraResponse:
        raise NotImplementedError("Adapter must implement generate.")

    async def stream(self, request: TantraRequest) -> AsyncGenerator[TantraResponse, None]:
        raise NotImplementedError("Adapter must implement stream.")

    def health_check(self) -> bool:
        raise NotImplementedError("Adapter must implement health_check.")


class ModalityEncoder(Protocol):
    """Protocol enforcing shape and dimension constraints on all sensory modality encoders."""

    @property
    def embed_dim(self) -> int:
        """The target embedding dimension required by the base model projection."""
        ...

    def encode(self, data: Any) -> "torch.Tensor":
        """
        Processes sensory input data and returns a normalized embedding tensor.

        Returns:
            A torch.Tensor of shape [1, embed_dim].
        """
        ...


class MemoryChunk(BaseModel):
    """Represent a retrieved slice of episodic or semantic memory."""
    content: str
    score: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryStore(Protocol):
    """Protocol defining the memory storage service boundary (RAG interface)."""
    async def retrieve(self, query: str, k: int = 5) -> List[MemoryChunk]:
        """Retrieve the top k most relevant memory chunks for the given query."""
        ...

    async def write(self, content: str, metadata: Dict[str, Any]) -> None:
        """Write a new memory content block along with metadata to the store."""
        ...

    async def consolidate(self) -> None:
        """Run background consolidation tasks (e.g. clustering, summary, indexing)."""
        ...


class TantraMiddleware:
    """Base class for all Tantra middleware layers."""
    async def __call__(
        self,
        request: TantraRequest,
        context: RequestContext,
        call_next: Callable[[TantraRequest, RequestContext], Awaitable[TantraResponse]]
    ) -> TantraResponse:
        return await call_next(request, context)


# ── Identity, model config, settings (from fusion_config.py) ────────────────

IDENTITY = {
    "name": "Tantra",
    "version": "0.1-origins",
    "capability": "Core architecture and basic IO routing operational",

    # Speaking style profile
    "speaking_style": {
        "default": "concise, direct, analytical",
        "verbosity": 0.4,  # 0 = minimal, 1 = verbose
        "formality": 0.6,  # 0 = casual, 1 = formal
        "assertiveness": 0.7,  # 0 = tentative, 1 = decisive
        "humor": 0.2,  # 0 = serious, 1 = playful
    },

    # Reasoning style profile
    "reasoning_style": {
        "default": "step-by-step, memory-aware, context-adaptive",
        "chain_of_thought": True,
        "show_work": False,  # Don't expose internal reasoning by default
        "confidence_calibration": 0.8,  # 0 = overconfident, 1 = calibrated
        "uncertainty_acknowledgment": True,
    },

    # Memory personality
    "memory_personality": {
        "default": "selective retention, importance-weighted",
        "retention_threshold": 0.5,  # 0 = keep all, 1 = keep critical only
        "compression_ratio": 0.7,  # How aggressively to compress memories
        "consolidation_frequency": "weekly",  # How often to merge similar memories
        "forgetting_curve": "exponential",  # How memories decay
    },

    # Performance preferences
    "latency_vs_precision": {
        "default": "balanced; shorten under pressure",
        "fast_threshold_ms": 500,  # Target for simple queries
        "medium_threshold_ms": 2000,  # Target for reasoning tasks
        "deep_threshold_ms": 10000,  # Target for complex analysis
        "precision_tradeoff": 0.8,  # 0 = speed, 1 = accuracy
    },

    # Behavioral boundaries
    "behavioral_boundaries": {
        "ethical_constraints": [
            "no instructions for irreversible harm",
            "no personal data extraction or doxxing",
            "no fabrication presented as verified truth",
            "do not damage user's world, body, or reputation",
        ],
        "allowed_risks": [
            "provocative ideas framed as speculation",
            "rigorous critique of assumptions",
            "exploration of controversial but non-harmful topics",
        ],
        "privacy_level": "restricted",  # public | restricted | sealed
        "fact_checking": True,
        "hallucination_prevention": True,
    },

    # Interaction preferences
    "interaction_style": {
        "interrupt_allowed": True,
        "correction_welcome": True,
        "feedback_learning": True,
        "conversation_pacing": "adaptive",  # user-controlled | adaptive | fixed
        "response_format": "markdown",  # plain | markdown | structured
    },
}


MEMORY_CONFIG = {
    # Working memory window (token count)
    "working_tokens": 8192,
    # Episodic memory consolidation threshold
    "episodic_threshold": 0.7,
    # Embedding dimensionality (aligned with model hidden size)
    "embedding_dim": 4096,
    # Maximum episodic memories before eviction
    "max_episodic": 10000,
    # How often (in steps) to consolidate similar memories
    "consolidation_frequency": 100,
}


MODEL_CONFIG = {
    "model_dim": 4096,  # Fusion embedding dimension
    "wm_tokens": MEMORY_CONFIG["working_tokens"],  # Working memory window (single source)
    "npdna": {
        "checkpoint_path": "model/latest",
    },
    "vision": {
        "embed_dim": 4096,  # Aligned with model_dim
        "remote": True,
        "api_url": None,
        "local_path": None,
    },
    "audio": {
        "embed_dim": 4096,  # Aligned with model_dim
        "remote": False,
        "model_name": "openai/whisper-large-v3",
        "local_path": None,
    },
    # Memory settings: see MEMORY_CONFIG above
    "memory": MEMORY_CONFIG,
    # Personality settings: personality_config.json loaded at runtime by PersonalityLayer.
    "compute": {
        "max_tokens": 200,
        "context_window": 4096,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    },
}


class MemorySettings(BaseModel):
    working_tokens: int = 8192
    episodic_threshold: float = 0.7
    embedding_dim: int = 4096
    max_episodic: int = 10000
    consolidation_frequency: int = 100


class NpDnaSettings(BaseModel):
    checkpoint_path: str = "model/latest"


class VisionSettings(BaseModel):
    embed_dim: int = 4096
    remote: bool = True
    api_url: Optional[str] = None
    local_path: Optional[str] = None


class AudioSettings(BaseModel):
    embed_dim: int = 4096
    remote: bool = False
    model_name: str = "openai/whisper-large-v3"
    local_path: Optional[str] = None


class ComputeSettings(BaseModel):
    max_tokens: int = 200
    context_window: int = 4096
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.1


class TantraSettings(BaseModel):
    model_dim: int = 4096
    memory: MemorySettings = Field(default_factory=MemorySettings)
    npdna: NpDnaSettings = Field(default_factory=NpDnaSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    compute: ComputeSettings = Field(default_factory=ComputeSettings)

    @model_validator(mode="after")
    def validate_dimensions(self) -> 'TantraSettings':
        dim = self.model_dim
        if self.vision.embed_dim != dim:
            raise ValueError(f"Vision embed_dim ({self.vision.embed_dim}) must match model_dim ({dim})")
        if self.audio.embed_dim != dim:
            raise ValueError(f"Audio embed_dim ({self.audio.embed_dim}) must match model_dim ({dim})")
        if self.memory.embedding_dim != dim:
            raise ValueError(f"Memory embedding_dim ({self.memory.embedding_dim}) must match model_dim ({dim})")
        return self


# Global singleton settings loaded and validated at startup
_settings = None


def get_settings() -> TantraSettings:
    global _settings
    if _settings is None:
        try:
            _settings = TantraSettings(**MODEL_CONFIG)
        except Exception as e:
            # Fall back to default schema config if error loading
            print(f"[Config] Failed to load config: {e}. Falling back to default schema.")
            _settings = TantraSettings()
    return _settings


LONG_FORM_HINTS = {
    "code",
    "function",
    "write",
    "story",
    "essay",
    "explain",
    "describe",
    "compare",
    "summarize",
    "steps",
    "plan",
}


def infer_max_tokens(prompt: str) -> int:
    """Pick a practical generation cap until the model learns reliable EOS."""
    words = prompt.split()
    lowered = prompt.lower()
    if any(hint in lowered for hint in LONG_FORM_HINTS):
        return 120
    if len(words) <= 8 and prompt.strip().endswith("?"):
        return 40
    if len(words) <= 20:
        return 64
    return 96
