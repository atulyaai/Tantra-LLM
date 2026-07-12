from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

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
