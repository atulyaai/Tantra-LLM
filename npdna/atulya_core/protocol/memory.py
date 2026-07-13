from typing import Protocol, List, Dict, Any
from pydantic import BaseModel, Field

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
