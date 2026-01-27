from atulya_core.schema.models import TantraRequest, TantraResponse, ModelProvider
from atulya_core.protocol.adapter import BaseTantraAdapter
import uuid
import time

class CPUOptimizedAdapter(BaseTantraAdapter):
    """
    A high-performance, local-first brain stub.
    Simulates a local model running on CPU with minimal footprint.
    """
    def __init__(self, model_path: str = "local/small-brain"):
        self.model_path = model_path

    async def generate(self, request: TantraRequest) -> TantraResponse:
        # Simulate local CPU reasoning delay
        start_time = time.time()
        
        # Simulation of "Jarvis-level" reasoning logic
        content = f"Local CPU (Tantra-LLM) analysis complete for: '{request.messages[-1].content}'. "
        content += "Reasoning: Direct, local, and private."
        
        latency = (time.time() - start_time) * 1000
        
        return TantraResponse(
            content=content,
            model=request.model or "tantra-cpu-v1",
            provider=ModelProvider.LOCAL,
            usage={"prompt_tokens": 15, "completion_tokens": 10},
            cost=0.0,
            trace_id=request.trace_id or str(uuid.uuid4()),
            entropy_score=0.05
        )

    async def stream(self, request: TantraRequest):
        yield await self.generate(request)

    def health_check(self) -> bool:
        return True
