from atulya_core.protocol.adapter import BaseTantraAdapter
from atulya_core.schema.models import TantraRequest, TantraResponse, ModelProvider
import uuid

class LocalInferenceAdapter(BaseTantraAdapter):
    """
    CPU-Optimized Local Inference Adapter.
    Production-grade stub for RWKV/Llama weights.
    """
    def __init__(self, model_id: str = "tantra-7b-cpu"):
        self.model_id = model_id

    async def generate(self, request: TantraRequest) -> TantraResponse:
        return TantraResponse(
            content=f"Processed via {self.model_id}: {request.messages[-1].content}",
            model=self.model_id,
            provider=ModelProvider.LOCAL,
            usage={"tokens": 10},
            trace_id=request.trace_id or str(uuid.uuid4())
        )

    async def stream(self, request: TantraRequest):
        yield await self.generate(request)

    def health_check(self) -> bool:
        return True
