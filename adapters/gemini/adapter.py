from tantra_core.protocol.adapter import BaseTantraAdapter
from tantra_core.schema.models import TantraRequest, TantraResponse, ModelProvider
import uuid

class GeminiAdapter(BaseTantraAdapter):
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def generate(self, request: TantraRequest) -> TantraResponse:
        # Stub for Gemini API call
        return TantraResponse(
            content=f"Gemini response to: {request.messages[-1].content}",
            model=request.model,
            provider=ModelProvider.GEMINI,
            usage={"prompt_tokens": 8, "completion_tokens": 4},
            cost=0.005,
            trace_id=request.trace_id or str(uuid.uuid4())
        )

    async def stream(self, request: TantraRequest):
        yield await self.generate(request)

    def health_check(self) -> bool:
        return True
