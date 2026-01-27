from atulya_core.protocol.adapter import BaseTantraAdapter
from atulya_core.schema.models import TantraRequest, TantraResponse, ModelProvider
import uuid

class OpenAIAdapter(BaseTantraAdapter):
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def generate(self, request: TantraRequest) -> TantraResponse:
        # Stub for OpenAI API call
        return TantraResponse(
            content=f"OpenAI response to: {request.messages[-1].content}",
            model=request.model,
            provider=ModelProvider.OPENAI,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            cost=0.01,
            trace_id=request.trace_id or str(uuid.uuid4())
        )

    async def stream(self, request: TantraRequest):
        # Stub for streaming
        yield await self.generate(request)

    def health_check(self) -> bool:
        return True
