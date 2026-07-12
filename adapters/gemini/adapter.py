from atulya_core.protocol.adapter import BaseTantraAdapter
from atulya_core.schema.models import TantraRequest, TantraResponse, ModelProvider
import uuid
import os
from typing import Optional

class GeminiAdapter(BaseTantraAdapter):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    async def generate(self, request: TantraRequest) -> TantraResponse:
        content = f"Gemini response to: {request.messages[-1].content}"
        
        # Calculate real-world token estimation
        prompt_char_count = sum(len(m.content) for m in request.messages)
        prompt_tokens = max(1, prompt_char_count // 4)
        completion_tokens = max(1, len(content) // 4)
        
        # Gemini pricing estimate ($1.25/1M input, $5.00/1M output tokens)
        cost = (prompt_tokens * 0.00000125) + (completion_tokens * 0.000005)
        
        return TantraResponse(
            content=content,
            model="gemini-1.5-pro",
            provider=ModelProvider.GEMINI,
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            cost=round(cost, 6),
            trace_id=request.trace_id or str(uuid.uuid4())
        )

    async def stream(self, request: TantraRequest):
        yield await self.generate(request)

    def health_check(self) -> bool:
        # Simple health check based on key availability
        return len(self.api_key) > 0 or "GEMINI_API_KEY" in os.environ
