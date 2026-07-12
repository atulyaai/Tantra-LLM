from atulya_core.protocol.adapter import BaseTantraAdapter
from atulya_core.schema.models import TantraRequest, TantraResponse, ModelProvider
import uuid
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class GeminiAdapter(BaseTantraAdapter):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    async def generate(self, request: TantraRequest) -> TantraResponse:
        # If API key and SDK are available, execute actual Gemini API request
        if genai and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(request.model or "gemini-1.5-pro")
                
                # Call async API
                response = await model.generate_content_async(
                    request.messages[-1].content,
                    generation_config={
                        "temperature": request.temperature or 0.7,
                        "top_p": request.top_p or 0.9,
                    }
                )
                content = response.text
                
                # Fetch actual API usage metadata
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count
                
                # Gemini 1.5 Pro pricing ($1.25/1M input, $5.00/1M output tokens)
                cost = (prompt_tokens * 0.00000125) + (completion_tokens * 0.000005)
                
                return TantraResponse(
                    content=content,
                    model="gemini-1.5-pro",
                    provider=ModelProvider.GEMINI,
                    usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
                    cost=round(cost, 6),
                    trace_id=request.trace_id or str(uuid.uuid4())
                )
            except Exception as e:
                logger.warning(f"Gemini actual API call failed: {e}. Falling back to simulation mode.")

        # Simulation Mode Fallback
        content = f"Gemini response to: {request.messages[-1].content}"
        prompt_char_count = sum(len(m.content) for m in request.messages)
        prompt_tokens = max(1, prompt_char_count // 4)
        completion_tokens = max(1, len(content) // 4)
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
        # If API key and SDK are available, support real streaming
        if genai and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(request.model or "gemini-1.5-pro")
                response = await model.generate_content_async(
                    request.messages[-1].content,
                    generation_config={
                        "temperature": request.temperature or 0.7,
                        "top_p": request.top_p or 0.9,
                    },
                    stream=True
                )
                async for chunk in response:
                    if chunk.text:
                        yield TantraResponse(
                            content=chunk.text,
                            model="gemini-1.5-pro",
                            provider=ModelProvider.GEMINI,
                            usage={"prompt_tokens": 0, "completion_tokens": 0},
                            cost=0.0,
                            trace_id=request.trace_id or str(uuid.uuid4())
                        )
                return
            except Exception as e:
                logger.warning(f"Gemini actual streaming failed: {e}. Falling back to simulation stream.")

        yield await self.generate(request)

    def health_check(self) -> bool:
        return len(self.api_key) > 0 or "GEMINI_API_KEY" in os.environ
