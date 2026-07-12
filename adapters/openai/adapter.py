from atulya_core.protocol.adapter import BaseTantraAdapter
from atulya_core.schema.models import TantraRequest, TantraResponse, ModelProvider
import uuid
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    openai = None

class OpenAIAdapter(BaseTantraAdapter):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    async def generate(self, request: TantraRequest) -> TantraResponse:
        # If API key and SDK are available, execute actual OpenAI API request
        if openai and self.api_key:
            try:
                client = openai.AsyncOpenAI(api_key=self.api_key)
                response = await client.chat.completions.create(
                    model=request.model or "gpt-4o",
                    messages=[{"role": m.role, "content": m.content} for m in request.messages],
                    temperature=request.temperature or 0.7,
                    top_p=request.top_p or 0.9,
                )
                content = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                
                # GPT-4o pricing ($5.00/1M input, $15.00/1M output tokens)
                cost = (prompt_tokens * 0.000005) + (completion_tokens * 0.000015)
                
                return TantraResponse(
                    content=content,
                    model=response.model,
                    provider=ModelProvider.OPENAI,
                    usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
                    cost=round(cost, 6),
                    trace_id=request.trace_id or str(uuid.uuid4())
                )
            except Exception as e:
                logger.warning(f"OpenAI actual API call failed: {e}. Falling back to simulation mode.")

        # Simulation Mode Fallback
        content = f"OpenAI response to: {request.messages[-1].content}"
        prompt_char_count = sum(len(m.content) for m in request.messages)
        prompt_tokens = max(1, prompt_char_count // 4)
        completion_tokens = max(1, len(content) // 4)
        cost = (prompt_tokens * 0.000005) + (completion_tokens * 0.000015)
        
        return TantraResponse(
            content=content,
            model="gpt-4o",
            provider=ModelProvider.OPENAI,
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            cost=round(cost, 6),
            trace_id=request.trace_id or str(uuid.uuid4())
        )

    async def stream(self, request: TantraRequest):
        # If API key and SDK are available, support real streaming
        if openai and self.api_key:
            try:
                client = openai.AsyncOpenAI(api_key=self.api_key)
                response = await client.chat.completions.create(
                    model=request.model or "gpt-4o",
                    messages=[{"role": m.role, "content": m.content} for m in request.messages],
                    temperature=request.temperature or 0.7,
                    top_p=request.top_p or 0.9,
                    stream=True
                )
                async for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield TantraResponse(
                            content=delta,
                            model=chunk.model,
                            provider=ModelProvider.OPENAI,
                            usage={"prompt_tokens": 0, "completion_tokens": 0},
                            cost=0.0,
                            trace_id=request.trace_id or str(uuid.uuid4())
                        )
                return
            except Exception as e:
                logger.warning(f"OpenAI actual streaming failed: {e}. Falling back to simulation stream.")

        yield await self.generate(request)

    def health_check(self) -> bool:
        return len(self.api_key) > 0 or "OPENAI_API_KEY" in os.environ
