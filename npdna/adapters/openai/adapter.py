from npdna.atulya_core.protocol.adapter import BaseTantraAdapter
from npdna.atulya_core.schema.models import TantraRequest, TantraResponse, ModelProvider
import uuid
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import openai as _openai_sdk
except ImportError:
    _openai_sdk = None


class OpenAIAdapter(BaseTantraAdapter):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def _is_configured(self) -> bool:
        return bool(_openai_sdk and self.api_key)

    async def generate(self, request: TantraRequest) -> TantraResponse:
        if self._is_configured():
            try:
                client = _openai_sdk.AsyncOpenAI(api_key=self.api_key)
                response = await client.chat.completions.create(
                    model=getattr(request, "model", None) or "gpt-4o",
                    messages=[{"role": m.role, "content": m.content} for m in request.messages],
                    temperature=request.temperature or 0.7,
                    top_p=request.top_p or 0.9,
                )
                content = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                # GPT-4o pricing: $5.00/1M input, $15.00/1M output
                cost = (prompt_tokens * 0.000005) + (completion_tokens * 0.000015)
                return TantraResponse(
                    content=content,
                    model=response.model,
                    provider=ModelProvider.OPENAI,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "simulated": False,
                    },
                    cost=round(cost, 6),
                    trace_id=request.trace_id or str(uuid.uuid4()),
                )
            except Exception as e:
                logger.warning(
                    "OpenAI API call failed (%s). "
                    "Set OPENAI_API_KEY to enable real calls. Falling back to simulation.",
                    e,
                )

        # --- Simulation fallback (no key or SDK not installed) ---
        content = f"[SIMULATED] OpenAI response to: {request.messages[-1].content}"
        prompt_tokens = max(1, sum(len(m.content) for m in request.messages) // 4)
        completion_tokens = max(1, len(content) // 4)
        cost = (prompt_tokens * 0.000005) + (completion_tokens * 0.000015)
        return TantraResponse(
            content=content,
            model="gpt-4o",
            provider=ModelProvider.OPENAI,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "simulated": True,
            },
            cost=round(cost, 6),
            trace_id=request.trace_id or str(uuid.uuid4()),
        )

    async def stream(self, request: TantraRequest):
        if self._is_configured():
            try:
                client = _openai_sdk.AsyncOpenAI(api_key=self.api_key)
                stream = await client.chat.completions.create(
                    model=getattr(request, "model", None) or "gpt-4o",
                    messages=[{"role": m.role, "content": m.content} for m in request.messages],
                    temperature=request.temperature or 0.7,
                    top_p=request.top_p or 0.9,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield TantraResponse(
                            content=delta,
                            model=chunk.model,
                            provider=ModelProvider.OPENAI,
                            usage={"prompt_tokens": 0, "completion_tokens": 0, "simulated": False},
                            cost=0.0,
                            trace_id=request.trace_id or str(uuid.uuid4()),
                        )
                return
            except Exception as e:
                logger.warning("OpenAI streaming failed (%s). Falling back to simulation stream.", e)

        yield await self.generate(request)

    def health_check(self) -> bool:
        return self._is_configured()
