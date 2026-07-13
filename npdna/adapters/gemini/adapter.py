from npdna.atulya_core.protocol.adapter import BaseTantraAdapter
from npdna.atulya_core.schema.models import TantraRequest, TantraResponse, ModelProvider
import uuid
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from google import genai as _genai_sdk
    from google.genai import types as _genai_types
except ImportError:
    _genai_sdk = None
    _genai_types = None


class GeminiAdapter(BaseTantraAdapter):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    def _is_configured(self) -> bool:
        return bool(_genai_sdk and self.api_key)

    async def generate(self, request: TantraRequest) -> TantraResponse:
        if self._is_configured():
            try:
                client = _genai_sdk.Client(api_key=self.api_key)
                model_name = getattr(request, "model", None) or "gemini-1.5-pro"
                config = _genai_types.GenerateContentConfig(
                    temperature=request.temperature or 0.7,
                    top_p=request.top_p or 0.9,
                )
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=request.messages[-1].content,
                    config=config,
                )
                content = response.text
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count
                # Gemini 1.5 Pro pricing: $1.25/1M input, $5.00/1M output
                cost = (prompt_tokens * 0.00000125) + (completion_tokens * 0.000005)
                return TantraResponse(
                    content=content,
                    model=model_name,
                    provider=ModelProvider.GEMINI,
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
                    "Gemini API call failed (%s). "
                    "Set GEMINI_API_KEY to enable real calls. Falling back to simulation.",
                    e,
                )

        # --- Simulation fallback (no key or SDK not installed) ---
        content = f"[SIMULATED] Gemini response to: {request.messages[-1].content}"
        prompt_tokens = max(1, sum(len(m.content) for m in request.messages) // 4)
        completion_tokens = max(1, len(content) // 4)
        cost = (prompt_tokens * 0.00000125) + (completion_tokens * 0.000005)
        return TantraResponse(
            content=content,
            model="gemini-1.5-pro",
            provider=ModelProvider.GEMINI,
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
                client = _genai_sdk.Client(api_key=self.api_key)
                model_name = getattr(request, "model", None) or "gemini-1.5-pro"
                config = _genai_types.GenerateContentConfig(
                    temperature=request.temperature or 0.7,
                    top_p=request.top_p or 0.9,
                )
                async for chunk in await client.aio.models.generate_content_stream(
                    model=model_name,
                    contents=request.messages[-1].content,
                    config=config,
                ):
                    if chunk.text:
                        yield TantraResponse(
                            content=chunk.text,
                            model=model_name,
                            provider=ModelProvider.GEMINI,
                            usage={"prompt_tokens": 0, "completion_tokens": 0, "simulated": False},
                            cost=0.0,
                            trace_id=request.trace_id or str(uuid.uuid4()),
                        )
                return
            except Exception as e:
                logger.warning("Gemini streaming failed (%s). Falling back to simulation stream.", e)

        yield await self.generate(request)

    def health_check(self) -> bool:
        return self._is_configured()
