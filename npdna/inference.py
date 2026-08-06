"""Inference hub: model orchestration, cognitive wiring, middleware chain, adapters.

Single flat module (previously ``core/inference.py`` + ``adapters/`` + ``middleware.py``).
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from collections import Counter
from pathlib import Path

from npdna import NpDnaCore
from npdna.schema import (
    BaseTantraAdapter,
    Message,
    ModelProvider,
    RequestContext,
    TantraMiddleware,
    TantraRequest,
    infer_max_tokens,
    TantraResponse,
)
from npdna.cognition import EventBus, ComputeRouter, DynamicContextManager

logger = logging.getLogger(__name__)

_BUS_AVAILABLE = True


class UnifiedInferenceHub:
    """
    Production-level Inference Hub for the Atulya Tantra OS-Organism.
    Strictly handles model orchestration, cognitive wiring, and middleware.
    """
    def __init__(self, bus: Optional[Any] = None, max_failures: int = 3, cooldown_seconds: float = 30.0):
        self.adapters: Dict[ModelProvider, List[BaseTantraAdapter]] = {}
        self.middlewares: List[TantraMiddleware] = []
        self.bus = bus
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds

        # Track circuit breaker status per adapter
        # Key: adapter, Value: {"failures": int, "tripped_until": Optional[float]}
        self.circuit_breakers: Dict[BaseTantraAdapter, Dict[str, Any]] = {}

        self.context_manager = DynamicContextManager()
        self.router = ComputeRouter()

        # Register middleware chain in execution order:
        # 1. Personality processing (mutates request early)
        # 2. Observability monitoring (times inner layers)
        # 3. Safety audits (filters outputs late)
        self.add_middleware(PersonalityMiddleware())
        self.add_middleware(TraceObservabilityMiddleware())
        self.add_middleware(SafetyMiddleware())

    def register_adapter(self, provider: ModelProvider, adapter: BaseTantraAdapter):
        if provider not in self.adapters:
            self.adapters[provider] = []
        self.adapters[provider].append(adapter)
        self.circuit_breakers[adapter] = {"failures": 0, "tripped_until": None}

    def add_middleware(self, middleware: TantraMiddleware):
        self.middlewares.append(middleware)

    def _get_active_adapters(self, provider: ModelProvider) -> List[BaseTantraAdapter]:
        """Finds all adapters that are registered and not currently tripped by the circuit breaker."""
        now = time.time()
        candidates = list(self.adapters.get(provider, []))
        if provider != ModelProvider.LOCAL:
            candidates.extend(self.adapters.get(ModelProvider.LOCAL, []))

        active = []
        for adapter in candidates:
            if adapter in active:
                continue
            cb = self.circuit_breakers.get(adapter, {"failures": 0, "tripped_until": None})
            tripped_until = cb.get("tripped_until")
            if tripped_until and now < tripped_until:
                # Tripped, skip
                continue
            active.append(adapter)
        return active

    def get_active_adapter(self, provider: ModelProvider) -> Optional[BaseTantraAdapter]:
        """Returns the first active, non-tripped adapter for the given provider."""
        active = self._get_active_adapters(provider)
        return active[0] if active else None

    def _record_failure(self, adapter: BaseTantraAdapter):
        """Records a failure on the adapter, tripping the circuit if max_failures is reached."""
        cb = self.circuit_breakers.setdefault(adapter, {"failures": 0, "tripped_until": None})
        cb["failures"] += 1
        if cb["failures"] >= self.max_failures:
            cb["tripped_until"] = time.time() + self.cooldown_seconds
            print(f"[CircuitBreaker] Tripped adapter {adapter} until {cb['tripped_until']}")

    def _record_success(self, adapter: BaseTantraAdapter):
        """Resets consecutive failures on successful execution."""
        cb = self.circuit_breakers.setdefault(adapter, {"failures": 0, "tripped_until": None})
        cb["failures"] = 0
        cb["tripped_until"] = None

    async def execute(self, request: TantraRequest) -> TantraResponse:
        # Create execution request context
        trace_id = request.trace_id or f"TRC-{uuid.uuid4().hex[:8]}"
        context = RequestContext(trace_id=trace_id)

        # Select path and apply Dynamic Context trimming (sliding window) early
        selected_path = "medium"
        if request.messages:
            raw_content = request.messages[-1].content
            provider = request.provider or self._route(request)
            provider_str = provider.value if hasattr(provider, 'value') else str(provider)
            selected_path = self.router.select_path(raw_content, provider_str)

            # Apply dynamic sliding window limit based on path
            target_limit = self.router.get_max_context(selected_path)
            pseudo_tokens = [ord(c) for c in raw_content]
            trimmed_tokens = self.context_manager.trim(pseudo_tokens, target_limit)

            if len(trimmed_tokens) < len(pseudo_tokens):
                request.messages[-1].content = "".join([chr(t) for t in trimmed_tokens])

        async def _core_call(req: TantraRequest, ctx: RequestContext) -> TantraResponse:
            provider = req.provider or self._route(req)
            candidates = self._get_active_adapters(provider)

            if not candidates:
                raise RuntimeError("Primary and Fallback Adapters unavailable or tripped.")

            last_err = None
            for adapter in candidates:
                try:
                    res = await adapter.generate(req)
                    self._record_success(adapter)
                    return res
                except Exception as e:
                    self._record_failure(adapter)
                    last_err = e
                    ctx.retry_count += 1
                    print(f"[Fallback] Adapter {adapter} failed: {e}. Retrying with next candidate (retry={ctx.retry_count}).")

            if last_err:
                raise RuntimeError(f"All adapters failed in execution chain. Last error: {last_err}") from last_err
            raise RuntimeError("Primary and Fallback Adapters unavailable or tripped.")

        # Build Middleware Chain: lambda r, c: middleware(r, c, next_handler)
        handler = _core_call
        for mw in reversed(self.middlewares):
            def wrap(m, h):
                return lambda r, c: m(r, c, h)
            handler = wrap(mw, handler)

        # Nervous system pulse (Async)
        if self.bus and _BUS_AVAILABLE:
            asyncio.create_task(self.bus.emit("inference_start", {"trace_id": context.trace_id}))

        response = await handler(request, context)

        if self.bus and _BUS_AVAILABLE:
            # Safely check if the response usage metadata flags it as simulated
            is_simulated = False
            if hasattr(response, "usage") and isinstance(response.usage, dict):
                is_simulated = response.usage.get("simulated", False)

            asyncio.create_task(self.bus.emit("inference_complete", {
                "trace_id": response.trace_id,
                "cost": 0.0 if is_simulated else response.cost,
                "entropy": response.entropy_score,
                "simulated": is_simulated
            }))

        # Record performance history back to ComputeRouter
        latency_ms = context.metadata.get("latency_ms", 0.0)
        is_simulated = False
        if hasattr(response, "usage") and isinstance(response.usage, dict):
            is_simulated = response.usage.get("simulated", False)
        if not is_simulated and latency_ms > 0.0:
            provider_str = response.provider.value if hasattr(response.provider, 'value') else str(response.provider)
            self.router.record_performance(selected_path, provider_str, latency_ms, response.cost)

        return response

    def _route(self, request: TantraRequest) -> ModelProvider:
        # Dynamic minimalist routing
        return ModelProvider.LOCAL


# ── Model Adapters (from adapters.py) ───────────────────────────────────────

class NpDnaAdapter(BaseTantraAdapter):
    """Run NP-DNA as the primary local model in ``UnifiedInferenceHub``."""

    DEFAULT_CHECKPOINTS = (Path("model/latest"),)

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        core: Optional[NpDnaCore] = None,
    ) -> None:
        self.checkpoint_path = self._select_checkpoint(checkpoint_path)
        if core is None:
            if self.checkpoint_path is None:
                raise FileNotFoundError(
                    "No NP-DNA checkpoint found. Expected model/latest."
                )
            core = NpDnaCore.load(self.checkpoint_path)
            suffix = self.checkpoint_path.name if self.checkpoint_path else "injected"
        else:
            suffix = "injected"
        self.core = core
        self.model_name = f"npdna-{suffix}"

    @classmethod
    def _select_checkpoint(cls, requested: str | Path | None) -> Optional[Path]:
        if requested:
            path = Path(requested)
            return path if path.exists() else None
        return next((path for path in cls.DEFAULT_CHECKPOINTS if path.exists()), None)

    async def generate(self, request: TantraRequest) -> TantraResponse:
        start = time.perf_counter()
        prompt = request.messages[-1].content if request.messages else ""
        metadata = getattr(request, "metadata", {}) or {}
        image_ref = metadata.get("image") or metadata.get("image_path")
        audio_ref = metadata.get("audio") or metadata.get("audio_path")
        structured_ref = metadata.get("structured")

        if image_ref or audio_ref or structured_ref:
            from npdna.brain import build_multimodal_prompt
            prompt = build_multimodal_prompt(
                prompt,
                image=image_ref,
                audio=audio_ref,
                structured=structured_ref,
            )

        max_tokens = infer_max_tokens(prompt)
        text = await asyncio.to_thread(
            self.core.generate,
            prompt,
            max_tokens=max_tokens,
            temperature=request.temperature if request.temperature is not None else 0.35,
            top_k=30,
            top_p=request.top_p if request.top_p is not None else 0.9,
            context_window=256,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        return TantraResponse(
            content=text,
            model=self.model_name,
            provider=ModelProvider.LOCAL,
            usage={
                "prompt_tokens": len(self.core.encode(prompt, allow_growth=False)),
                "completion_tokens": len(self.core.encode(text, allow_growth=False)),
                "latency_ms": latency_ms,
            },
            cost=0.0,
            trace_id=request.trace_id or str(uuid.uuid4()),
        )

    async def stream(self, request: TantraRequest) -> AsyncGenerator[TantraResponse, None]:
        yield await self.generate(request)

    def health_check(self) -> bool:
        return self.core is not None


try:
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE, PIPELINE_ARGS
    _RWKV_AVAILABLE = True
except ImportError:
    _RWKV_AVAILABLE = False


class RWKVAdapter(BaseTantraAdapter):
    """
    RWKV-v4/v5/v6 Adapter for Tantra-LLM.
    Provides efficient RNN-based inference.
    """
    def __init__(self, model_path: str = "models/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth"):
        self.model_path = model_path
        self.model = None
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        if not _RWKV_AVAILABLE:
            print("[RWKV] Library not installed. Please install 'rwkv'.")
            return

        if not os.path.exists(self.model_path):
            print(f"[RWKV] Model not found at {self.model_path}. Running in MOCK mode.")
            return

        print(f"[RWKV] Loading model from {self.model_path}...")
        strategy = "cuda fp16" if os.environ.get("USE_GPU") == "1" else "cpu fp32"

        try:
            self.model = RWKV(model=self.model_path, strategy=strategy)
            self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")
            print(f"[RWKV] Model loaded successfully with strategy: {strategy}")
        except Exception as e:
            print(f"[RWKV] Failed to load model: {e}")

    async def generate(self, request: TantraRequest) -> TantraResponse:
        start_time = time.time()
        prompt = request.messages[-1].content

        generated_text = ""

        if self.pipeline:
            args = PIPELINE_ARGS(
                temperature=request.temperature or 1.0,
                top_p=request.top_p or 0.7,
                top_k=100,
                alpha_frequency=0.25,
                alpha_presence=0.25,
                token_ban=[],
                token_stop=[],
                chunk_len=256,
            )

            loop = asyncio.get_event_loop()
            generated_text = await loop.run_in_executor(None, self.pipeline.generate, prompt, args)
        else:
            await asyncio.sleep(0.5)
            generated_text = f"[RWKV MOCK] Processed: '{prompt}'. (Model file missing)"

        latency = (time.time() - start_time) * 1000

        return TantraResponse(
            content=generated_text,
            model="rwkv-world",
            provider=ModelProvider.LOCAL,
            usage={"prompt_tokens": len(prompt) // 4, "completion_tokens": len(generated_text) // 4},
            cost=0.0,
            trace_id=request.trace_id or str(uuid.uuid4()),
            entropy_score=0.1
        )

    async def stream(self, request: TantraRequest) -> AsyncGenerator[TantraResponse, None]:
        response = await self.generate(request)
        yield response

    def health_check(self) -> bool:
        return _RWKV_AVAILABLE and (self.model is not None or not os.path.exists(self.model_path))


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
                logger.error("OpenAI API call failed: %s", e)
                raise RuntimeError(f"OpenAI API call failed: {e}") from e

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
                logger.error("OpenAI streaming failed: %s", e)
                raise RuntimeError(f"OpenAI streaming failed: {e}") from e

        yield await self.generate(request)

    def health_check(self) -> bool:
        return self._is_configured()


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
                logger.error("Gemini API call failed: %s", e)
                raise RuntimeError(f"Gemini API call failed: {e}") from e

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
                logger.error("Gemini streaming failed: %s", e)
                raise RuntimeError(f"Gemini streaming failed: {e}") from e

        yield await self.generate(request)

    def health_check(self) -> bool:
        return self._is_configured()


# ── Middleware (from middleware.py) ──────────────────────────────────────────

class PersonalityMiddleware(TantraMiddleware):
    """Middleware that dynamically updates the model parameters and prompt prefix based on personality/tones."""
    def __init__(self, personality_config: Optional[Dict] = None):
        if personality_config is None:
            personality_config = {}
            config_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "personality_config.json")
            )
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        personality_config = json.load(f)
                except Exception:
                    pass
        self.personality = PersonalityLayer(personality_config)

    async def __call__(
        self,
        request: TantraRequest,
        context: RequestContext,
        call_next: Callable[[TantraRequest, RequestContext], Awaitable[TantraResponse]]
    ) -> TantraResponse:
        if request.messages:
            user_prompt = request.messages[-1].content
            mode = self.personality.select_mode(user_prompt)
            params = self.personality.parameterize(mode)

            # Store selected mode in RequestContext metadata
            context.metadata["personality_mode"] = mode

            # Inject prompt prefix as system context if present
            prefix = params.get("prompt_prefix")
            if prefix:
                system_msg = Message(role="system", content=prefix)
                # Keep system prefix at start
                request.messages.insert(0, system_msg)

            # Apply parameterized top_p/temperature if not set
            if request.temperature is None:
                request.temperature = params.get("temperature", 0.7)
            if request.top_p is None:
                request.top_p = params.get("top_p", 0.9)

        return await call_next(request, context)


class SafetyMiddleware(TantraMiddleware):
    """Middleware that audits the output response against safety standards (deny list and toxicity)."""
    def __init__(self):
        self.safety = SafetyModule()

    async def __call__(
        self,
        request: TantraRequest,
        context: RequestContext,
        call_next: Callable[[TantraRequest, RequestContext], Awaitable[TantraResponse]]
    ) -> TantraResponse:
        # Safety middleware runs downstream of execution
        response = await call_next(request, context)

        safety_result = self.safety.evaluate(response.content, {})
        if safety_result["action"] == "deny":
            response.content = f"Response blocked by safety policy: {', '.join(safety_result['reasons'])}"
            response.confidence_level = "Blocked"
        elif safety_result["action"] == "modify":
            response.content = f"[Modified] {response.content}"

        return response


def compute_content_entropy(content: str) -> float:
    """Computes empirical Shannon entropy from response content token distribution."""
    if not content:
        return 0.1000
    words = content.split()
    if not words:
        return 0.1000
    total = len(words)
    counts = Counter(words)
    shannon = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_shannon = math.log2(total) if total > 1 else 1.0
    norm = shannon / max_shannon if max_shannon > 0 else 0.0
    return round(min(0.5000, max(0.0500, norm * 0.35 + 0.05)), 4)


class TraceObservabilityMiddleware(TantraMiddleware):
    async def __call__(
        self,
        request: TantraRequest,
        context: RequestContext,
        call_next: Callable[[TantraRequest, RequestContext], Awaitable[TantraResponse]]
    ) -> TantraResponse:
        # 1. PRE-PROCESS: Inject Trace ID if missing in request/context
        if not request.trace_id:
            request.trace_id = context.trace_id
        print(f"[Trace] Initializing observability for Trace: {request.trace_id}")

        # Start timer to measure actual latency
        start_time = time.time()

        # 2. CALL NEXT
        response = await call_next(request, context)

        # Measure actual latency
        latency_ms = (time.time() - start_time) * 1000

        # Store latency inside context metadata for routing/cost calculations later
        context.metadata["latency_ms"] = latency_ms

        # 3. POST-PROCESS: Compute dynamic telemetry metrics based on real content
        response.entropy_score = compute_content_entropy(response.content)

        if response.entropy_score < 0.16:
            response.confidence_level = "High"
        elif response.entropy_score < 0.23:
            response.confidence_level = "Medium"
        else:
            response.confidence_level = "Low"

        print(f"[Trace] Trace {response.trace_id} completed in {latency_ms:.2f}ms with confidence {response.confidence_level} (Entropy: {response.entropy_score})")
        return response


# ── Personality (from personality.py) ───────────────────────────────────────

AUTO_CUES = {
    "DirectAssertive": ["just give me", "final answer", "short"],
    "MentorBuilder": ["how do i", "i'm unsure", "can you guide"],
    "CriticalChallenger": ["are you sure", "prove", "seems wrong"],
    "CreativeExplorer": ["ideas", "alternatives", "brainstorm", "creative"],
}

OVERRIDES = {
    "mode: direct": "DirectAssertive",
    "mode: mentor": "MentorBuilder",
    "mode: critical": "CriticalChallenger",
    "mode: creative": "CreativeExplorer",
}


class PersonalityLayer:
    """Stub: selects mode (auto + overrides) and parameterizes decoding/prefixes."""

    def __init__(self, config: Dict):
        self.config = config
        self.default_mode = "DirectAssertive"

    def select_mode(self, user_text: str) -> str:
        lt = user_text.lower()
        for k, v in OVERRIDES.items():
            if k in lt:
                return v
        for mode, cues in AUTO_CUES.items():
            if any(c in lt for c in cues):
                return mode
        return self.default_mode

    def parameterize(self, mode: str) -> Dict:
        tones = self.config.get("tones", {})
        params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "prompt_prefix": tones.get("default", {}).get("prompt_prefix", ""),
        }
        mapping = {
            "DirectAssertive": "concise",
            "MentorBuilder": "mentor",
            "CriticalChallenger": "critical",
            "CreativeExplorer": "creative",
        }
        tone = mapping.get(mode, "default")
        preset = tones.get(tone, tones.get("default", {}))
        params["prompt_prefix"] = preset.get("prompt_prefix", params["prompt_prefix"])
        return params


class SafetyModule:
    """Evaluate safety and return action (pass/modify/deny)."""

    def __init__(self):
        self.deny_patterns = [
            r"illegal", r"harmful", r"violent", r"explosive", r"weapon",
            r"drug", r"suicide", r"hack", r"virus", r"malware"
        ]
        self.warn_patterns = [
            r"hate", r"discrimination", r"racism", r"sexism", r"bigotry"
        ]

    def evaluate(self, draft: str, context: Dict) -> Dict:
        """Return action (pass/modify/deny) + reasons."""

        # Check deny list
        for pattern in self.deny_patterns:
            if re.search(pattern, draft, re.IGNORECASE):
                return {"action": "deny", "reasons": [f"Matches dangerous pattern: {pattern}"]}

        # Check warn list (toxicity)
        for pattern in self.warn_patterns:
            if re.search(pattern, draft, re.IGNORECASE):
                return {"action": "modify", "reasons": [f"Matches warning pattern: {pattern}"]}

        # Pass if safe
        return {"action": "pass", "reasons": []}


class StyleModule:
    """Stub interface for style decisions and decoding parameter hints."""

    def __init__(self):
        pass

    def forward(self, values: Dict[str, float], context: Dict) -> Dict:
        return {}


class ValuesModule:
    """Stub interface for computing value scores from context."""

    def __init__(self):
        pass

    def forward(self, context: Dict) -> Dict[str, float]:
        return {}
