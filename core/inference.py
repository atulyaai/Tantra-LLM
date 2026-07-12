from typing import Dict, List, Optional, Callable, Any, Awaitable
import asyncio
import uuid
import time
import os
import json

from atulya_core.protocol.adapter import BaseTantraAdapter
from atulya_core.schema.models import TantraRequest, TantraResponse, ModelProvider, Message, RequestContext
from atulya_core.protocol.middleware import TantraMiddleware

# Import cognitive components & middleware
from core.dynamic_context import DynamicContextManager
from core.observability import TraceObservabilityMiddleware
from core.cognitive_middleware import PersonalityMiddleware, SafetyMiddleware

# Nervous System Link (vendored from Tantra-Bus)
try:
    from core.nervous_system import AtulyaNervousSystem
    _BUS_AVAILABLE = True
except ImportError:
    _BUS_AVAILABLE = False
    AtulyaNervousSystem = None


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
        candidates = self.adapters.get(provider, []) + self.adapters.get(ModelProvider.LOCAL, [])
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

        # Apply Dynamic Context trimming (sliding window) early
        if request.messages:
            meta = {"complexity": 0.5}
            target_limit = self.context_manager.select_window(meta)
            
            # Convert user message to pseudo token IDs to apply trim
            raw_content = request.messages[-1].content
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
            asyncio.create_task(self.bus.emit("inference_complete", {
                "trace_id": response.trace_id,
                "cost": response.cost,
                "entropy": response.entropy_score
            }))

        return response

    def _route(self, request: TantraRequest) -> ModelProvider:
        # Dynamic minimalist routing
        return ModelProvider.LOCAL
