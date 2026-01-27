from typing import Dict, List, Optional, Callable, Any
from atulya_core.protocol.adapter import BaseTantraAdapter
from atulya_core.schema.models import TantraRequest, TantraResponse, ModelProvider
from atulya_core.protocol.middleware import TantraMiddleware
import asyncio
import uuid
import sys

# Nervous System Link
sys.path.append(r"d:\Atulya Tantra\Tantra-Bus")
from src.nervous_system import AtulyaNervousSystem

class UnifiedInferenceHub:
    """
    Production-level Inference Hub for the Atulya Tantra OS-Organism.
    Strictly handles model orchestration and middleware injection.
    No planning or memory logic allowed here.
    """
    def __init__(self, bus: Optional[AtulyaNervousSystem] = None):
        self.adapters: Dict[ModelProvider, BaseTantraAdapter] = {}
        self.middlewares: List[TantraMiddleware] = []
        self.bus = bus

    def register_adapter(self, provider: ModelProvider, adapter: BaseTantraAdapter):
        self.adapters[provider] = adapter

    def add_middleware(self, middleware: TantraMiddleware):
        self.middlewares.append(middleware)

    async def execute(self, request: TantraRequest) -> TantraResponse:
        provider = request.provider or self._route(request)
        adapter = self.adapters.get(provider) or self.adapters.get(ModelProvider.LOCAL)
        
        if not adapter:
            raise RuntimeError("Primary and Fallback Adapters unavailable.")

        async def _core_call(req: TantraRequest) -> TantraResponse:
            return await adapter.generate(req)

        # Middleware Chain
        handler = _core_call
        for mw in reversed(self.middlewares):
            def wrap(m, h):
                return lambda r: m(r, h)
            handler = wrap(mw, handler)

        # 4. Nervous system pulse (Async)
        if self.bus:
            asyncio.create_task(self.bus.emit("inference_start", {"trace_id": request.trace_id}))

        response = await handler(request)

        if self.bus:
            asyncio.create_task(self.bus.emit("inference_complete", {
                "trace_id": response.trace_id,
                "cost": response.cost,
                "entropy": response.entropy_score
            }))

        return response

    def _route(self, request: TantraRequest) -> ModelProvider:
        # Routing logic should be dynamic but minimalist
        return ModelProvider.LOCAL
