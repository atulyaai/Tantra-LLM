from typing import Dict, List, Optional, Callable
from tantra_core.protocol.adapter import BaseTantraAdapter
from tantra_core.schema.models import TantraRequest, TantraResponse, ModelProvider
from tantra_core.protocol.middleware import TantraMiddleware

class UnifiedHub:
    def __init__(self):
        self.adapters: Dict[ModelProvider, BaseTantraAdapter] = {}
        self.middlewares: List[TantraMiddleware] = []

    def register_adapter(self, provider: ModelProvider, adapter: BaseTantraAdapter):
        self.adapters[provider] = adapter

    def add_middleware(self, middleware: TantraMiddleware):
        self.middlewares.append(middleware)

    async def execute(self, request: TantraRequest) -> TantraResponse:
        # Determine adapter
        provider = request.provider or self.auto_route(request)
        adapter = self.adapters.get(provider)
        if not adapter:
            raise ValueError(f"No adapter found for provider {provider}")

        # Middleware Chain (Onion)
        async def call_adapter(req: TantraRequest) -> TantraResponse:
            return await adapter.generate(req)

        # Build the chain from the end back to the start
        current_call = call_adapter
        for middleware in reversed(self.middlewares):
            def create_next(mw=middleware, nxt=current_call):
                return lambda req: mw(req, nxt)
            current_call = create_next()

        return await current_call(request)

    def auto_route(self, request: TantraRequest) -> ModelProvider:
        # Simplistic routing for now
        return ModelProvider.LOCAL
