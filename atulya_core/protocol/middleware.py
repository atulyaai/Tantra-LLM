from typing import Callable, Awaitable
from atulya_core.schema.models import TantraRequest, TantraResponse, RequestContext

class TantraMiddleware:
    """Base class for all Tantra middleware layers."""
    async def __call__(
        self, 
        request: TantraRequest, 
        context: RequestContext, 
        call_next: Callable[[TantraRequest, RequestContext], Awaitable[TantraResponse]]
    ) -> TantraResponse:
        return await call_next(request, context)
