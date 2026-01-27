from atulya_core.protocol.middleware import TantraMiddleware
from atulya_core.schema.models import TantraRequest, TantraResponse
from typing import Callable
import uuid

class TraceObservabilityMiddleware(TantraMiddleware):
    async def __call__(self, request: TantraRequest, call_next: Callable) -> TantraResponse:
        # 1. PRE-PROCESS: Inject Trace ID if missing
        if not request.trace_id:
            request.trace_id = f"TRC-{uuid.uuid4().hex[:8]}"
        print(f"[Trace] Initializing observability for Trace: {request.trace_id}")
        
        # 2. CALL NEXT
        response = await call_next(request)
        
        # 3. POST-PROCESS: Add entropy profiling (stub)
        response.entropy_score = 0.12 # Reality check
        response.confidence_level = "High"
        print(f"[Trace] Trace {response.trace_id} completed with confidence {response.confidence_level}")
        return response
