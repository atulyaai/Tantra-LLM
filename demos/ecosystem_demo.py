import asyncio
import sys
import os

# Ensure we can find the packages
sys.path.append(r"d:\Atulya Tantra\Tantra-Core")
sys.path.append(r"d:\Atulya Tantra\Tantra-LLM")
sys.path.append(r"d:\Atulya Tantra\Tantra-Kosha")
sys.path.append(r"d:\Atulya Tantra\Tantra-Trace")

from tantra_core.schema.models import TantraRequest, TantraMessage, ModelProvider
from core.hub import UnifiedHub
from adapters.openai.adapter import OpenAIAdapter
from adapters.gemini.adapter import GeminiAdapter
from tantra_kosha.middleware.optimizer import KoshaOptimizerMiddleware
from tantra_trace.middleware.observability import TraceObservabilityMiddleware
from tantra_smriti.middleware.memory import SmritiMemoryMiddleware
from tantra_raksha.middleware.guardrails import RakshaGuardrailMiddleware

async def main():
    print("--- Atulya Tantra: Interlinked Ecosystem Demo ---")
    
    # 1. Setup Hub
    hub = UnifiedHub()
    
    # 2. Register Adapters
    hub.register_adapter(ModelProvider.OPENAI, OpenAIAdapter(api_key="sk-stub"))
    hub.register_adapter(ModelProvider.GEMINI, GeminiAdapter(api_key="gemini-stub"))
    
    # 3. Inject Modular Middleware (The Multi-Repo Nervous System)
    hub.add_middleware(TraceObservabilityMiddleware()) # ID Generation
    hub.add_middleware(RakshaGuardrailMiddleware())     # Safety First
    hub.add_middleware(SmritiMemoryMiddleware())        # Memory Recall
    hub.add_middleware(KoshaOptimizerMiddleware())     # Cost Compression
    
    # 4. Create request
    request = TantraRequest(
        messages=[TantraMessage(role="user", content="How do I save tokens in the Tantra ecosystem?")],
        model="gpt-4o",
        provider=ModelProvider.OPENAI
    )
    
    # 5. Execute
    response = await hub.execute(request)
    
    print(f"\n[Final Response] {response.content}")
    print(f"[Metadata] Cost: {response.cost}$, TraceID: {response.trace_id}, Confidence: {response.confidence_level}")

if __name__ == "__main__":
    asyncio.run(main())
