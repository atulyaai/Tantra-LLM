import os
import sys
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.inference import UnifiedInferenceHub
from atulya_core.schema.models import TantraRequest, TantraResponse, Message, ModelProvider
from adapters.local.rwkv_adapter import RWKVAdapter
from adapters.gemini.adapter import GeminiAdapter
from adapters.openai.adapter import OpenAIAdapter

app = FastAPI(title="Tantra-LLM API Server", version="1.0.0")

# Initialize Inference Hub
hub = UnifiedInferenceHub()

# Register adapters
try:
    rwkv_model_path = os.environ.get("TANTRA_SPB", "models/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth")
    hub.register_adapter(ModelProvider.LOCAL, RWKVAdapter(model_path=rwkv_model_path))
except Exception as e:
    print(f"Failed to register local RWKV adapter: {e}")

try:
    hub.register_adapter(ModelProvider.GEMINI, GeminiAdapter())
except Exception as e:
    print(f"Failed to register Gemini adapter: {e}")

try:
    hub.register_adapter(ModelProvider.OPENAI, OpenAIAdapter())
except Exception as e:
    print(f"Failed to register OpenAI adapter: {e}")

# Request/Response Schemas
class MessageSchema(BaseModel):
    role: str
    content: str

class GenerateRequestSchema(BaseModel):
    messages: List[MessageSchema]
    provider: Optional[str] = "local"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    trace_id: Optional[str] = None

@app.get("/")
async def root():
    return {"status": "online", "message": "Tantra-LLM Cognitive Brain API is operational."}

@app.get("/health")
async def health():
    health_status = {}
    for provider, adapter in hub.adapters.items():
        try:
            health_status[provider.value] = adapter.health_check()
        except Exception:
            health_status[provider.value] = False
    return {"status": "ok", "adapters": health_status}

@app.post("/generate")
async def generate(req: GenerateRequestSchema):
    try:
        messages = [Message(role=m.role, content=m.content) for m in req.messages]
        provider_enum = ModelProvider.LOCAL
        if req.provider == "gemini":
            provider_enum = ModelProvider.GEMINI
        elif req.provider == "openai":
            provider_enum = ModelProvider.OPENAI
            
        tantra_req = TantraRequest(
            messages=messages,
            provider=provider_enum,
            temperature=req.temperature,
            top_p=req.top_p,
            trace_id=req.trace_id
        )
        
        response = await hub.execute(tantra_req)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_stream")
async def generate_stream(req: GenerateRequestSchema):
    messages = [Message(role=m.role, content=m.content) for m in req.messages]
    provider_enum = ModelProvider.LOCAL
    if req.provider == "gemini":
        provider_enum = ModelProvider.GEMINI
    elif req.provider == "openai":
        provider_enum = ModelProvider.OPENAI
        
    tantra_req = TantraRequest(
        messages=messages,
        provider=provider_enum,
        temperature=req.temperature,
        top_p=req.top_p,
        trace_id=req.trace_id
    )

    async def event_generator():
        adapter = hub.adapters.get(provider_enum) or hub.adapters.get(ModelProvider.LOCAL)
        if not adapter:
            yield f"data: {json.dumps({'error': 'Adapter unavailable'})}\n\n"
            return
            
        try:
            async for chunk in adapter.stream(tantra_req):
                yield f"data: {json.dumps({'content': chunk.content, 'model': chunk.model, 'trace_id': chunk.trace_id})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
