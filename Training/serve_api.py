import json
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

try:
    from Training.agent import Agent, ToolSpec
    from Training.memory import Memory
    from Training.tools_basic import TOOL_SPECS
    from Training.model_runtime import TextRuntime
    from Training.mamba_runtime import MambaRuntime
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    # Define minimal fallbacks
    class Agent:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, prompt):
            return [], "Service temporarily unavailable"
    
    class ToolSpec:
        def __init__(self, *args, **kwargs):
            pass
    
    class Memory:
        def __init__(self, *args, **kwargs):
            pass
        def build_context(self, text):
            return ""
        def observe_turn(self, *args, **kwargs):
            pass
    
    TOOL_SPECS = {}
    
    class TextRuntime:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, prompt, gen):
            return "Model not available"
        def stream(self, prompt, gen):
            yield "Model not available"
    
    class MambaRuntime:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, prompt, gen):
            return "Model not available"
        def stream(self, prompt, gen):
            yield "Model not available"


app = FastAPI()


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {path} not found, using defaults")
        return {}
    except Exception as e:
        print(f"Warning: Error loading {path}: {e}, using defaults")
        return {}


_runtime = None


def ensure_runtime():
    global _runtime
    if _runtime is None:
        try:
            serve_cfg = load_yaml("Config/serve.yaml")
            arch = serve_cfg.get("inference", {}).get("architecture", "mamba")
            tok = serve_cfg.get("paths", {}).get("tokenizer", "Model/tokenizer.json")
            wts = serve_cfg.get("paths", {}).get("weights", "Model/tantra_weights.safetensors")
            
            if arch == "mamba":
                mcfg = serve_cfg.get("model", {"d_model": 512, "n_layers": 8, "d_state": 32, "d_conv": 4, "dropout": 0.1})
                _runtime = MambaRuntime(tok, wts, mcfg)
            else:
                _runtime = TextRuntime(tok, wts, device="cpu")
        except Exception as e:
            print(f"Error initializing runtime: {e}")
            # Create a fallback runtime
            class FallbackRuntime:
                def generate(self, prompt, gen):
                    return "Runtime initialization failed. Please check model files."
                def stream(self, prompt, gen):
                    yield "Runtime initialization failed. Please check model files."
            _runtime = FallbackRuntime()
    return _runtime


def build_agent() -> Agent:
    agent_cfg = load_yaml("Config/agent.yaml")
    memory = Memory(agent_cfg)
    tools = {name: ToolSpec(name, spec["schema"], spec["func"], spec.get("destructive", False)) for name, spec in TOOL_SPECS.items()}

    def llm_infer(prompt: str, gen: Dict[str, Any]) -> str:
        rt = ensure_runtime()
        return rt.generate(prompt, gen)

    return Agent(agent_cfg, tools, memory, llm_infer)


agent = build_agent()


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/infer")
async def infer(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    traces, final_text = agent.run(prompt)
    return {"text": final_text, "traces": traces}


@app.post("/chat/stream")
async def chat_stream(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    gen = body.get("gen", {"max_tokens": 256, "temperature": 0.7, "top_p": 0.9})

    rt = ensure_runtime()

    async def gen_sse():
        for chunk in rt.stream(prompt, gen):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(gen_sse(), media_type="text/event-stream")

@app.post("/agent/stream")
async def agent_stream(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")

    traces, final_text = agent.run(prompt)

    async def gen():
        yield json.dumps({"type": "traces", "data": traces}) + "\n"
        yield json.dumps({"type": "final", "data": final_text}) + "\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/memory/flush")
async def memory_flush():
    agent.memory.short_history.clear()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    import sys
    from pathlib import Path
    
    # Add workspace to Python path
    workspace_root = Path(__file__).parent.parent
    sys.path.insert(0, str(workspace_root))
    
    cfg = load_yaml("Config/serve.yaml")
    host = cfg.get("server", {}).get("host", "0.0.0.0")
    port = int(cfg.get("server", {}).get("port", 8000))
    uvicorn.run("Training.serve_api:app", host=host, port=port, reload=False)


