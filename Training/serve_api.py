import json
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from Training.agent import Agent, ToolSpec
from Training.memory import Memory
from Training.tools_basic import TOOL_SPECS
from Training.model_runtime import TextRuntime


app = FastAPI()


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_runtime: TextRuntime = None


def ensure_runtime() -> TextRuntime:
    global _runtime
    if _runtime is None:
        serve_cfg = load_yaml("Config/serve.yaml")
        tok = serve_cfg.get("paths", {}).get("tokenizer", "Model/tokenizer.json")
        wts = serve_cfg.get("paths", {}).get("weights", "Model/tantra_weights.safetensors")
        _runtime = TextRuntime(tok, wts, device="cpu")
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


