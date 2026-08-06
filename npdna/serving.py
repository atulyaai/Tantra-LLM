"""Combined serving module for NP-DNA / Atulya Tantra.

This module consolidates the CLI entrypoints, the Web Studio (Gradio),
and the FastAPI-based REST API server into a single file.  Import or
run whichever surface you need – the logic is unchanged.
"""

from __future__ import annotations

import os
import sys
import asyncio
import argparse
import hmac
import subprocess
import time
import json
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .model import NpDnaCore
from npdna.inference import (
    UnifiedInferenceHub,
    NpDnaAdapter,
    RWKVAdapter,
    GeminiAdapter,
    OpenAIAdapter,
)
from npdna.schema import TantraRequest, TantraResponse, Message, ModelProvider, infer_max_tokens

logger = logging.getLogger(__name__)

# ── Project root on sys.path so we can import npdna ──────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


###############################################################################
# CLI – npdna/cli.py
###############################################################################

DEFAULT_CHECKPOINTS = (
    Path("model/latest"),
)


def _ensure_utf8() -> None:
    if sys.stdout.encoding.lower() != "utf-8":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def info_main() -> None:
    _ensure_utf8()
    core = NpDnaCore.from_config("seed")
    print("NP-DNA seed configuration")
    print(f"  hidden_size  = {core.config.hidden_size}")
    print(f"  layers       = {core.config.num_layers}")
    print(f"  initial_vocab = {core.config.initial_vocab}")
    print(f"  max_vocab    = {core.config.max_vocab}")
    print(f"  parameters   = {core.model.parameter_count():,}")


def chat_main() -> None:
    _ensure_utf8()
    parser = argparse.ArgumentParser(description="Generate text with NP-DNA.")
    parser.add_argument("prompt", nargs="?", default=None, help="Prompt to generate from.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory.")
    parser.add_argument("--lora-adapter", default=None,
                        help="Optional LoRA adapter .pt file to apply after loading the checkpoint.")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="Rank used by --lora-adapter (must match the saved adapter).")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.35,
        help="Sampling temperature. Lower values are faster to stabilize on weak checkpoints.",
    )
    parser.add_argument("--top-k", type=int, default=30, help="Top-k sampling limit.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling limit.")
    parser.add_argument(
        "--context-window",
        type=int,
        default=256,
        help="Recent tokens to recompute during generation. Smaller is faster.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens. Defaults to an automatic prompt-based cap.",
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="Start a terminal chat loop.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else next(
        (path for path in DEFAULT_CHECKPOINTS if path.exists()),
        DEFAULT_CHECKPOINTS[-1],
    )
    if checkpoint.exists():
        core = NpDnaCore.load(checkpoint)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        core = NpDnaCore.from_config("seed")
        print("Loaded fresh seed model; no checkpoint found.")

    if args.lora_adapter:
        from .model import inject_lora, load_lora_adapter

        inject_lora(core.model, rank=args.lora_rank)
        load_lora_adapter(core.model, args.lora_adapter)
        print(f"Loaded LoRA adapter: {args.lora_adapter}")

    if args.interactive or args.prompt is None:
        print("Type /exit or /quit to stop.")
        while True:
            try:
                prompt = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.lower() in {"/exit", "/quit"}:
                break
            if not prompt:
                continue
            max_tokens = args.max_tokens or infer_max_tokens(prompt)
            print(
                "npdna>",
                core.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    context_window=args.context_window,
                ),
            )
        return

    max_tokens = args.max_tokens or infer_max_tokens(args.prompt)
    print(
        core.generate(
            args.prompt,
            max_tokens=max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            context_window=args.context_window,
        )
    )


if __name__ == "__main__":
    chat_main()


###############################################################################
# Studio – npdna/studio.py
###############################################################################

# ── Hub + adapters (deferred / best-effort) ───────────────────────────────────
_hub = None
_adapter_status: dict[str, str] = {}

def _init_hub():
    global _hub
    if _hub is not None:
        return _hub
    try:
        from npdna.inference import UnifiedInferenceHub
        _hub = UnifiedInferenceHub()
    except Exception as e:
        logger.warning(f"[Studio] UnifiedInferenceHub unavailable: {e}")
        _hub = None

    # Local NP-DNA
    try:
        from npdna.inference import NpDnaAdapter
        from npdna.schema import ModelProvider
        ckpt = os.environ.get("NPDNA_CHECKPOINT", "model/latest")
        adapter = NpDnaAdapter(checkpoint_path=ckpt)
        if _hub:
            _hub.register_adapter(ModelProvider.LOCAL, adapter)
        _adapter_status["Local NP-DNA"] = "✅ loaded"
    except Exception as e:
        _adapter_status["Local NP-DNA"] = f"⚠️ {str(e)[:60]}"

    # Gemini
    try:
        from npdna.inference import GeminiAdapter
        from npdna.schema import ModelProvider
        if _hub:
            _hub.register_adapter(ModelProvider.GEMINI, GeminiAdapter())
        _adapter_status["Gemini"] = "✅ loaded"
    except Exception as e:
        _adapter_status["Gemini"] = f"⚠️ {str(e)[:60]}"

    # OpenAI
    try:
        from npdna.inference import OpenAIAdapter
        from npdna.schema import ModelProvider
        if _hub:
            _hub.register_adapter(ModelProvider.OPENAI, OpenAIAdapter())
        _adapter_status["OpenAI"] = "✅ loaded"
    except Exception as e:
        _adapter_status["OpenAI"] = f"⚠️ {str(e)[:60]}"

    return _hub


# ── Generation ────────────────────────────────────────────────────────────────
def studio_generate(prompt: str, provider: str, temperature: float, top_p: float, files):
    """Synchronous wrapper for the Gradio fn= callback."""
    if not prompt.strip():
        return "⚠️ Please enter a prompt.", "—", "—", "—"

    hub = _init_hub()
    if hub is None:
        return ("⚠️ Inference hub not available. "
                "Check that npdna is installed correctly."), "—", "—", "—"

    try:
        from npdna.schema import TantraRequest, Message, ModelProvider
        _map = {"Local NP-DNA": ModelProvider.LOCAL,
                "Gemini": ModelProvider.GEMINI,
                "OpenAI": ModelProvider.OPENAI}
        provider_enum = _map.get(provider, ModelProvider.LOCAL)
        if (
            provider_enum != ModelProvider.LOCAL
            and os.environ.get("NPDNA_ENABLE_CLOUD_PROVIDERS") != "1"
        ):
            return "⚠️ Cloud providers are disabled.", "—", "—", "—"

        req = TantraRequest(
            messages=[Message(role="user", content=prompt)],
            provider=provider_enum,
            temperature=temperature,
            top_p=top_p,
        )

        t0 = time.perf_counter()
        response = asyncio.run(hub.execute(req))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        content = getattr(response, "content", str(response))
        confidence = f"{getattr(response, 'confidence', 0.0):.3f}"
        latency = f"{getattr(response, 'latency_ms', elapsed_ms):.1f} ms"
        entropy = f"{getattr(response, 'entropy', 0.0):.4f}"
        return content, confidence, latency, entropy

    except Exception:
        logger.exception("Studio generation failed")
        return "❌ Generation failed. Check server logs.", "—", "—", "—"


# ── Benchmark ─────────────────────────────────────────────────────────────────
def run_benchmark(ckpt_path: str):
    """Run eval_npdna.py and capture output."""
    checkpoint = _model_artifact_path(ckpt_path or "model/latest")
    if checkpoint is None or not checkpoint.is_dir():
        return "⚠️ Checkpoint must be an existing directory inside model/."
    script = _ROOT / "tools" / "eval_npdna.py"
    if not script.exists():
        return "⚠️ tools/eval_npdna.py not found."
    cmd = [sys.executable, str(script), "--checkpoint", str(checkpoint)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        out = result.stdout + ("\n" + result.stderr if result.stderr else "")
        return out.strip() or "✅ No output (benchmark may need a checkpoint)."
    except subprocess.TimeoutExpired:
        return "⏱️ Benchmark timed out after 5 minutes."
    except Exception as e:
        return f"❌ {e}"


# ── Memory Cortex Inspector ───────────────────────────────────────────────────
def load_cortex_stats(cortex_path: str):
    path = _model_artifact_path(cortex_path.strip()) if cortex_path.strip() else None
    if not path or not path.exists():
        return "⚠️ Cortex file must exist inside model/."
    try:
        import torch
        data = torch.load(path, map_location="cpu", weights_only=True)
        entries = data.get("entries", [])
        if not entries:
            return "Cortex is empty."
        topics = {}
        for e in entries:
            t = e.get("topic", "(none)")
            topics[t] = topics.get(t, 0) + 1
        lines = [f"📚 Cortex entries: {len(entries)}",
                 f"📂 Config: {data.get('config', {})}",
                 "", "Topic breakdown:"]
        for t, c in sorted(topics.items(), key=lambda x: -x[1])[:20]:
            lines.append(f"  {t}: {c}")
        return "\n".join(lines)
    except Exception:
        logger.exception("Cortex inspection failed")
        return "❌ Failed to load cortex. Check server logs."


# ── Health ────────────────────────────────────────────────────────────────────
def get_health():
    _init_hub()
    lines = ["## Adapter Status\n"]
    for name, status in _adapter_status.items():
        lines.append(f"- **{name}**: {status}")
    lines += ["", "## Python Environment"]
    lines.append(f"- Python: {sys.version.split()[0]}")
    try:
        import torch
        lines.append(f"- PyTorch: {torch.__version__}")
        lines.append(f"- CUDA available: {torch.cuda.is_available()}")
        lines.append(f"- CPU threads: {torch.get_num_threads()}")
    except ImportError:
        lines.append("- PyTorch: not available")
    try:
        import psutil
        vm = psutil.virtual_memory()
        lines.append(f"- RAM: {vm.used // (1024**2)} MB / {vm.total // (1024**2)} MB ({vm.percent:.1f}%)")
    except ImportError:
        pass
    return "\n".join(lines)


# ── App builder ───────────────────────────────────────────────────────────────
_CSS = """
body { font-family: 'Inter', sans-serif; }
.gr-button-primary { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important; }
.gr-textbox textarea { font-size: 14px; }
.metric-box { padding: 8px 12px; border-radius: 8px; background: rgba(99,102,241,0.1); }
footer { display: none !important; }
"""

def build_app():
    import gradio as gr
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(theme=theme, css=_CSS, title="Atulya Tantra — Studio") as app:
        gr.Markdown(
            "# 🧠 Atulya Tantra — Studio\n"
            "*Multimodal · Local-first · CPU-optimised*"
        )

        # ── Tab 1: Chat ────────────────────────────────────────────────────
        with gr.Tab("🧠 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Ask Atulya anything…",
                        lines=5,
                        elem_id="chat-prompt",
                    )
                    files = gr.File(
                        label="Attach files (image / audio)",
                        file_count="multiple",
                        file_types=["image", "audio"],
                        elem_id="chat-files",
                    )
                    with gr.Row():
                        provider = gr.Dropdown(
                            choices=["Local NP-DNA", "Gemini", "OpenAI"],
                            value="Local NP-DNA",
                            label="Provider",
                            scale=2,
                        )
                        temp = gr.Slider(0.0, 1.5, value=0.7, step=0.05,
                                         label="Temperature", scale=2)
                        top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05,
                                          label="Top-P", scale=2)
                    gen_btn = gr.Button("⚡ Generate", variant="primary", size="lg")

                with gr.Column(scale=4):
                    output = gr.Textbox(label="Response", lines=14, elem_id="chat-output",
                                        show_copy_button=True)
                    with gr.Row():
                        confidence_box = gr.Textbox(label="Confidence", elem_classes="metric-box")
                        latency_box = gr.Textbox(label="Latency", elem_classes="metric-box")
                        entropy_box = gr.Textbox(label="Entropy", elem_classes="metric-box")

            gen_btn.click(
                fn=studio_generate,
                inputs=[prompt, provider, temp, top_p, files],
                outputs=[output, confidence_box, latency_box, entropy_box],
            )

        # ── Tab 2: Benchmark ───────────────────────────────────────────────
        with gr.Tab("📊 Benchmark"):
            gr.Markdown("### One-click perplexity & throughput evaluation")
            with gr.Row():
                ckpt_input = gr.Textbox(
                    value="model/latest",
                    label="Checkpoint path",
                    scale=4,
                )
                bench_btn = gr.Button("▶ Run Benchmark", variant="primary", scale=1)
            bench_out = gr.Textbox(label="Results", lines=20, show_copy_button=True)
            bench_btn.click(fn=run_benchmark, inputs=[ckpt_input], outputs=[bench_out])

        # ── Tab 3: Memory Cortex ───────────────────────────────────────────
        with gr.Tab("💾 Memory (Cortex)"):
            gr.Markdown(
                "### MemoryCortex Inspector\n"
                "The Cortex provides unlimited factual memory without retraining. "
                "1M entries ≈ 100B params of knowledge."
            )
            with gr.Row():
                cortex_path_input = gr.Textbox(
                    value="model/latest/cortex/cortex.pt",
                    label="Cortex checkpoint path (.pt)",
                    scale=4,
                )
                cortex_btn = gr.Button("🔍 Inspect", variant="secondary", scale=1)
            cortex_out = gr.Textbox(label="Cortex Stats", lines=20)
            cortex_btn.click(fn=load_cortex_stats, inputs=[cortex_path_input],
                             outputs=[cortex_out])

        # ── Tab 4: System Health ───────────────────────────────────────────
        with gr.Tab("❤️ System"):
            gr.Markdown("### Live system health & adapter status")
            health_btn = gr.Button("🔄 Refresh", variant="secondary")
            health_out = gr.Markdown()
            health_btn.click(fn=get_health, inputs=[], outputs=[health_out])
            app.load(fn=get_health, inputs=[], outputs=[health_out])

    return app


# ── Entry point ───────────────────────────────────────────────────────────────
def studio_main():
    p = argparse.ArgumentParser(description="Atulya Tantra Web Studio")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (use 0.0.0.0 for LAN access)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a public Gradio link")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    print(f"\n  🧠 Atulya Tantra Studio  →  http://{args.host}:{args.port}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Checkpoint env: NPDNA_CHECKPOINT={os.environ.get('NPDNA_CHECKPOINT','model/latest')}")
    print(f"  Share link:     {args.share}")
    print()

    _init_hub()   # pre-warm adapters
    app = build_app()
    studio_auth = None
    is_local_host = args.host in {"127.0.0.1", "localhost", "::1"}
    if args.share or not is_local_host:
        username = os.environ.get("NPDNA_STUDIO_USERNAME")
        password = os.environ.get("NPDNA_STUDIO_PASSWORD")
        if not username or not password:
            raise SystemExit(
                "Network-accessible Studio requires NPDNA_STUDIO_USERNAME and NPDNA_STUDIO_PASSWORD."
            )
        studio_auth = (username, password)

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=studio_auth,
        show_error=True,
    )


if __name__ == "__main__":
    studio_main()


###############################################################################
# API Server – npdna/api_server.py
###############################################################################

app = FastAPI(title="Tantra-LLM API Server", version="1.0.0")


def _model_artifact_path(value: str) -> Optional[Path]:
    """Return a resolved path only when it stays inside the local model directory."""
    try:
        candidate = Path(value).resolve()
        candidate.relative_to((_ROOT / "model").resolve())
    except (OSError, ValueError):
        return None
    return candidate


def _inference_concurrency() -> int:
    """Keep a public-facing CPU model from being exhausted by parallel requests."""
    try:
        requested = int(os.environ.get("NPDNA_MAX_CONCURRENT_REQUESTS", "1"))
    except ValueError:
        requested = 1
    return max(1, min(requested, 16))


_INFERENCE_SLOTS = asyncio.Semaphore(_inference_concurrency())
_REQUEST_BUCKETS: dict[str, tuple[float, int]] = {}
_REQUEST_BUCKET_LOCK = asyncio.Lock()


def _request_limit() -> int:
    try:
        requested = int(os.environ.get("NPDNA_RATE_LIMIT_PER_MINUTE", "60"))
    except ValueError:
        requested = 60
    return max(1, min(requested, 10_000))


async def _allow_request(client_key: str) -> bool:
    """Apply a bounded, process-local fixed-window rate limit."""
    now = time.monotonic()
    window_start = now - 60.0
    async with _REQUEST_BUCKET_LOCK:
        for key, (started, _) in list(_REQUEST_BUCKETS.items()):
            if started < window_start:
                del _REQUEST_BUCKETS[key]
        started, count = _REQUEST_BUCKETS.get(client_key, (now, 0))
        if started < window_start:
            started, count = now, 0
        if count >= _request_limit():
            return False
        _REQUEST_BUCKETS[client_key] = (started, count + 1)
        return True


def _is_loopback_client(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in {"127.0.0.1", "::1", "localhost"}


@app.middleware("http")
async def protect_api(request: Request, call_next):
    """Require a key for network clients and limit every inference request."""
    if request.url.path in {"/", "/health", "/docs", "/openapi.json"}:
        return await call_next(request)

    expected = os.environ.get("NPDNA_API_KEY")
    supplied = request.headers.get("X-API-Key", "")
    client_host = request.client.host if request.client else "unknown"
    if expected:
        if not hmac.compare_digest(supplied, expected):
            return JSONResponse(status_code=401, content={"detail": "Valid X-API-Key required."})
        client_key = f"network:{client_host}"
    elif not _is_loopback_client(request):
        return JSONResponse(
            status_code=503,
            content={"detail": "Set NPDNA_API_KEY before accepting network clients."},
        )
    else:
        client_key = "loopback"

    if not await _allow_request(client_key):
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded."})
    return await call_next(request)

# Initialize Inference Hub
hub = UnifiedInferenceHub()

# NP-DNA is the real trained local model. RWKV remains an optional fallback.
try:
    npdna_checkpoint = os.environ.get("NPDNA_CHECKPOINT", "model/latest")
    hub.register_adapter(ModelProvider.LOCAL, NpDnaAdapter(checkpoint_path=npdna_checkpoint))
except Exception as e:
    print(f"Failed to register local NP-DNA adapter: {e}")

try:
    rwkv_model_path = os.environ.get("TANTRA_RWKV", "models/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth")
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
    role: str = Field(pattern="^(system|user|assistant|tool)$")
    content: str = Field(min_length=1, max_length=8_192)

class GenerateRequestSchema(BaseModel):
    messages: List[MessageSchema] = Field(min_length=1, max_length=32)
    provider: Optional[str] = "local"
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, gt=0.0, le=1.0)
    trace_id: Optional[str] = Field(default=None, max_length=128)


class OpenAIChatRequest(BaseModel):
    """Subset of the OpenAI chat-completions request contract supported locally."""
    messages: List[MessageSchema] = Field(min_length=1, max_length=32)
    model: str = Field(default="npdna-local", max_length=128)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=500)
    stream: bool = False


def _provider_from_name(name: Optional[str]) -> ModelProvider:
    return {
        "gemini": ModelProvider.GEMINI,
        "openai": ModelProvider.OPENAI,
    }.get((name or "local").lower(), ModelProvider.LOCAL)


def _require_cloud_access(provider: ModelProvider, api_key: Optional[str]) -> None:
    """Keep paid cloud adapters opt-in and authenticated at the HTTP boundary."""
    if provider == ModelProvider.LOCAL:
        return
    if os.environ.get("NPDNA_ENABLE_CLOUD_PROVIDERS") != "1":
        raise HTTPException(status_code=403, detail="Cloud providers are disabled.")
    expected = os.environ.get("NPDNA_API_KEY")
    if not expected or not api_key or not hmac.compare_digest(api_key, expected):
        raise HTTPException(status_code=401, detail="Valid X-API-Key required for cloud providers.")


def _tantra_request(
    messages: List[MessageSchema],
    provider: Optional[str] = "local",
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 0.9,
    trace_id: Optional[str] = None,
) -> TantraRequest:
    return TantraRequest(
        messages=[Message(role=m.role, content=m.content) for m in messages],
        provider=_provider_from_name(provider),
        temperature=temperature,
        top_p=top_p,
        trace_id=trace_id,
    )

@app.get("/")
async def root():
    return {"status": "online", "message": "Tantra-LLM Cognitive Brain API is operational."}

@app.get("/health")
async def health():
    health_status = {}
    for provider in ModelProvider:
        adapter = hub.get_active_adapter(provider)
        if adapter:
            try:
                health_status[provider.value] = adapter.health_check()
            except Exception:
                health_status[provider.value] = False
        else:
            health_status[provider.value] = False
    return {"status": "ok", "adapters": health_status}

@app.post("/generate")
async def generate(req: GenerateRequestSchema, x_api_key: Optional[str] = Header(default=None)):
    try:
        tantra_req = _tantra_request(**req.model_dump())
        _require_cloud_access(tantra_req.provider or ModelProvider.LOCAL, x_api_key)
        async with _INFERENCE_SLOTS:
            response = await hub.execute(tantra_req)
        return response
    except HTTPException:
        raise
    except Exception:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail="Generation failed.")

@app.post("/generate_stream")
async def generate_stream(req: GenerateRequestSchema, x_api_key: Optional[str] = Header(default=None)):
    tantra_req = _tantra_request(**req.model_dump())
    provider_enum = tantra_req.provider or ModelProvider.LOCAL
    _require_cloud_access(provider_enum, x_api_key)

    if not tantra_req.trace_id:
        tantra_req.trace_id = f"TRC-{uuid.uuid4().hex[:8]}"

    async def event_generator():
        async with _INFERENCE_SLOTS:
            adapter = hub.get_active_adapter(provider_enum)
            if not adapter:
                yield f"data: {json.dumps({'error': 'Adapter unavailable'})}\n\n"
                return

            try:
                async for chunk in adapter.stream(tantra_req):
                    yield f"data: {json.dumps({'content': chunk.content, 'model': chunk.model, 'trace_id': tantra_req.trace_id})}\n\n"
                hub._record_success(adapter)
            except Exception:
                hub._record_failure(adapter)
                logger.exception("Streaming generation failed")
                yield f"data: {json.dumps({'error': 'Generation failed.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(req: OpenAIChatRequest):
    """OpenAI-compatible local chat endpoint for existing clients and UIs."""
    tantra_req = _tantra_request(
        req.messages,
        provider="local",
        temperature=req.temperature,
        top_p=req.top_p,
        trace_id=f"chatcmpl-{uuid.uuid4().hex}",
    )
    model_name = req.model or "npdna-local"

    if req.stream:
        async def openai_events():
            async with _INFERENCE_SLOTS:
                adapter = hub.get_active_adapter(ModelProvider.LOCAL)
                if not adapter:
                    yield "data: " + json.dumps({"error": {"message": "Local adapter unavailable"}}) + "\n\n"
                    return
                try:
                    async for chunk in adapter.stream(tantra_req):
                        payload = {
                            "id": tantra_req.trace_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": chunk.content}, "finish_reason": None}],
                        }
                        yield "data: " + json.dumps(payload) + "\n\n"
                    yield "data: [DONE]\n\n"
                except Exception:
                    logger.exception("OpenAI-compatible streaming failed")
                    yield "data: " + json.dumps({"error": {"message": "Generation failed."}}) + "\n\n"

        return StreamingResponse(openai_events(), media_type="text/event-stream")

    try:
        async with _INFERENCE_SLOTS:
            response = await hub.execute(tantra_req)
    except Exception as exc:
        logger.exception("OpenAI-compatible generation failed")
        raise HTTPException(status_code=500, detail="Generation failed.") from exc
    return {
        "id": tantra_req.trace_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response.content},
            "finish_reason": "stop",
        }],
        "usage": response.usage,
    }


def serve_main() -> None:
    """Run the local OpenAI-compatible server (requires ``npdna[api]``)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install API support with: pip install -e '.[api]'") from exc

    uvicorn.run(
        "npdna.serving:app",
        host=os.environ.get("NPDNA_HOST", "127.0.0.1"),
        port=int(os.environ.get("NPDNA_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    serve_main()

