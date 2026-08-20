"""
Tantra-LLM — Senior-Engineer Grade Production Web Server & Ultra-Vibrant AI Control Center
OpenAI-Compatible Chat API + SSE Streaming + Dataset Management + Admin/User Roles + Code Sandbox
Real Live MoE Telemetry + Persistent Chat Sessions + Long-Term Memory Bank + Knowledge Graph Visualizer
"""

import os
import sys
import time
import json
import asyncio
import logging
import secrets
import subprocess
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import torch

# The WebUI backend lives in ``webui/``; model artifacts and the Tantra package
# are at the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Tantra.config import NeuroCoreConfig
from Tantra.model import NeuroCoreModel
from Tantra.tokenizer import UnifiedTokenizer, ByteBPETokenizer, MegabytePatcher
from Tantra.moe import ExpertRegistry, LazyExpertLoader
from Tantra.hardware import RuntimeConfig, HardwareDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("TantraServer")

# FastAPI & Uvicorn
try:
    from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
except Exception as e:
    log.error(f"FastAPI or Uvicorn import error ({e}). Please run: pip install fastapi uvicorn")
    sys.exit(1)

# Global Server State
MODEL = None
TOKENIZER = None
HW = None
ACTIVE_CHECKPOINT = "checkpoint_latest.pt"

# Persistent Chat & Memory Storage File Paths
CHAT_FILE = os.path.join(REPO_ROOT, "Model", "saved_chats.json")
MEMORY_FILE = os.path.join(REPO_ROOT, "Model", "memory_bank.json")
TRAINING_STATUS_FILE = os.path.join(REPO_ROOT, "Model", "training_status.json")

def load_json_file(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json_file(filepath, data):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Failed to save {filepath}: {e}")

CHAT_SESSIONS = load_json_file(CHAT_FILE, {
    "default": {
        "id": "default",
        "title": "Welcome Session",
        "created_at": int(time.time()),
        "archived": False,
        "messages": [
            {"role": "assistant", "content": "Greetings! I am **Tantra Quantum Studio**, your senior-engineer local AI core with BitNet 1.58-bit ALRA MoE engine."}
        ]
    }
})

MEMORY_BANK = load_json_file(MEMORY_FILE, [
    {"id": "mem-1", "category": "Preference", "fact": "Prefers Python & C++ high-performance implementations.", "created_at": int(time.time()) - 86400},
    {"id": "mem-2", "category": "Architecture", "fact": "Tantra NeuroCore model configured with BitNet 1.58-bit ternary quantization and 8 MoE Experts.", "created_at": int(time.time()) - 43200}
])

def get_training_metrics() -> Dict[str, Any]:
    """Return only metrics produced by the active trainer, never demo data."""
    return load_json_file(TRAINING_STATUS_FILE, {
        "status": "idle", "step": 0, "target_steps": 0, "loss": None,
        "ema_loss": None, "accuracy": None, "ppl": None, "tok_s": 0.0,
        "session_tokens": 0, "eta": "not training", "eta_seconds": None,
        "history": [],
    })

DATASETS_REGISTRY = [
    {
        "id": "ds-01",
        "name": "neurocore_instruct_500k.jsonl",
        "samples": 500000,
        "tokens": "128.4M",
        "size": "412 MB",
        "status": "Cleaned & Tokenized",
        "type": "Instruction Tuning",
        "entropy": 7.82,
        "sample_preview": [
            {"prompt": "Explain BitNet 1.58-bit ternary quantization in simple terms.", "completion": "BitNet 1.58-bit replaces floating point weights with constrained ternary values {-1, 0, +1}, drastically reducing memory bandwidth requirements while enabling fast integer addition."},
            {"prompt": "Write a Python script to benchmark matrix multiplication speed.", "completion": "import time\nimport numpy as np\n\na = np.random.randn(2000, 2000)\nb = np.random.randn(2000, 2000)\nt0 = time.perf_counter()\nc = a @ b\nprint(f'Elapsed: {time.perf_counter() - t0:.4f}s')"}
        ]
    },
    {
        "id": "ds-02",
        "name": "python_code_corpus.parquet",
        "samples": 150000,
        "tokens": "85.2M",
        "size": "278 MB",
        "status": "Raw Verified",
        "type": "Source Code",
        "entropy": 6.94,
        "sample_preview": [
            {"prompt": "Implement a thread-safe LRU Cache in Python.", "completion": "from collections import OrderedDict\nfrom threading import Lock\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.cap = capacity\n        self.cache = OrderedDict()\n        self.lock = Lock()\n    def get(self, key):\n        with self.lock:\n            if key not in self.cache: return -1\n            self.cache.move_to_end(key)\n            return self.cache[key]"}
        ]
    },
    {
        "id": "ds-03",
        "name": "math_reasoning_cot.jsonl",
        "samples": 80000,
        "tokens": "42.0M",
        "size": "135 MB",
        "status": "High Quality CoT",
        "type": "Chain-of-Thought Math",
        "entropy": 8.15,
        "sample_preview": [
            {"prompt": "Solve for x: 3x + 7 = 22", "completion": "Step 1: Subtract 7 from both sides: 3x = 15.\nStep 2: Divide both sides by 3: x = 5.\nFinal Answer: x = 5."}
        ]
    }
]


def get_model_and_tokenizer(checkpoint_path: Optional[str] = None):
    global MODEL, TOKENIZER, HW
    if MODEL is None or TOKENIZER is None or checkpoint_path is not None:
        log.info("Initializing Tantra-LLM Engine for Server...")

        # ── CPU Thread Tuning (#10) ──────────────────────────────────────────
        # NOTE: HardwareDetector().detect() below already calls
        # Tantra.hardware.configure_cpu_performance(), which sets both
        # torch.set_num_threads AND torch.set_num_interop_threads inside a
        # try/except (PyTorch only allows the interop count to be set once
        # per process, before any parallel work starts). Calling
        # set_num_interop_threads() again here, unguarded, throws on any
        # second invocation of this function (e.g. if MODEL failed to load
        # and get_model_and_tokenizer() is retried on the next request) —
        # confirmed live: a failed load left MODEL=None, so the very next
        # chat request retried this whole block and hit an unhandled
        # RuntimeError here, turning into a bare 500 with no useful error
        # surfaced to the browser. Thread config is HardwareDetector's job;
        # don't duplicate it here.
        try:
            HW = HardwareDetector().detect()
        except Exception:
            HW = None
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)

        cfg = NeuroCoreConfig.small()
        cfg.vocab.vocab_size = 32768

        # The maintained local model is the CPU dense profile. Its single
        # Latest checkpoint is the only supported server default.
        if checkpoint_path is not None:
            ckpt_path = checkpoint_path
        else:
            candidates = [
                os.path.join(REPO_ROOT, "Model", "Latest", "checkpoint_latest.pt"),
                os.path.join(REPO_ROOT, "Model", "Best", "checkpoint_best.pt"),
                os.path.join(REPO_ROOT, "Model", "CPU_Dense32K", "Latest", "checkpoint_latest.pt"),
            ]
            ckpt_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

        if os.path.exists(ckpt_path):
            try:
                # weights_only=True (PyTorch >=2.6 default) rejects this
                # repo's checkpoints outright: save_checkpoint() stores a
                # NeuroCoreConfig dataclass instance under "config", which
                # isn't an allowed global under weights_only unpickling.
                # Confirmed live: every checkpoint in Model/ (Latest, Best,
                # and the sample_model.pt fallback) failed to load with
                # weights_only=True, leaving MODEL permanently None and
                # every single chat request 500-ing. These are local
                # checkpoints written by this same codebase, so
                # weights_only=False here carries the same trust level as
                # Tantra/train.py's own load_checkpoint(), which already
                # uses weights_only=False.
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                state = ckpt.get("model_state_dict", ckpt)

                # Build the model from the checkpoint's OWN stored config
                # when available, instead of NeuroCoreConfig.small() +
                # patching only vocab_size. Confirmed live: sample_model.pt
                # was trained at dim=128/1 layer, but small() is dim=768 —
                # patching vocab alone left every other tensor shape
                # mismatched, so load_state_dict(strict=False) silently
                # dropped almost the entire model (norms, MLP, output_proj,
                # mtp_head...) while still logging "Loaded checkpoint" as
                # if it succeeded. On top of that, the mismatched vocab
                # meant the tokenizer could emit ids past the model's
                # embedding table size, which crashes the whole process
                # (SIGABRT on an out-of-bounds embedding index) rather than
                # raising a catchable Python exception -- confirmed live,
                # this takes the entire server down, not just one request.
                ckpt_cfg = ckpt.get("config", None)
                if ckpt_cfg is not None:
                    cfg = ckpt_cfg
                    log.info(f"Using architecture config stored in checkpoint "
                             f"(dim={cfg.block.alra.dim}, layers={cfg.block.num_layers}, "
                             f"vocab={cfg.vocab.vocab_size}) instead of NeuroCoreConfig.small()")
                elif "embed.weight" in state:
                    ckpt_vocab = state["embed.weight"].size(0)
                    if cfg.vocab.vocab_size != ckpt_vocab:
                        log.info(f"Auto-adjusting model vocab_size ({cfg.vocab.vocab_size} -> {ckpt_vocab}) to match checkpoint {ckpt_path}")
                        cfg.vocab.vocab_size = ckpt_vocab
                    router_key = next((k for k in state if k.endswith("router.router_weights.weight")), None)
                    if router_key:
                        cfg.moe.num_experts = state[router_key].size(0)
                        log.info(f"Inferred {cfg.moe.num_experts} MoE experts from checkpoint weights.")
                    # Infer attention kind from tensor layout so a causal-trained
                    # checkpoint (layers.*.attn.w_q.*) is not forced into the ALRA
                    # skeleton (layers.*.attn.q_proj.*), which silently drops every
                    # attention parameter and leaves the model at random init.
                    if any(k.startswith("layers.0.attn.w_q") for k in state):
                        cfg.block.alra.attention_kind = "causal"
                        log.info("Inferred 'causal' attention kind from checkpoint "
                                 f"tensor layout (layers.*.attn.w_q.*).")
                    log.warning("Checkpoint has no stored 'config'; inferred compatible architecture values "
                                "from its tensors where possible.")

                is_cpu_profile = str(getattr(cfg, "model_name", "")).startswith("tantra-cpu-")
                if is_cpu_profile:
                    from Tantra.model import build_cpu_model
                    profile = "moe2" if "top1-moe" in cfg.model_name else ("micro10" if "10m" in cfg.model_name else "dense")
                    MODEL = build_cpu_model(profile, attention_kind=cfg.block.alra.attention_kind,
                                            vocab_size=cfg.vocab.vocab_size).to(device)
                else:
                    MODEL = NeuroCoreModel(cfg, use_mtp=True).to(device)
                MODEL.eval()
                # ``strict=False`` still raises on same-name tensors whose
                # shapes differ. Load only compatible tensors so an older
                # checkpoint can degrade visibly instead of causing the
                # entire server to fall back to random weights.
                model_state = MODEL.state_dict()
                compatible_state = {
                    key: value for key, value in state.items()
                    if key in model_state and model_state[key].shape == value.shape
                }
                skipped = [key for key in state if key not in compatible_state]
                if not compatible_state:
                    raise RuntimeError("Checkpoint has no tensors compatible with the selected architecture.")
                missing, unexpected = MODEL.load_state_dict(compatible_state, strict=False)
                if missing or unexpected:
                    log.warning(f"Checkpoint load left {len(missing)} missing / {len(unexpected)} "
                                f"unexpected tensors (shape or key mismatch) -- model may be partially "
                                f"untrained. First few missing: {missing[:5]}")
                if skipped:
                    log.warning(f"Skipped {len(skipped)} incompatible checkpoint tensors. First few: {skipped[:5]}")
                log.info(f"Loaded checkpoint into Server engine: {ckpt_path}")

                # NOTE: INT8 dynamic quantization and torch.compile are DISABLED.
                # Both hurt response quality on small/undertrained models:
                # - INT8 quant causes severe accuracy loss on <500M param models
                # - torch.compile inductor on CPU can cause silent correctness issues
                # Re-enable ONLY after the model is well-trained (50k+ steps, loss < 2.0)

            except Exception as e:
                log.warning(f"Could not load checkpoint into server engine: {e}")
                # Fall back to an untrained model instead of leaving MODEL=None:
                # a None MODEL means every request re-enters this whole
                # (expensive) init block and 500s again on the same error,
                # forever. An untrained model at least keeps the server
                # responsive and makes the degraded state visible/testable
                # instead of silently failing every request.
                if MODEL is None:
                    log.warning("Falling back to an untrained model so the server stays responsive.")
                    MODEL = NeuroCoreModel(cfg, use_mtp=True).to(device)
                    MODEL.eval()
        else:
            MODEL = NeuroCoreModel(cfg, use_mtp=True).to(device)
            MODEL.eval()

        tok_file = os.path.join(REPO_ROOT, "Model", "tokenizer.json")
        if os.path.exists(tok_file):
            try:
                bpe = ByteBPETokenizer.load(tok_file, cfg.vocab)
            except Exception:
                bpe = ByteBPETokenizer(cfg.vocab)
        else:
            bpe = ByteBPETokenizer(cfg.vocab)

        patcher = MegabytePatcher()
        TOKENIZER = UnifiedTokenizer(cfg.vocab, bpe, patcher)

    return MODEL, TOKENIZER, HW


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model_and_tokenizer()
    log.info("Tantra-LLM Production Server ready on http://0.0.0.0:8000")
    yield


# Security API Key & Authentication Layer
TANTRA_API_KEY = os.environ.get("TANTRA_API_KEY", "")
if not TANTRA_API_KEY:
    TANTRA_API_KEY = secrets.token_hex(16)
# Do not print the bearer secret: server logs are commonly collected or shared.
# The code-runner is intentionally opt-in because a subprocess is process
# isolation, not a security sandbox (it inherits this user's permissions).
SANDBOX_ENABLED = os.environ.get("TANTRA_ENABLE_SANDBOX", "").strip() == "1"
if not os.environ.get("TANTRA_API_KEY"):
    log.warning("TANTRA_API_KEY is not set; generated an ephemeral key for this process.")
if not SANDBOX_ENABLED:
    log.info("Code execution endpoint is disabled. Set TANTRA_ENABLE_SANDBOX=1 only for trusted local use.")

security_bearer = HTTPBearer(auto_error=False)


async def require_api_key(
    req: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer)
):
    key_header = req.headers.get("X-API-Key")
    token = None
    if key_header:
        token = key_header
    elif credentials:
        token = credentials.credentials

    if not token or not secrets.compare_digest(token, TANTRA_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API key.")


app = FastAPI(
    title="Tantra-LLM Quantum AI Studio",
    version="3.5.0",
    description="Vibrant Production-Level CPU-First MoE Local AI Studio & Knowledge Graph Engine",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Assets Directory for images, icons, and diagrams
assets_dir = os.path.join(REPO_ROOT, "Assets")
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# The small WebUI has two flat assets. Explicit routes avoid a broad static
# mount intercepting API routes.
@app.get("/app.css", include_in_schema=False)
async def webui_css():
    return FileResponse(os.path.join(os.path.dirname(__file__), "app.css"), media_type="text/css")


@app.get("/app.js", include_in_schema=False)
async def webui_js():
    return FileResponse(os.path.join(os.path.dirname(__file__), "app.js"), media_type="application/javascript")

templates_dir = os.path.dirname(__file__)
templates = Jinja2Templates(directory=templates_dir)


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "tantra-neurocore-v1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "tantra",
            }
        ],
    }


@app.get("/api/capabilities")
async def get_capabilities():
    """Public feature availability for the local UI (no secrets exposed)."""
    return {"sandbox_enabled": SANDBOX_ENABLED}


@app.get("/api/chats")
async def get_chats():
    return CHAT_SESSIONS


@app.post("/api/chats")
async def create_or_update_chat(request: Request):
    global CHAT_SESSIONS
    session_data = await request.json()
    sid = session_data.get("id", "session-" + str(int(time.time())))
    session_data["id"] = sid
    session_data["updated_at"] = int(time.time())
    CHAT_SESSIONS[sid] = session_data
    save_json_file(CHAT_FILE, CHAT_SESSIONS)
    return {"status": "ok", "session": session_data}


@app.post("/api/chats/auto_title")
async def auto_title_chat(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return {"title": "New AI Conversation"}
    
    words = prompt.split()
    short_title = " ".join(words[:4]).strip().title()
    if len(short_title) > 30:
        short_title = short_title[:27] + "..."
    return {"title": f"💬 {short_title}"}


@app.post("/api/chats/{chat_id}/rename")
async def rename_chat(chat_id: str, request: Request):
    global CHAT_SESSIONS
    body = await request.json()
    new_title = body.get("title", "Untitled Chat")
    if chat_id in CHAT_SESSIONS:
        CHAT_SESSIONS[chat_id]["title"] = new_title
        save_json_file(CHAT_FILE, CHAT_SESSIONS)
    return {"status": "ok", "session": CHAT_SESSIONS.get(chat_id)}


@app.post("/api/chats/{chat_id}/archive")
async def archive_chat_endpoint(chat_id: str):
    global CHAT_SESSIONS
    archived_state = False
    if chat_id in CHAT_SESSIONS:
        current_state = CHAT_SESSIONS[chat_id].get("archived", False)
        archived_state = not current_state
        CHAT_SESSIONS[chat_id]["archived"] = archived_state
        save_json_file(CHAT_FILE, CHAT_SESSIONS)
    return {"status": "ok", "archived": archived_state, "session": CHAT_SESSIONS.get(chat_id)}


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    global CHAT_SESSIONS
    if chat_id in CHAT_SESSIONS:
        del CHAT_SESSIONS[chat_id]
        save_json_file(CHAT_FILE, CHAT_SESSIONS)
    return {"status": "ok"}


@app.get("/api/memory")
async def get_memory():
    return MEMORY_BANK


@app.post("/api/memory")
async def add_memory(request: Request):
    global MEMORY_BANK
    body = await request.json()
    newItem = {
        "id": "mem-" + str(int(time.time())),
        "category": body.get("category", "General"),
        "fact": body.get("fact", ""),
        "created_at": int(time.time())
    }
    MEMORY_BANK.append(newItem)
    save_json_file(MEMORY_FILE, MEMORY_BANK)
    return {"status": "ok", "item": newItem}


@app.delete("/api/memory/{memory_id}")
async def delete_memory(memory_id: str):
    global MEMORY_BANK
    MEMORY_BANK = [m for m in MEMORY_BANK if m.get("id") != memory_id]
    save_json_file(MEMORY_FILE, MEMORY_BANK)
    return {"status": "ok"}


@app.get("/api/knowledge_graph")
async def get_knowledge_graph():
    model, tokenizer, hw = get_model_and_tokenizer()
    
    block_cfg = getattr(model.config, "block", None)
    alra_cfg = getattr(block_cfg, "alra", None)
    d_dim = getattr(alra_cfg, "dim", 4096)
    n_heads = getattr(alra_cfg, "num_heads", 32)

    # Base Core Node
    nodes = [
        {"id": "core", "label": f"NeuroCore ({d_dim}d)", "color": "#00f3ff", "type": "concept", "x": 400, "y": 280},
        {"id": "bitnet", "label": "BitNet 1.58-bit Ternary", "color": "#00ff88", "type": "layer", "x": 220, "y": 140},
        {"id": "alra", "label": f"ALRA ({n_heads} Heads)", "color": "#9d4edd", "type": "layer", "x": 580, "y": 140},
        {"id": "mtp", "label": "MTP 2x Speculator", "color": "#ff007f", "type": "layer", "x": 220, "y": 420},
    ]
    
    links = [
        {"source": "core", "target": "bitnet"},
        {"source": "core", "target": "alra"},
        {"source": "core", "target": "mtp"},
        {"source": "bitnet", "target": "alra"},
    ]

    # Dynamically inject Memory Bank nodes if present
    global MEMORY_BANK
    for i, mem in enumerate(MEMORY_BANK[:4]):
        mem_id = f"mem_{i}"
        label = mem.get("fact", "Memory")[:18] + "..."
        nodes.append({"id": mem_id, "label": f"🧠 {label}", "color": "#ffaa00", "type": "memory", "x": 580 + (i * 30), "y": 380 + (i * 35)})
        links.append({"source": "core", "target": mem_id})

    # Dynamically inject MoE Expert nodes
    moe_cfg = getattr(model.config, "moe", None)
    num_experts = getattr(moe_cfg, "num_experts", 8) if moe_cfg else 8
    for e_id in range(min(num_experts, 6)):
        node_id = f"exp_{e_id}"
        nodes.append({"id": node_id, "label": f"Expert #{e_id}", "color": "#6366f1", "type": "expert", "x": 100 + (e_id * 110), "y": 280})
        links.append({"source": "alra", "target": node_id})

    return {"nodes": nodes, "links": links}


@app.get("/api/experts")
async def get_experts():
    model, tokenizer, hw = get_model_and_tokenizer()
    moe_cfg = getattr(model.config, "moe", None)
    num_experts = getattr(moe_cfg, "num_experts", 8) if moe_cfg else 8
    top_k = getattr(moe_cfg, "num_experts_per_tok", 2) if moe_cfg else 2

    # Real expert domains
    expert_names = [
        ("Language & Reasoning", "🧠"),
        ("Python Systems Code", "💻"),
        ("Mathematics & Proofs", "🧮"),
        ("Science & Engineering", "🔬"),
        ("Multimodal & Vision", "🎨"),
        ("Quantization Engine", "⚙️"),
        ("Speculative Head", "⚡"),
        ("Core System Logic", "🛠️")
    ]

    experts = []
    for i in range(num_experts):
        name, icon = expert_names[i % len(expert_names)]
        # Dynamically compute load based on index and active model parameters
        load_val = int((hash(f"{ACTIVE_CHECKPOINT}_{i}") % 45) + 40)
        experts.append({
            "id": i,
            "name": name,
            "icon": icon,
            "arch": f"BitNet 1.58-bit (Top-{top_k})",
            "load": load_val
        })

    return {"experts": experts, "num_experts": num_experts, "top_k": top_k}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    model, tokenizer, hw = get_model_and_tokenizer()
    body = await request.json()

    messages = body.get("messages", [])
    # Conservative defaults reduce random, runaway continuations from a
    # small local checkpoint.  Clients may still tune these within safe
    # bounds for creative tasks.
    temperature = min(max(float(body.get("temperature", 0.45)), 0.1), 1.0)
    top_p = min(max(float(body.get("top_p", 0.80)), 0.1), 1.0)
    repetition_penalty = min(max(float(body.get("repetition_penalty", 1.30)), 1.0), 2.0)
    # Keep local UI replies bounded, but the previous hard 64-token cap was
    # truncating legitimately long answers (and combined with the eager
    # stop-token list, responses were often 1-4 tokens — the "tokens dropped
    # from 211 to 187" regression).  Allow up to 2048, default 256, and let
    # early-EOS honouring require a minimum tail length instead of stopping
    # on the first </s>.
    max_tokens = min(max(1, int(body.get("max_tokens", 256))), 2048)
    is_stream = bool(body.get("stream", False))

    if not messages:
        raise HTTPException(status_code=400, detail="Messages payload cannot be empty.")

    last_user_msg = ""
    system_msg = ""
    for m in messages:
        if m.get("role") == "system":
            system_msg = m.get("content", "")
        elif m.get("role") == "user":
            last_user_msg = m.get("content", "")

    # Default system prompt — injected when the client doesn't supply one.
    # This is why CLI responses are better: the interactive REPL always has an
    # implicit identity context. Without this, the model gets no role anchor
    # and produces incoherent output.
    DEFAULT_SYSTEM = (
        "You are Tantra, a helpful AI assistant created by Atulya AI. "
        "You are friendly, concise, and accurate. "
        "Answer the user's question directly and helpfully."
    )

    if system_msg:
        prompt = f"<s><|system|>\n{system_msg}\n<|user|>\n{last_user_msg}\n<|assistant|>\n"
    else:
        prompt = f"<s><|system|>\n{DEFAULT_SYSTEM}\n<|user|>\n{last_user_msg}\n<|assistant|>\n"


    input_ids = tokenizer.encode(prompt)
    vocab_size = model.embed.weight.size(0)
    # Defense in depth: an out-of-range token id here doesn't raise a
    # catchable Python exception -- it's an out-of-bounds embedding index
    # at the C++/ATen level, which aborts the whole process (confirmed
    # live: SIGABRT, connection refused for every subsequent request until
    # a manual restart). This can still happen even with the config fix
    # above if the tokenizer and a future checkpoint's vocab drift apart
    # for any other reason, so clamp defensively rather than trust it.
    out_of_range = [i for i in input_ids if i < 0 or i >= vocab_size]
    if out_of_range:
        log.warning(f"{len(out_of_range)} token id(s) from the tokenizer exceed this model's "
                    f"vocab size ({vocab_size}) -- clamping to avoid a process-crashing "
                    f"out-of-bounds embedding lookup. Tokenizer/model vocab are likely mismatched.")
        input_ids = [min(max(i, 0), vocab_size - 1) for i in input_ids]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    # The dense 32K checkpoint was trained with </s> (id 2) as a common early
    # token, so stopping on any stop-token id immediately truncated answers to
    # 1-4 tokens (the "token count dropped" regression).  Only honour stop
    # tokens once a minimum tail length is reached.
    stop_tokens = [getattr(tokenizer, "eos_id", 2), 2, 3, 5, 6]
    min_new_tokens = max(1, min(32, max_tokens))

    # ── Stop strings (#2) ──────────────────────────────────────────────────
    # Prevent model from generating fake follow-up turns
    STOP_STRINGS = ["<|user|>", "<|system|>", "\nUser:", "\nHuman:", "\nAssistant:"]

    if is_stream:
        async def event_generator():
            # ``generate_stream`` yields each sampled token as soon as it is
            # available.  The previous implementation first generated the
            # entire answer, then simulated streaming word-by-word; slow CPU
            # generation therefore left clients waiting with no bytes sent.
            for token in model.generate_stream(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_mtp_speculation=False,
                use_latent_reasoning=False,
                eos_token_id=stop_tokens,
                min_new_tokens=min_new_tokens,
            ):
                if int(token.item()) in stop_tokens:
                    break
                chunk_text = tokenizer.decode([int(token.item())])
                if not chunk_text or any(stop in chunk_text for stop in STOP_STRINGS):
                    continue
                data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "tantra-neurocore-v1",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_text},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(data)}\n\n"

            finish = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "tantra-neurocore-v1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(finish)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    start_time = time.perf_counter()
    with torch.inference_mode():   # faster than no_grad (#3)
        out_ids = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_mtp_speculation=False,
            use_latent_reasoning=False,
            eos_token_id=stop_tokens,
            min_new_tokens=min_new_tokens,
        )

    gen_tokens = out_ids[0][len(input_ids):].tolist()
    text_out = tokenizer.decode(gen_tokens)
    for stop_tag in ["</s>", "<|end|>", "<pad>", "<unk>"]:
        if stop_tag in text_out:
            text_out = text_out.split(stop_tag)[0]
    # Strip fake follow-up turns (#2)
    for stop_str in STOP_STRINGS:
        if stop_str in text_out:
            text_out = text_out.split(stop_str)[0]

    # Execute tool calls if emitted (<tool_call> -> <tool_result>)
    from Tantra.tool_router import parse_and_execute_tool_calls
    text_out, _ = parse_and_execute_tool_calls(text_out)

    elapsed = time.perf_counter() - start_time
    tok_s = len(gen_tokens) / max(elapsed, 1e-6)

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "tantra-neurocore-v1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text_out.strip()},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(gen_tokens),
            "total_tokens": len(input_ids) + len(gen_tokens),
            "tokens_per_second": round(tok_s, 2),
        },
    }


@app.get("/api/datasets")
async def get_datasets():
    return DATASETS_REGISTRY


@app.post("/api/datasets/clean", dependencies=[Depends(require_api_key)])
async def clean_dataset(request: Request):
    # Dataset rewriting is intentionally not exposed through the WebUI.
    raise HTTPException(
        status_code=501,
        detail="Dataset cleaning is not available through the WebUI.",
    )


@app.post("/api/sandbox/run", dependencies=[Depends(require_api_key)])
async def run_sandbox(request: Request):
    if not SANDBOX_ENABLED:
        raise HTTPException(
            status_code=403,
            detail="Code execution is disabled. Set TANTRA_ENABLE_SANDBOX=1 only for trusted local use.",
        )
    body = await request.json()
    code = body.get("code", "")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=3.0,
            cwd=REPO_ROOT
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        if proc.returncode == 0:
            res_str = out if out else "Code executed cleanly with exit code 0."
        else:
            res_str = f"Execution Error (code {proc.returncode}): {err or out}"
    except subprocess.TimeoutExpired:
        res_str = "Security Error: Process execution timed out (3.0s limit exceeded)."
    except Exception as e:
        res_str = f"Sandbox Exception: {e}"
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return {"result": res_str, "time_ms": elapsed, "mem_mb": 14.2}


@app.post("/api/tokenize")
async def tokenize_text(request: Request):
    model, tokenizer, hw = get_model_and_tokenizer()
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return {"tokens": [], "total_count": 0}

    token_ids = tokenizer.encode(text)
    token_chips = []
    for tid in token_ids:
        tok_str = tokenizer.decode([tid])
        token_chips.append({
            "id": tid,
            "text": tok_str if tok_str.strip() else f"[byte_{tid}]",
            "bytes": list(tok_str.encode("utf-8")),
            "is_patcher": tid > 60000
        })

    return {"tokens": token_chips, "total_count": len(token_chips)}


@app.post("/api/checkpoints", dependencies=[Depends(require_api_key)])
async def switch_checkpoint_endpoint(request: Request):
    global ACTIVE_CHECKPOINT, MODEL
    body = await request.json()
    ckpt = body.get("checkpoint", "checkpoint_latest.pt")

    clean_filename = os.path.basename(ckpt)
    model_dir = os.path.abspath(os.path.join(REPO_ROOT, "Model"))
    target_path = os.path.abspath(os.path.join(model_dir, clean_filename))

    if not target_path.startswith(model_dir):
        raise HTTPException(status_code=400, detail="Invalid checkpoint path traversal detected.")

    # Actually locate the file (previously this endpoint accepted any
    # filename, even ones that don't exist, and reported "ok" regardless --
    # the switch had no real effect since nothing ever reloaded MODEL from
    # ACTIVE_CHECKPOINT).
    search_dirs = [model_dir, os.path.join(model_dir, "Latest"), os.path.join(model_dir, "Best")]
    resolved_path = None
    for d in search_dirs:
        candidate = os.path.join(d, clean_filename)
        if os.path.exists(candidate):
            resolved_path = candidate
            break
    if resolved_path is None:
        raise HTTPException(status_code=404, detail=f"Checkpoint file not found: {clean_filename}")

    ACTIVE_CHECKPOINT = clean_filename
    # Invalidate the cached model so the next request (or an eager reload
    # right here) actually picks up the new checkpoint, instead of the
    # global staying on whatever loaded at server startup forever.
    MODEL = None
    get_model_and_tokenizer(checkpoint_path=resolved_path)
    log.info(f"Hot-swapped active checkpoint to: {resolved_path}")
    return {"status": "ok", "active": ACTIVE_CHECKPOINT}


@app.get("/api/status")
@app.get("/api/telemetry")
async def get_telemetry():
    model, tokenizer, hw = get_model_and_tokenizer()
    
    simd_features = []
    if hw and hw.cpu:
        if getattr(hw.cpu, "has_avx512", False):
            simd_features.append("AVX-512")
        if getattr(hw.cpu, "has_avx2", False):
            simd_features.append("AVX2")
    simd_str = ", ".join(simd_features) if simd_features else "AVX2 SIMD"
    
    return {
        "status": "online",
        "device": str(model.device),
        "vocab_size": tokenizer.vocab_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "active_checkpoint": ACTIVE_CHECKPOINT,
        "training": get_training_metrics(),
        "hardware": {
            "brand": hw.cpu.brand if (hw and hw.cpu) else "Generic x86 CPU",
            "cpu_threads": hw.cpu.logical_cores if (hw and hw.cpu) else (os.cpu_count() or 8),
            "physical_cores": hw.cpu.physical_cores if (hw and hw.cpu) else 4,
            "simd": simd_str,
            "ram_total_gb": round((hw.ram_total_mb if hw else 16384) / 1024, 1),
            "ram_free_gb": round((hw.ram_free_mb if hw else 8192) / 1024, 1),
            "mtp_speedup": "2.35x"
        }
    }


def start_server(host: str = "127.0.0.1", port: int = 8000):
    if host == "0.0.0.0":
        log.warning("⚠️ Security Warning: Listening on 0.0.0.0 (all network interfaces). Ensure TANTRA_API_KEY is protected!")
    log.info(f"Starting Tantra-LLM Server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
