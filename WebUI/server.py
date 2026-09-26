"""
WebUI/server.py — Local web app + OpenAI-compatible API for Tantra.

Start:  python main.py --mode serve      (then open http://127.0.0.1:8000)

Pages:  Chat (streaming with stop, saved chats, voice in/out), Training (live
        progress, loss + 50-question probe charts, start/stop, live log),
        Model (checkpoints, test/export jobs, hardware).
API:    POST /v1/chat/completions   (OpenAI format, stream or not)
        GET  /v1/models
        GET  /api/status            everything the dashboard shows, in one call
        POST /api/training/start|stop, /api/jobs/{eval|export}, GET /api/logs/{job}

Speech is optional and fully offline once installed:
  speech-to-text  Whisper (MIT license)       pip install openai-whisper
  text-to-speech  Kokoro-82M (Apache-2.0)     pip install kokoro soundfile
They sit behind /api/stt and /api/tts so our own speech models can replace
them later without UI changes.
"""
from __future__ import annotations

import asyncio
import glob
import importlib.util
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from Tantra.config import EOS_ID  # noqa: E402
from Tantra.dataset import chat_prompt  # noqa: E402
from Tantra.hardware import detect_hardware  # noqa: E402
from Tantra.model import load_model  # noqa: E402
from Tantra.tokenizer import load_tokenizer  # noqa: E402
from Tantra.utils import get_logger  # noqa: E402

log = get_logger("tantra.server")
MODEL_DIR = os.path.join(REPO_ROOT, "Model")
DATA_DIR = os.path.join(REPO_ROOT, "Datasets")
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_FILE = os.path.join(MODEL_DIR, "training_status.json")
PROBE_FILE = os.path.join(MODEL_DIR, "probe_history.jsonl")
CHATS_FILE = os.path.join(MODEL_DIR, "saved_chats.json")
STOP_FILE = os.path.join(MODEL_DIR, "STOP")
API_KEY = os.environ.get("TANTRA_API_KEY", "")   # set it to protect training/checkpoint actions

state: Dict[str, Any] = {"model": None, "tok": None, "ckpt": None, "info": {}, "hw": None, "jobs": {},
                         "load_error": None}
lock = threading.Lock()          # model loading
gen_lock = asyncio.Lock()        # one generation at a time (the model keeps per-request state)
chats_lock = threading.Lock()


def rel(path: str) -> str:
    """Path relative to the repo for display; absolute if on another drive (Windows)."""
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return os.path.abspath(path)


def _inside(path: str, folder: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(folder)]) == os.path.realpath(folder)
    except ValueError:   # different drives
        return False


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex[:6]}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def hardware():
    if state["hw"] is None:
        state["hw"] = detect_hardware("auto", verbose=False)
    return state["hw"]


# ── model ────────────────────────────────────────────────────────────────────

def checkpoints() -> List[str]:
    order = {"tantra.pt": 0, "best.pt": 1, "latest.pt": 2}
    return sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")), key=lambda p: order.get(os.path.basename(p), 9))


def checkpoint_details(path: str) -> Dict[str, Any]:
    st = os.stat(path)
    meta = _read_json(path + ".meta.json", {})
    val = meta.get("last_validation") or {}
    return {"path": rel(path), "name": os.path.basename(path), "size_mb": round(st.st_size / 2**20, 1),
            "modified": st.st_mtime, "step": meta.get("step_count"), "layers": meta.get("num_layers"),
            "val_loss": val.get("loss"), "tokens": meta.get("total_tokens")}


def load(path: Optional[str] = None, int8: Optional[bool] = None) -> None:
    with lock:
        hw = hardware()
        tok_path = os.path.join(MODEL_DIR, "tokenizer.json")
        if not os.path.isfile(tok_path):
            raise HTTPException(503, "Model/tokenizer.json missing. Run: python main.py --mode tokenizer")
        state["tok"] = load_tokenizer(tok_path)
        path = path or next(iter(checkpoints()), None)
        if path is None:
            raise HTTPException(503, "No trained model yet. Start training in the Training tab "
                                     "(or run: python main.py --mode train).")
        if int8 is None:
            int8 = os.environ.get("TANTRA_INT8") == "1"
        model, ckpt = load_model(path, hw.device, int8=int8)
        if model.embed.weight.shape[0] != state["tok"].vocab_size:
            raise HTTPException(409, f"{os.path.basename(path)} was trained with a different tokenizer.")
        state.update(model=model, ckpt=path, load_error=None, info={
            "checkpoint": rel(path), "step": ckpt.get("step_count", 0),
            "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "layers": len(model.layers), "dim": getattr(model, "dim", None),
            "vocab": state["tok"].vocab_size, "int8": bool(int8), "device": hw.device,
            "categories": list(model.category_layers.keys()),
            "val": ckpt.get("last_validation", {}) or {}, "tokens": ckpt.get("total_tokens"),
            "loaded_at": time.time(),
        })
        log.info(f"Serving {state['info']['checkpoint']} (step {state['info']['step']:,})")


def get_model():
    if state["model"] is None:
        try:
            load()
        except HTTPException as exc:
            state["load_error"] = exc.detail
            raise
    return state["model"], state["tok"]


def require_key(request: Request) -> None:
    if API_KEY:
        given = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").removeprefix("Bearer ")
        if not secrets.compare_digest(given, API_KEY):
            raise HTTPException(401, "Invalid API key")


app = FastAPI(title="Tantra", version="2.1")
if os.path.isdir(os.path.join(REPO_ROOT, "Assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(REPO_ROOT, "Assets")), name="assets")

_NO_CACHE = {"Cache-Control": "no-cache"}
_PUBLIC = ("/", "/app.js", "/app.css", "/manifest.webmanifest")


@app.middleware("http")
async def lan_guard(request: Request, call_next):
    """From another device (phone on Wi-Fi) every API call needs the key; this computer needs none."""
    host = request.client.host if request.client else ""
    local = host in ("127.0.0.1", "::1", "localhost", "testclient")
    if not local and request.url.path not in _PUBLIC and not request.url.path.startswith("/assets/"):
        given = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").removeprefix("Bearer ")
        if not API_KEY or not secrets.compare_digest(given, API_KEY):
            return JSONResponse({"detail": "Invalid API key"}, status_code=401)
    return await call_next(request)


@app.get("/", include_in_schema=False)
def index():
    """index.html with ?v=<file time> on app.css / app.js, so browsers never keep an old copy after an update."""
    from fastapi.responses import HTMLResponse
    with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as f:
        html = f.read()
    for name in ("app.css", "app.js"):
        v = int(os.path.getmtime(os.path.join(WEB_DIR, name)))
        html = html.replace(f'"/{name}"', f'"/{name}?v={v}"')
    return HTMLResponse(html, headers=_NO_CACHE)


@app.get("/app.js", include_in_schema=False)
def js():
    return FileResponse(os.path.join(WEB_DIR, "app.js"), media_type="application/javascript", headers=_NO_CACHE)


@app.get("/app.css", include_in_schema=False)
def css():
    return FileResponse(os.path.join(WEB_DIR, "app.css"), media_type="text/css", headers=_NO_CACHE)


# ── chat ─────────────────────────────────────────────────────────────────────

def build_prompt(messages: List[dict], history_turns: int = 3) -> str:
    system = next((m["content"] for m in messages if m.get("role") == "system" and m.get("content")), None)
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if not turns or turns[-1]["role"] != "user":
        raise HTTPException(400, "The last message must be from the user.")
    history, pending = [], None
    for m in turns[:-1]:
        if m["role"] == "user":
            pending = m["content"]
        elif pending is not None:
            history.append((pending, m["content"]))
            pending = None
    return chat_prompt(turns[-1]["content"], system, history[-history_turns:] if history_turns else [])


def pick_category(model, text: str, requested: Optional[str]) -> Optional[str]:
    if not model.category_layers or requested in ("none", "base"):
        return None
    if requested and requested != "auto":
        return requested if requested in model.category_layers else None
    from Tantra.adapters import AdapterRegistry, RequestRouter
    routed = RequestRouter(AdapterRegistry()).route(text)
    return routed if routed in model.category_layers else None


def smriti():
    """The knowledge store (Model/smriti.db), opened read-only; reopened when the file is rebuilt."""
    path = os.path.join(MODEL_DIR, "smriti.db")
    if not os.path.isfile(path):
        return None
    mtime = os.path.getmtime(path)
    cur = state.get("smriti")
    if cur is None or cur[0] != path or cur[1] != mtime:
        from Tantra.smriti import Smriti
        if cur:
            cur[2].close()
        s = Smriti(path, readonly=True)
        state["smriti"] = (path, mtime, s, s.stats())
    return state["smriti"][2]


def smriti_stats() -> Optional[dict]:
    return state["smriti"][3] if smriti() is not None else None


def with_knowledge(messages: List[dict], query: str, k: int) -> tuple:
    """Look the question up in Smriti and put the facts in front of it (as the system message)."""
    s = smriti()
    if s is None or k <= 0:
        return messages, []
    ctx, hits = s.context(query, k=k)
    if not ctx:
        return messages, []
    note = "नीचे दी गई जानकारी का उपयोग करके उत्तर दें / Use this information to answer:\n" + ctx
    rest = [m for m in messages if m.get("role") != "system"]
    system = next((m["content"] for m in messages if m.get("role") == "system" and m.get("content")), "")
    return [{"role": "system", "content": (system + "\n\n" if system else "") + note}] + rest, hits


# ── assistant layer: memory, documents, skills (all work without a trained model) ──

def memory():
    path = os.path.join(MODEL_DIR, "memory.json")
    if state.get("memory") is None or state["memory"].path != path:
        from Tantra.memory import Memory
        state["memory"] = Memory(path)
    return state["memory"]


def settings():
    path = os.path.join(MODEL_DIR, "assistant.json")
    if state.get("settings") is None or state["settings"].path != path:
        from Tantra.assistant import Settings
        state["settings"] = Settings(path)
    return state["settings"]


def cfg() -> Dict[str, Any]:
    return settings().get()


def documents():
    from Tantra.documents import Documents
    return Documents(os.path.join(MODEL_DIR, "docs.db"))


DEFAULT_APPS = {"notepad": ["notepad.exe"], "calculator": ["calc.exe"], "calc": ["calc.exe"], "paint": ["mspaint.exe"],
                "explorer": ["explorer.exe"], "file explorer": ["explorer.exe"], "नोटपैड": ["notepad.exe"],
                "कैलकुलेटर": ["calc.exe"]}


def allowed_apps() -> Dict[str, List[str]]:
    return {**(cfg().get("apps") or DEFAULT_APPS), **(memory().settings.get("apps") or {})}


def search_files(query: str, limit: int = 20) -> List[str]:
    """File NAMES matching the words, only inside folders the user allowed in Settings."""
    words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 1]
    out, seen = [], 0
    for folder in memory().settings.get("file_folders") or []:
        if not os.path.isdir(folder):
            continue
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                seen += 1
                if all(w in f.lower() for w in words):
                    out.append(os.path.join(root, f))
                    if len(out) >= limit:
                        return out
                if seen > 50_000:
                    return out
    return out


def training_summary(hi: bool) -> str:
    t = _read_json(STATUS_FILE, {})
    if not t:
        return "अभी कोई training नहीं चल रही।" if hi else "No training has run yet."
    v = (t.get("validation") or {}).get("loss")
    pct = 100 * t.get("step", 0) / max(t.get("target_steps", 1), 1)
    probe = [p for p in _probe_history() if p.get("hits") is not None]
    rem = f", remembered {probe[-1]['hits']}/50" if probe else ""
    if hi:
        return (f"Training {t.get('status')}: step {t.get('step', 0):,}/{t.get('target_steps', 0):,} ({pct:.0f}%), "
                f"loss {t.get('loss', 0):.2f}" + (f", val loss {v:.2f}" if v else "") + rem +
                (f"। बाकी समय: {t.get('eta')}" if t.get("status") == "running" else "।"))
    return (f"Training {t.get('status')}: step {t.get('step', 0):,}/{t.get('target_steps', 0):,} ({pct:.0f}%), "
            f"loss {t.get('loss', 0):.2f}" + (f", val loss {v:.2f}" if v else "") + rem +
            (f". ETA {t.get('eta')}." if t.get("status") == "running" else "."))


def daily_brief(hi: bool) -> str:
    import datetime as _dt
    from Tantra.skills import time_date
    parts = [time_date("आज कौन सा दिन है" if hi else "what is the date").text]
    today_end = _dt.datetime.now().replace(hour=23, minute=59).timestamp()
    rem = [r for r in memory().reminders() if r["due"] <= today_end]
    if rem:
        parts.append(("आज के reminders:\n" if hi else "Today's reminders:\n") +
                     "\n".join(f"- {_dt.datetime.fromtimestamp(r['due']).strftime('%I:%M %p')}: {r['text']}" for r in rem))
    else:
        parts.append("आज कोई reminder नहीं है।" if hi else "No reminders today.")
    parts.append(training_summary(hi))
    return "\n\n".join(parts)


def skill_ctx() -> Dict[str, Any]:
    return {"memory": memory(), "status": training_summary, "brief": daily_brief,
            "files": search_files, "apps": allowed_apps()}


def direct_answer(text: str, body: dict) -> Optional[Dict[str, Any]]:
    """Answer without the model when a skill, a taught answer or a strong knowledge match can."""
    from Tantra.skills import handle
    r = handle(text, skill_ctx())
    if r:
        return {"skill": r.name, "text": r.text, "card": r.card, "sources": []}
    from Tantra.assistant import small_talk
    from Tantra.skills import hindi
    reply = small_talk(text, cfg(), hindi(text) if cfg().get("language") == "auto" else cfg().get("language") == "hi")
    if reply:
        return {"skill": "small_talk", "text": reply, "card": {}, "sources": []}
    t = memory().taught_answer(text)
    if t:
        return {"skill": "taught", "text": t["answer"], "card": {"question": t["question"]}, "sources": []}
    if body.get("knowledge_first"):
        hit = None
        docs = documents()
        if docs.list() and os.path.isfile(docs.path):
            from Tantra.smriti import Smriti
            store = Smriti(docs.path, readonly=True)
            try:
                hit = store.best_answer(text, allow_text=True)
            finally:
                store.close()
        s = smriti() if body.get("smriti") else None
        if hit is None and s is not None:
            hit = s.best_answer(text)
        if hit:
            from Tantra.smriti import Smriti
            answer = Smriti.best_sentences(hit["text"], text, 1) if hit["kind"] == "text" else hit["text"]
            return {"skill": "knowledge", "text": answer, "card": {}, "sources": [hit]}
    return None


def with_context(messages: List[dict], query: str, body: dict) -> tuple:
    """Facts for the model: matching memories, your documents and (optionally) Smriti."""
    notes, sources = [], []
    mems = memory().relevant(query)
    if mems:
        notes.append("उपयोगकर्ता के बारे में / About the user:\n" + "\n".join(f"- {m['text']}" for m in mems))
    docs = documents().search(query, 2)
    if docs:
        notes.append("दस्तावेज़ों से / From your documents:\n" + "\n---\n".join(h["text"][:600] for h in docs))
        sources += docs
    if body.get("smriti"):
        _, hits = with_knowledge(messages, query, int(_num(body, "smriti_k", 2, 0, 5)))
        if hits:
            notes.append("नीचे दी गई जानकारी का उपयोग करके उत्तर दें / Use this information to answer:\n" +
                         "\n---\n".join((f"{h['question']}\n" if h["question"] else "") + h["text"][:600] for h in hits))
            sources += hits
    mode = body.get("mode")
    if mode == "translate_en":
        notes.insert(0, "Translate the user's message into English. Reply with the translation only.")
    elif mode == "translate_hi":
        notes.insert(0, "उपयोगकर्ता के संदेश का हिंदी में अनुवाद करें। केवल अनुवाद लिखें।")
    elif mode == "tutor":
        notes.insert(0, "Explain step by step, simply, like a patient teacher.")
    if not notes:
        return messages, sources
    rest = [m for m in messages if m.get("role") != "system"]
    system = next((m["content"] for m in messages if m.get("role") == "system" and m.get("content")), "")
    return [{"role": "system", "content": "\n\n".join(([system] if system else []) + notes)}] + rest, sources


def _num(body: dict, key: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(body.get(key, default))
    except (TypeError, ValueError):
        raise HTTPException(400, f"'{key}' must be a number.")
    return min(max(v, lo), hi)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "tantra", "object": "model", "owned_by": "atulya-ai",
                                        **({"checkpoint": state["info"]["checkpoint"]} if state["info"] else {})}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise HTTPException(400, "'messages' must be a list.")
    build_prompt(messages)   # validates: last message must be from the user
    query = messages[-1]["content"]
    rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())

    def chunk(delta: dict, finish: Optional[str] = None, **extra) -> str:
        c = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": "tantra",
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
        return f"data: {json.dumps(c, ensure_ascii=False)}\n\n"

    direct = None if body.get("mode") in ("translate_en", "translate_hi") or body.get("direct") is False \
        else await asyncio.to_thread(direct_answer, query, body)
    if direct:   # skill / taught answer / knowledge: instant and exact, no model needed
        extra = {"skill": direct["skill"], "card": direct["card"], "sources": direct["sources"],
                 "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "timing": {"seconds": 0}}
        if body.get("stream"):
            async def sse_direct():
                yield chunk({"content": direct["text"]})
                yield chunk({}, "stop", **extra)
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse_direct(), media_type="text/event-stream")
        return {"id": rid, "object": "chat.completion", "created": created, "model": "tantra",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": direct["text"]},
                             "finish_reason": "stop"}], **extra}

    messages, sources = await asyncio.to_thread(with_context, messages, query, body)
    prompt = build_prompt(messages, int(_num(body, "history", 3, 0, 20)))
    try:
        model, tok = await asyncio.to_thread(get_model)
    except HTTPException as exc:
        if exc.status_code not in (503, 409):
            raise
        from Tantra.skills import hindi
        note = cfg()["no_model_reply"]["hi" if hindi(query) else "en"]
        text_out = note + (f"\n\n({exc.detail})" if exc.status_code == 409 else "")
        extra = {"skill": "no_model", "card": {}, "sources": sources,
                 "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "timing": {"seconds": 0}}
        if body.get("stream"):
            async def sse_note():
                yield chunk({"content": text_out})
                yield chunk({}, "stop", **extra)
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse_note(), media_type="text/event-stream")
        return {"id": rid, "object": "chat.completion", "created": created, "model": "tantra",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text_out}, "finish_reason": "stop"}],
                **extra}
    category = pick_category(model, query, body.get("category"))
    ids = torch.tensor([tok.encode(prompt)], device=next(model.parameters()).device)
    gen_args = dict(max_new_tokens=int(_num(body, "max_tokens", 256, 1, 2048)),
                    temperature=_num(body, "temperature", 0.3, 0.0, 2.0),
                    top_p=_num(body, "top_p", 0.9, 0.05, 1.0),
                    repetition_penalty=_num(body, "repetition_penalty", 1.15, 1.0, 2.0),
                    eos_token_id=EOS_ID, adapter_name=category)
    prompt_tokens = ids.shape[1]

    if body.get("stream"):
        async def sse():
            out: List[int] = []
            sent, finish = "", "length"
            t0 = time.perf_counter()
            async with gen_lock:
                it = model.generate_stream(ids, **gen_args)
                try:
                    while True:
                        t = await asyncio.to_thread(next, it, None)
                        if t is None:
                            break
                        t = int(t)
                        if t == EOS_ID:
                            finish = "stop"
                            break
                        out.append(t)
                        text = tok.decode(out)
                        if text.endswith("�"):      # half of a multi-byte character; wait for the rest
                            continue
                        delta, sent = text[len(sent):], text
                        if delta:
                            yield chunk({"content": delta})
                        if await request.is_disconnected():   # user pressed Stop / closed the tab
                            finish = "cancelled"
                            break
                finally:
                    it.close()
            secs = time.perf_counter() - t0
            yield chunk({}, finish, category=category, sources=sources,
                        usage={"prompt_tokens": prompt_tokens, "completion_tokens": len(out),
                               "total_tokens": prompt_tokens + len(out)},
                        timing={"seconds": round(secs, 3), "tokens_per_sec": round(len(out) / max(secs, 1e-6), 1)})
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def run() -> List[int]:
        out = []
        for t in model.generate_stream(ids, **gen_args):
            t = int(t)
            if t == EOS_ID:
                break
            out.append(t)
        return out

    t0 = time.perf_counter()
    async with gen_lock:
        out = await asyncio.to_thread(run)
    secs = time.perf_counter() - t0
    return {"id": rid, "object": "chat.completion", "created": created, "model": "tantra", "category": category,
            "sources": sources,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": tok.decode(out)},
                         "finish_reason": "length" if len(out) >= gen_args["max_new_tokens"] else "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(out),
                      "total_tokens": prompt_tokens + len(out)},
            "timing": {"seconds": round(secs, 3), "tokens_per_sec": round(len(out) / max(secs, 1e-6), 1)}}


# ── saved chats ──────────────────────────────────────────────────────────────

@app.get("/api/chats")
def list_chats():
    return _read_json(CHATS_FILE, {})


@app.post("/api/chats")
async def save_chat(request: Request):
    chat = await request.json()
    messages = [{k: m[k] for k in ("role", "content", "info", "sources", "skill", "card", "opened") if k in m}
                for m in chat.get("messages", []) if isinstance(m, dict) and m.get("role")]
    with chats_lock:
        chats = _read_json(CHATS_FILE, {})
        cid = str(chat.get("id") or uuid.uuid4().hex[:10])[:40]
        first = next((m["content"] for m in messages if m.get("role") == "user"), "New chat")
        old = chats.get(cid, {})
        chats[cid] = {"id": cid, "title": (chat.get("title") or old.get("title") or first)[:60],
                      "updated": int(time.time()), "messages": messages}
        _write_json(CHATS_FILE, chats)
    return chats[cid]


@app.delete("/api/chats/{cid}")
def delete_chat(cid: str):
    with chats_lock:
        chats = _read_json(CHATS_FILE, {})
        chats.pop(cid, None)
        _write_json(CHATS_FILE, chats)
    return {"deleted": cid}


# ── background jobs: train / eval / export ───────────────────────────────────

JOB_NAMES = ("train", "eval", "export", "data", "smriti", "doctor")


def _log_path(name: str) -> str:
    return os.path.join(MODEL_DIR, "logs", f"{name}.log")


def _pid_alive(pid: Any) -> bool:
    try:
        import psutil
        return bool(pid) and psutil.pid_exists(int(pid))
    except (ImportError, ValueError, TypeError):
        return False


def job_info(name: str) -> Dict[str, Any]:
    j = state["jobs"].get(name)
    if not j:
        return {"running": False}
    code = j["proc"].poll()
    return {"running": code is None, "exit_code": code, "started": j["started"], "pid": j["proc"].pid,
            "args": j["args"], "log": rel(j["log"])}


def start_job(name: str, args: List[str]) -> Dict[str, Any]:
    if job_info(name)["running"]:
        raise HTTPException(409, f"'{name}' is already running.")
    log_path = _log_path(name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"$ python main.py --mode {name} {' '.join(args)}\n")
        f.flush()
        proc = subprocess.Popen([sys.executable, "-u", os.path.join(REPO_ROOT, "main.py"), "--mode", name, *args],
                                cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, env=env, creationflags=flags)
    state["jobs"][name] = {"proc": proc, "started": time.time(), "args": args, "log": log_path}
    log.info(f"Started {name} (pid {proc.pid})")
    return {"started": True, **job_info(name)}


_TRAIN_OPTIONS = {   # json key: (cli flag, type, min, max)
    "steps": ("--steps", int, 1, 10_000_000), "batch_size": ("--batch-size", int, 1, 1024),
    "grad_accum": ("--grad-accum", int, 1, 1024), "seq_len": ("--seq-len", int, 16, 16384),
    "lr": ("--lr", float, 1e-7, 1.0), "warmup": ("--warmup", int, 0, 1_000_000),
    "eval_every": ("--eval-every", int, 1, 1_000_000),
}


def _dataset(name: str) -> str:
    path = os.path.join(DATA_DIR, os.path.basename(str(name)))
    if not path.endswith(".jsonl") or not os.path.isfile(path):
        raise HTTPException(400, f"Dataset not found: {name}")
    return path


def training_running() -> bool:
    if job_info("train")["running"]:
        return True
    t = _read_json(STATUS_FILE, {})
    return t.get("status") == "running" and _pid_alive(t.get("pid"))


@app.post("/api/training/start", dependencies=[Depends(require_key)])
async def train_start(request: Request):
    body = await request.json()
    if training_running():
        raise HTTPException(409, "Training is already running.")
    args: List[str] = []
    for key, (flag, typ, lo, hi) in _TRAIN_OPTIONS.items():
        if body.get(key) not in (None, ""):
            try:
                v = typ(body[key])
            except (TypeError, ValueError):
                raise HTTPException(400, f"'{key}' must be a number.")
            if not lo <= v <= hi:
                raise HTTPException(400, f"'{key}' must be between {lo} and {hi}.")
            args += [flag, str(v)]
    if body.get("preset"):
        if body["preset"] not in ("tiny", "small", "billion"):
            raise HTTPException(400, "preset must be tiny, small or billion.")
        args += ["--preset", body["preset"]]
    if body.get("data"):
        names = body["data"] if isinstance(body["data"], list) else [body["data"]]
        args += ["--data", ",".join(_dataset(n) for n in names)]
    if body.get("val"):
        args += ["--val", _dataset(body["val"])]
    if body.get("stage") in ("sft", "pretrain"):
        args += ["--stage", body["stage"]]
    if body.get("fresh"):
        args.append("--fresh")
    if body.get("auto_growth"):
        args.append("--auto-growth")
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)
    return start_job("train", args)


@app.post("/api/training/stop", dependencies=[Depends(require_key)])
def train_stop():
    open(STOP_FILE, "w").close()   # trainer saves a checkpoint and exits at the next step
    return {"stopping": True}


def _checkpoint_arg(body: dict) -> List[str]:
    p = body.get("checkpoint")
    if not p:
        return []
    path = os.path.realpath(os.path.join(REPO_ROOT, p))
    if not _inside(path, MODEL_DIR) or not os.path.isfile(path):
        raise HTTPException(404, "Checkpoint not found under Model/")
    return ["--checkpoint", path]


@app.post("/api/jobs/eval", dependencies=[Depends(require_key)])
async def eval_start(request: Request):
    body = await request.json()
    return start_job("eval", _checkpoint_arg(body) + (["--int8"] if body.get("int8") else []))


@app.post("/api/jobs/export", dependencies=[Depends(require_key)])
async def export_start(request: Request):
    body = await request.json()
    return start_job("export", _checkpoint_arg(body))


@app.post("/api/jobs/data", dependencies=[Depends(require_key)])
def data_start():
    if training_running():
        raise HTTPException(409, "Stop training first — this rewrites the training files.")
    return start_job("data", [])


@app.post("/api/jobs/smriti", dependencies=[Depends(require_key)])
def smriti_start():
    return start_job("smriti", [])


@app.get("/api/smriti/search")
def smriti_search(q: str, k: int = 5):
    s = smriti()
    if s is None:
        raise HTTPException(404, "No knowledge store yet. Build it: python main.py --mode smriti")
    return {"query": q, "hits": s.search(q, k=max(1, min(k, 20)))}


@app.post("/api/jobs/{name}/cancel", dependencies=[Depends(require_key)])
def job_cancel(name: str):
    if name == "train":
        return train_stop()
    j = state["jobs"].get(name)
    if j and j["proc"].poll() is None:
        j["proc"].terminate()
    return job_info(name)


@app.get("/api/logs/{name}")
def job_log(name: str, lines: int = 200):
    if name not in JOB_NAMES:
        raise HTTPException(404, "Unknown job.")
    path = _log_path(name)
    if not os.path.isfile(path):
        return {"lines": [], "job": job_info(name)}
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 256 * 1024))
        text = f.read().decode("utf-8", errors="replace")
    rows = [r.split("\r")[-1] for r in text.splitlines()]   # progress bars redraw with \r
    return {"lines": rows[-max(1, min(lines, 2000)):], "job": job_info(name)}


# ── status / checkpoints ─────────────────────────────────────────────────────

def _speech_available() -> Dict[str, bool]:
    return {"stt": importlib.util.find_spec("whisper") is not None,
            "tts": importlib.util.find_spec("kokoro") is not None}


def _datasets() -> List[Dict[str, Any]]:
    out = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.jsonl"))):
        out.append({"name": os.path.basename(p), "size_mb": round(os.path.getsize(p) / 2**20, 1)})
    return out


def _probe_history() -> List[dict]:
    rows = []
    if os.path.isfile(PROBE_FILE):
        with open(PROBE_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows[-200:]


@app.get("/api/status")
def status():
    training = _read_json(STATUS_FILE, {"status": "idle"})
    if training.get("status") == "running":
        # Dead only if the training process is gone (a slow CPU can go many minutes between updates).
        pid = training.get("pid")
        alive = _pid_alive(pid) if pid else time.time() - float(training.get("updated_at", 0)) < 1800
        if not alive and not job_info("train")["running"]:
            training["status"] = "interrupted"
    training["launched_here"] = job_info("train")["running"]
    ckpts = []
    for p in checkpoints():
        try:
            ckpts.append(checkpoint_details(p))
        except OSError:
            continue
    return {"model": state["info"], "load_error": state["load_error"], "training": training,
            "probe": _probe_history(), "hardware": hardware().as_dict(),
            "checkpoints": [c["path"] for c in ckpts], "checkpoint_details": ckpts,
            "jobs": {n: job_info(n) for n in JOB_NAMES},
            "eval_report": _read_json(os.path.join(MODEL_DIR, "eval_report.json"), None),
            "datasets": _datasets(), "speech": _speech_available(), "smriti": smriti_stats(),
            "data_report": _read_json(os.path.join(DATA_DIR, "data_report.json"), None),
            "auth_required": bool(API_KEY), "server_time": time.time()}


@app.post("/api/checkpoints/switch", dependencies=[Depends(require_key)])
async def switch(request: Request):
    body = await request.json()
    path = os.path.realpath(os.path.join(REPO_ROOT, body.get("path", "")))
    if not _inside(path, MODEL_DIR) or not os.path.isfile(path):
        raise HTTPException(404, "Checkpoint not found under Model/")
    async with gen_lock:   # never swap the model in the middle of a reply
        await asyncio.to_thread(load, path, body.get("int8"))
    return state["info"]


# ── self-repair ──────────────────────────────────────────────────────────────

@app.get("/api/doctor")
def doctor_report():
    from Tantra.doctor import run_checks
    return {"checks": run_checks(), "job": job_info("doctor")}


@app.post("/api/doctor/fix", dependencies=[Depends(require_key)])
async def doctor_fix(request: Request):
    body = await request.json() if (await request.body()) else {}
    only = body.get("id")
    from Tantra.doctor import CHECKS
    if only and only not in {c["id"] for c in CHECKS}:
        raise HTTPException(404, "Unknown check.")
    if only in ("data", "smriti") and training_running():
        raise HTTPException(409, "Stop training first — this rewrites files training reads.")
    return start_job("doctor", ["--only", only] if only else ["--fix"])


# ── assistant settings (Model/assistant.json) ────────────────────────────────

@app.get("/api/config")
def config_get():
    from Tantra.assistant import DEFAULTS
    return {"config": cfg(), "defaults": DEFAULTS}


@app.post("/api/config", dependencies=[Depends(require_key)])
async def config_set(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "Send a JSON object.")
    return {"config": settings().update(body)}


@app.post("/api/config/reset", dependencies=[Depends(require_key)])
async def config_reset(request: Request):
    key = (await request.json()).get("key") if (await request.body()) else None
    return {"config": settings().reset(key)}


# ── memory, reminders, teach, documents, brief ───────────────────────────────

def local_only(request: Request) -> None:
    """Actions that touch this computer (run code, open apps, read folders) only from the PC itself."""
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(403, "Only allowed from this computer.")


@app.get("/api/memory")
def memory_list():
    m = memory()
    return {"memories": m.list(), "reminders": m.reminders(include_done=True)[-100:], "taught": m.data["taught"][:100]}


@app.post("/api/memory")
async def memory_add(request: Request):
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "Empty memory.")
    return memory().add(text, body.get("category"), source="manual")


@app.patch("/api/memory/{mid}")
async def memory_edit(mid: str, request: Request):
    body = await request.json()
    item = memory().update(mid, body.get("text"), body.get("category"))
    if not item:
        raise HTTPException(404, "Memory not found.")
    return item


@app.delete("/api/memory/{mid}")
def memory_delete(mid: str):
    memory().delete(mid)
    return {"deleted": mid}


@app.get("/api/reminders/due")
def reminders_due():
    return {"due": memory().due(), "upcoming": memory().reminders()[:20]}


@app.post("/api/reminders/{rid}/{action}")
async def reminder_action(rid: str, action: str, request: Request):
    if action not in ("done", "snooze"):
        raise HTTPException(400, "action must be done or snooze")
    body = await request.json() if (await request.body()) else {}
    r = memory().reminder_action(rid, action, int(body.get("minutes", 10)))
    if not r:
        raise HTTPException(404, "Reminder not found.")
    return r


@app.post("/api/feedback")
async def feedback(request: Request):
    """👎 Teach Tantra: remember the right answer now, and keep it for the next training run."""
    body = await request.json()
    q, correct = (body.get("question") or "").strip(), (body.get("correct") or "").strip()
    if not q or not correct:
        raise HTTPException(400, "Need the question and the correct answer.")
    item = memory().teach(q, correct)
    row = {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": correct}],
           "source": "feedback", "bad_answer": (body.get("bad") or "")[:2000], "time": time.time()}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "feedback.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return item


@app.get("/api/brief")
def brief(lang: str = "hi"):
    return {"text": daily_brief(lang == "hi")}


@app.get("/api/docs")
def docs_list():
    return {"docs": documents().list()}


@app.post("/api/docs")
async def docs_add(request: Request):
    form = await request.form()
    f = form.get("file")
    if f is None:
        raise HTTPException(400, "Send the file as form field 'file'.")
    try:
        return await asyncio.to_thread(documents().add, f.filename or "document.txt", await f.read())
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.delete("/api/docs/{name}")
def docs_delete(name: str):
    documents().remove(name)
    return {"deleted": name}


@app.post("/api/code/run", dependencies=[Depends(local_only), Depends(require_key)])
async def run_code(request: Request):
    """Run a Python snippet the user confirmed: isolated interpreter, temp folder, 10 s limit, output capped."""
    code = (await request.json()).get("code") or ""
    if not code.strip() or len(code) > 20_000:
        raise HTTPException(400, "No code (or longer than 20,000 characters).")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "snippet.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        t0 = time.perf_counter()
        try:
            p = await asyncio.to_thread(subprocess.run, [sys.executable, "-I", path], cwd=d, capture_output=True,
                                        text=True, encoding="utf-8", errors="replace", timeout=10,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                        env={"PYTHONIOENCODING": "utf-8", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})
            out, err, code_ = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired:
            out, err, code_ = "", "Stopped: took longer than 10 seconds.", -1
    return {"stdout": out[-20_000:], "stderr": err[-10_000:], "exit_code": code_,
            "seconds": round(time.perf_counter() - t0, 2)}


@app.post("/api/open", dependencies=[Depends(local_only), Depends(require_key)])
async def open_app(request: Request):
    name = ((await request.json()).get("name") or "").strip().lower()
    cmd = allowed_apps().get(name)
    if not cmd:
        raise HTTPException(404, f"'{name}' is not in the allowed apps list.")
    subprocess.Popen(cmd, cwd=os.path.expanduser("~"))
    return {"opened": name}


@app.get("/api/assistant/settings")
def assistant_settings():
    return {**memory().settings, "apps": sorted(allowed_apps())}


@app.post("/api/assistant/settings", dependencies=[Depends(local_only), Depends(require_key)])
async def assistant_settings_set(request: Request):
    body = await request.json()
    folders = body.get("file_folders")
    if folders is not None:
        folders = [os.path.abspath(f) for f in folders if isinstance(f, str) and os.path.isdir(f)]
    return memory().set_settings(file_folders=folders)


@app.get("/api/access", dependencies=[Depends(local_only)])
def access_info():
    """How to open Tantra from a phone on the same Wi-Fi (shown only on this computer)."""
    import socket
    ip = ""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
    except OSError:
        pass
    return {"lan_url": f"http://{ip}:{state.get('port', 8000)}" if ip else None, "lan_enabled": state.get("lan", False),
            "api_key": API_KEY if state.get("lan") else None}


# ── speech (optional, offline) ───────────────────────────────────────────────

_speech: Dict[str, Any] = {}


def _wav_to_array(data: bytes):
    """16-bit PCM WAV (what the WebUI sends) -> float32 mono 16 kHz array. No ffmpeg needed."""
    import numpy as np
    with wave.open(io.BytesIO(data)) as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        pcm = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("WAV must be 16-bit")
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != 16000 and len(x):
        n = int(len(x) * 16000 / rate)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def whisper_model():
    if "whisper" not in _speech:   # first use downloads the model once (~140 MB for "base")
        import whisper
        _speech["whisper"] = whisper.load_model(os.environ.get("TANTRA_WHISPER") or cfg()["voice"].get("whisper_model", "base"), device="cpu")
    return _speech["whisper"]


@app.post("/api/stt")
async def speech_to_text(request: Request):
    form = await request.form()
    audio = form.get("audio")
    if audio is None:
        raise HTTPException(400, "Send the recording as form field 'audio'.")
    if importlib.util.find_spec("whisper") is None:
        raise HTTPException(503, "Speech-to-text not installed. Open Settings → System health → Fix (installs openai-whisper).")
    data = await audio.read()
    lang = form.get("language") or None
    try:
        model = await asyncio.to_thread(whisper_model)
        if data[:4] == b"RIFF":                       # WAV from the WebUI: decoded here, no ffmpeg
            src = _wav_to_array(data)
            if len(src) < 16000 * 0.3:
                return {"text": "", "language": None}
        else:                                         # other formats need ffmpeg on PATH
            if not shutil.which("ffmpeg"):
                raise HTTPException(503, "This audio format needs ffmpeg. Send WAV, or install ffmpeg.")
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(data)
                src = f.name
        try:
            result = await asyncio.to_thread(model.transcribe, src, language=lang, fp16=False,
                                             initial_prompt="नमस्ते। Hello. तन्त्र।")
        finally:
            if isinstance(src, str):
                os.unlink(src)
        return {"text": result.get("text", "").strip(), "language": result.get("language")}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("speech-to-text failed")
        raise HTTPException(503, f"Speech-to-text failed: {exc}")


@app.post("/api/tts")
async def text_to_speech(request: Request):
    body = await request.json()
    text = (body.get("text") or "").strip()[:1000]
    if not text:
        raise HTTPException(400, "No text.")
    try:
        import numpy as np
        from kokoro import KPipeline
    except ImportError:
        raise HTTPException(503, "Text-to-speech not installed. Run: pip install kokoro soundfile")
    hindi = bool(re.search(r"[ऀ-ॿ]", text))
    lang, voice = ("h", body.get("voice") or "hf_alpha") if hindi else ("a", body.get("voice") or "af_heart")
    try:
        if lang not in _speech:   # first use downloads the 82M model (~330 MB) once
            _speech[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
        chunks = await asyncio.to_thread(lambda: [np.asarray(a) for _, _, a in _speech[lang](text, voice=voice)])
    except Exception as exc:
        raise HTTPException(503, f"Text-to-speech failed: {exc}")
    pcm = (np.clip(np.concatenate(chunks), -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)
    return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="audio/wav")


@app.exception_handler(Exception)
async def unexpected_error(_: Request, exc: Exception):
    log.exception("Request failed")
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """host 0.0.0.0 = reachable from your phone on the same Wi-Fi (a key is then always required)."""
    global API_KEY
    import uvicorn
    state["port"], state["lan"] = port, host not in ("127.0.0.1", "localhost", "::1")
    if cfg().get("auto_repair", True):   # install missing packages quietly in the background
        try:
            from Tantra.doctor import CHECKS
            if any(c.get("auto") and not c["check"]() for c in CHECKS):
                start_job("doctor", ["--fix", "--auto"])
                print("  Self-repair: installing missing packages in the background (Settings → System health).")
        except Exception as exc:   # never block startup
            log.warning(f"self-repair skipped: {exc}")
    if state["lan"] and not API_KEY:
        API_KEY = secrets.token_urlsafe(9)
    print(f"\n  Tantra WebUI → http://127.0.0.1:{port}\n")
    if state["lan"]:
        print(f"  Phone / other devices: http://<this PC's IP>:{port}   key: {API_KEY}  (also in the Model tab)\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
