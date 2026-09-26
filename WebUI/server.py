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


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"), headers=_NO_CACHE)


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
    prompt = build_prompt(messages, int(_num(body, "history", 3, 0, 20)))
    model, tok = await asyncio.to_thread(get_model)
    category = pick_category(model, messages[-1]["content"], body.get("category"))
    ids = torch.tensor([tok.encode(prompt)], device=next(model.parameters()).device)
    gen_args = dict(max_new_tokens=int(_num(body, "max_tokens", 256, 1, 2048)),
                    temperature=_num(body, "temperature", 0.3, 0.0, 2.0),
                    top_p=_num(body, "top_p", 0.9, 0.05, 1.0),
                    repetition_penalty=_num(body, "repetition_penalty", 1.15, 1.0, 2.0),
                    eos_token_id=EOS_ID, adapter_name=category)
    rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
    prompt_tokens = ids.shape[1]

    def chunk(delta: dict, finish: Optional[str] = None, **extra) -> str:
        c = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": "tantra",
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
        return f"data: {json.dumps(c, ensure_ascii=False)}\n\n"

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
            yield chunk({}, finish, category=category,
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
    messages = [{k: m[k] for k in ("role", "content", "info") if k in m}
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
    return t.get("status") == "running" and time.time() - float(t.get("updated_at", 0)) < 300 \
        and _pid_alive(t.get("pid"))


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
    if name not in ("train", "eval", "export"):
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
        stale = time.time() - float(training.get("updated_at", 0)) > 300
        if stale or (training.get("pid") and not _pid_alive(training["pid"]) and not job_info("train")["running"]):
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
            "jobs": {n: job_info(n) for n in ("train", "eval", "export")},
            "eval_report": _read_json(os.path.join(MODEL_DIR, "eval_report.json"), None),
            "datasets": _datasets(), "speech": _speech_available(),
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


# ── speech (optional, offline) ───────────────────────────────────────────────

_speech: Dict[str, Any] = {}


@app.post("/api/stt")
async def speech_to_text(request: Request):
    form = await request.form()
    audio = form.get("audio")
    if audio is None:
        raise HTTPException(400, "Send the recording as form field 'audio'.")
    try:
        import whisper
    except ImportError:
        raise HTTPException(503, "Speech-to-text not installed. Run: pip install openai-whisper  (needs ffmpeg)")
    data = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        if "whisper" not in _speech:   # first use downloads the model once
            _speech["whisper"] = whisper.load_model(os.environ.get("TANTRA_WHISPER", "base"), device="cpu")
        lang = form.get("language") or None
        result = await asyncio.to_thread(_speech["whisper"].transcribe, tmp, language=lang, fp16=False)
        return {"text": result.get("text", "").strip(), "language": result.get("language")}
    except Exception as exc:
        raise HTTPException(503, f"Speech-to-text failed: {exc}")
    finally:
        os.unlink(tmp)


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
    import uvicorn
    print(f"\n  Tantra WebUI → http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
