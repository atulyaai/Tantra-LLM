"""
Tantra/doctor.py — checks every part of Tantra and repairs what it can, automatically.

  python main.py --mode doctor          report
  python main.py --mode doctor --fix    report + repair (installs missing packages, builds data / Smriti)
The WebUI runs the same checks at start (Settings → System health) and, with auto_repair on,
installs missing Python packages in the background — like a control panel that fixes itself.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "Model")
DATA_DIR = os.path.join(ROOT, "Datasets")


def _has(*modules: str) -> bool:
    importlib.invalidate_caches()
    return all(importlib.util.find_spec(m) is not None for m in modules)


def _exists(*parts: str) -> bool:
    return os.path.isfile(os.path.join(ROOT, *parts))


def _ckpt() -> bool:
    return any(_exists("Model", n) for n in ("tantra.pt", "best.pt", "latest.pt"))


# id: (title, why it matters, check, fix)   fix = {"pip": [...]} | {"mode": "<main.py mode>"} | {"hint": "..."}
CHECKS: List[Dict] = [
    {"id": "python", "title": "Python 3.10+", "why": "Tantra needs Python 3.10 or newer.",
     "check": lambda: sys.version_info >= (3, 10), "fix": {"hint": "Install Python 3.12 from python.org or: winget install Python.Python.3.12"}},
    {"id": "core", "title": "Core packages", "why": "PyTorch, tokenizers, FastAPI, … — needed for everything.",
     "check": lambda: _has("torch", "tokenizers", "numpy", "psutil", "fastapi", "uvicorn", "multipart"),
     "fix": {"pip": ["-r", os.path.join(ROOT, "requirements.txt")]}},
    {"id": "stt", "title": "Speech input (Whisper)", "why": "Voice mode and the mic button.",
     "check": lambda: _has("whisper"), "fix": {"pip": ["openai-whisper"]}, "auto": True},
    {"id": "tts", "title": "Tantra's own voice (Kokoro)", "why": "Natural offline speech; without it Windows voices are used.",
     "check": lambda: _has("kokoro", "soundfile"), "fix": {"pip": ["kokoro", "soundfile"]}, "auto": True},
    {"id": "pdf", "title": "PDF reading", "why": "Upload PDF documents.",
     "check": lambda: _has("pypdf"), "fix": {"pip": ["pypdf"]}, "auto": True},
    {"id": "tokenizer", "title": "Tokenizer", "why": "Model/tokenizer.json turns text into tokens.",
     "check": lambda: _exists("Model", "tokenizer.json"), "fix": {"mode": "tokenizer"}},
    {"id": "data", "title": "Cleaned training data", "why": "Datasets/pretrain.jsonl + sft.jsonl.",
     "check": lambda: _exists("Datasets", "pretrain.jsonl") and _exists("Datasets", "sft.jsonl"),
     "fix": {"mode": "data"}},
    {"id": "smriti", "title": "Smriti knowledge store", "why": "Facts Tantra looks up (Model/smriti.db).",
     "check": lambda: _exists("Model", "smriti.db"), "fix": {"mode": "smriti"}},
    {"id": "model", "title": "Trained model", "why": "Free-form answers need a checkpoint (latest.pt / best.pt / tantra.pt).",
     "check": _ckpt, "fix": {"hint": "Training saves the first checkpoint at step 250 — see the Training tab."}},
    {"id": "disk", "title": "Free disk space (5 GB+)", "why": "Checkpoints and data need room.",
     "check": lambda: shutil.disk_usage(ROOT).free > 5 * 2**30, "fix": {"hint": "Free some space (e.g. empty Model/_old)."}},
]


def run_checks() -> List[Dict]:
    out = []
    for c in CHECKS:
        try:
            ok = bool(c["check"]())
        except Exception:
            ok = False
        fix = c["fix"]
        out.append({"id": c["id"], "title": c["title"], "why": c["why"], "ok": ok, "auto": c.get("auto", False),
                    "fix": "install " + " ".join(p for p in fix["pip"] if not p.startswith("-") and not p.endswith(".txt"))
                    if "pip" in fix and fix["pip"][0] != "-r" else
                    "install requirements.txt" if "pip" in fix else
                    f"run: python main.py --mode {fix['mode']}" if "mode" in fix else fix.get("hint", "")})
    return out


def fix(check_id: str, log: Callable[[str], None] = print) -> bool:
    c = next((c for c in CHECKS if c["id"] == check_id), None)
    if c is None:
        raise KeyError(check_id)
    if c["check"]():
        log(f"{c['title']}: already fine.")
        return True
    f = c["fix"]
    if "pip" in f:
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *f["pip"]]
    elif "mode" in f:
        if f["mode"] == "smriti" and not (_exists("Datasets", "sft.jsonl") or _exists("Datasets", "pretrain.jsonl")):
            log("Smriti needs the cleaned data first — fixing 'data' first.")
            if not fix("data", log):
                return False
        if f["mode"] == "data" and not _exists("Datasets", "master_train.jsonl"):
            log("No Datasets/master_train.jsonl to clean — add your .jsonl data first.")
            return False
        cmd = [sys.executable, os.path.join(ROOT, "main.py"), "--mode", f["mode"]]
    else:
        log(f"{c['title']}: {f.get('hint')}")
        return False
    log(f"$ {' '.join(cmd)}")
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for line in p.stdout:
        line = line.rstrip()
        if line:
            log(line[-300:])
    p.wait()
    ok = p.returncode == 0 and bool(c["check"]())
    log(f"{c['title']}: {'fixed' if ok else 'still not fixed'} ({time.time() - t0:.0f}s)")
    return ok


def fix_all(auto_only: bool = False, log: Callable[[str], None] = print) -> Dict[str, bool]:
    """Repair everything repairable. auto_only=True: only the safe background installs (packages)."""
    done = {}
    for r in run_checks():
        c = next(c for c in CHECKS if c["id"] == r["id"])
        if r["ok"] or "hint" in c["fix"]:
            continue
        if auto_only and not c.get("auto"):
            continue
        done[r["id"]] = fix(r["id"], log)
    return done


def report(checks: Optional[List[Dict]] = None) -> str:
    checks = checks or run_checks()
    lines = [f"  {'✓' if c['ok'] else '✗'} {c['title']:<30} {'' if c['ok'] else '→ ' + c['fix']}" for c in checks]
    bad = sum(not c["ok"] for c in checks)
    return "\n".join(lines) + f"\n\n  {'All good.' if not bad else f'{bad} to fix.'}"
