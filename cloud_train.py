"""
cloud_train.py — "run training" on Kaggle or Google Colab (or any GPU machine). One file does everything.

In a notebook cell:
    !rm -rf /tmp/Tantra-LLM && git clone -q --depth 1 https://github.com/atulyaai/Tantra-LLM.git /tmp/Tantra-LLM
    %run /tmp/Tantra-LLM/cloud_train.py

Options (after the %run line):  --stage sft   --steps 60000   --hours 8   --batch 16   --gpus 1   --fresh

Where the data comes from and where results go:
  Kaggle  data: your dataset with pretrain.jsonl/sft.jsonl/tokenizer.json (Add data → tantra-data)
          results: /kaggle/working/tantra_out (+ tantra_out.zip) — download it, or add it as input next time
  Colab   data + results: Google Drive → MyDrive/tantra-data  (put the kaggle_upload files there once);
          checkpoints are written to Drive, so the next session simply continues
  other   data: ./tantra-data  results: ./tantra-data/out
Every run continues from the newest latest.pt it finds. All GPUs are used; batch size follows GPU memory.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
START = os.getcwd()   # where the user started it (local ./tantra-data lives here)


def say(msg: str) -> None:
    print(f"» {msg}", flush=True)


def environment() -> str:
    if os.path.isdir("/kaggle/input"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


def find_data(env: str) -> str:
    if env == "kaggle":
        hits = glob.glob("/kaggle/input/**/pretrain.jsonl", recursive=True) + glob.glob("/kaggle/input/**/sft.jsonl", recursive=True)
        if not hits:
            sys.exit("✗ No data. Right panel → Add data → Your Datasets → tantra-data (the files from kaggle_upload/).")
        return os.path.dirname(hits[0])
    if env == "colab":
        from google.colab import drive
        if not os.path.isdir("/content/drive/MyDrive"):
            drive.mount("/content/drive")
        d = "/content/drive/MyDrive/tantra-data"
        if not glob.glob(os.path.join(d, "*.jsonl")):
            sys.exit(f"✗ No data in Google Drive {d}. Upload the files from kaggle_upload/ there once.")
        return d
    d = os.path.join(START, "tantra-data")
    if not glob.glob(os.path.join(d, "*.jsonl")):
        sys.exit(f"✗ Put pretrain.jsonl, sft.jsonl, val_*.jsonl, probe_50.jsonl and tokenizer.json in {d}")
    return d


def output_dir(env: str, data: str) -> str:
    return {"kaggle": "/kaggle/working/tantra_out", "colab": os.path.join(data, "out")}.get(env, os.path.join(data, "out"))


def gpus() -> list:
    try:
        import torch
        return [(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 2**30)
                for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def main() -> None:
    p = argparse.ArgumentParser(description="Train Tantra on Kaggle / Colab / any GPU machine")
    p.add_argument("--stage", default="pretrain", choices=["pretrain", "sft"], help="pretrain first, then sft")
    p.add_argument("--steps", type=int, default=30000, help="target step (continues from the last checkpoint)")
    p.add_argument("--hours", type=float, default=0, help="stop cleanly after N hours (default: 11 Kaggle/Colab)")
    p.add_argument("--batch", type=int, default=0, help="sequences per GPU pass (default: from GPU memory)")
    p.add_argument("--total-batch", type=int, default=32, help="sequences per update across all GPUs")
    p.add_argument("--gpus", default="auto", help="'auto' = all GPUs, or a number")
    p.add_argument("--fresh", action="store_true", help="start a new model instead of continuing")
    p.add_argument("--eval-every", type=int, default=500)
    args, _ = p.parse_known_args()

    env = environment()
    cards = gpus()
    say(f"Environment: {env} · GPUs: " + (", ".join(f"{n} ({m:.0f} GB)" for n, m in cards) or "none (CPU — slow!)"))
    os.chdir(HERE)

    say("Installing packages …")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers", "psutil", "fastapi", "uvicorn",
                    "python-multipart"], check=True)

    data = find_data(env)
    out = output_dir(env, data)
    os.makedirs(out, exist_ok=True)
    os.makedirs("Datasets", exist_ok=True)
    os.makedirs("Model", exist_ok=True)
    for name in os.listdir(data):
        src, dst = os.path.join(data, name), os.path.join("Datasets", name)
        if name.endswith(".jsonl") and not os.path.exists(dst):
            os.symlink(src, dst)                                    # big files: link, not copy
    # tokenizer: one uploaded with the data, anywhere in the inputs, else the copy that ships with the code
    cands = [os.path.join(data, "tokenizer.json")] + glob.glob("/kaggle/input/**/tokenizer.json", recursive=True)
    tok = next((c for c in cands if os.path.isfile(c)), None)
    if tok:
        shutil.copy(tok, "Model/tokenizer.json")
        say(f"Tokenizer: {tok}")
    elif os.path.isfile("Model/tokenizer.json"):
        say("Tokenizer: the one included with the code (Model/tokenizer.json)")
    else:
        sys.exit("✗ tokenizer.json not found (it is in kaggle_upload/ and in the GitHub repo's Model folder).")

    # continue from the newest latest.pt: previous output first, then one uploaded with the data
    if not args.fresh and not os.path.isfile(os.path.join(out, "latest.pt")):
        for cand in [os.path.join(data, "latest.pt")] + glob.glob("/kaggle/input/**/latest.pt", recursive=True):
            if os.path.isfile(cand):
                say(f"Continuing from {cand}")
                for n in ("latest.pt", "latest.pt.meta.json", "training_status.json", "probe_history.jsonl"):
                    s = os.path.join(os.path.dirname(cand), n)
                    if os.path.isfile(s):
                        shutil.copy(s, out)
                break
    resume = os.path.isfile(os.path.join(out, "latest.pt")) and not args.fresh
    say("Continuing training" if resume else "Starting a new model")

    n_gpu = len(cards) if args.gpus == "auto" else min(int(args.gpus), len(cards))
    mem = min((m for _, m in cards), default=0)
    batch = args.batch or (4 if mem < 12 else 8 if mem < 20 else 16 if mem < 40 else 32)
    accum = max(1, args.total_batch // (batch * max(n_gpu, 1)))
    hours = args.hours or (11 if env in ("kaggle", "colab") else 0)
    say(f"Plan: {n_gpu or 'CPU'} GPU(s) × batch {batch} × accumulate {accum} = "
        f"{batch * accum * max(n_gpu, 1)} sequences per update · stop after {hours or '∞'} h")

    cmd = [sys.executable, "main.py", "--mode", "train", "--stage", args.stage, "--model-dir", out,
           "--device", "cuda" if cards else "cpu", "--gpus", str(max(n_gpu, 1)), "--batch-size", str(batch),
           "--grad-accum", str(accum), "--seq-len", "512", "--steps", str(args.steps), "--warmup", "500",
           "--eval-every", str(args.eval_every), "--log-every", "50", "--workers", "2"]
    if hours:
        cmd += ["--max-hours", str(hours)]
    if not resume:
        cmd.append("--fresh")
    say(" ".join(cmd))
    env_vars = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                    PYTORCH_ALLOC_CONF="expandable_segments:True")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env_vars, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:                                     # live progress in the notebook
        print(line, end="", flush=True)
    proc.wait()
    for n in ("training_status.json", "probe_history.jsonl"):   # copy the live status files next to the checkpoints
        if os.path.isfile(os.path.join("Model", n)):
            shutil.copy(os.path.join("Model", n), out)
    say(f"Training ended (exit {proc.returncode}) after {(time.time() - t0) / 3600:.1f} h")

    if env == "kaggle":
        shutil.make_archive("/kaggle/working/tantra_out", "zip", out)
        say("Download tantra_out.zip (Output panel) → put latest.pt and best.pt into your PC's Model/ folder.")
        say("Next session: add this notebook's output as input (Add data → Notebook output) and run again — it continues.")
    elif env == "colab":
        say(f"Checkpoints are in Google Drive: {out} — the next session continues from there automatically.")
    else:
        say(f"Checkpoints: {out}")


if __name__ == "__main__":
    main()
