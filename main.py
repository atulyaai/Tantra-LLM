"""
main.py — Tantra command line. Every task is one --mode.

  python main.py --mode train                 train (continues automatically if a checkpoint exists)
  python main.py --mode train --fresh         start a new model (old checkpoints are moved to Model/_old/)
  python main.py --mode chat                  talk to the model in the terminal
  python main.py --mode eval                  validation loss + 50-question probe + speed
  python main.py --mode serve                 WebUI + OpenAI-compatible API on http://127.0.0.1:8000
  python main.py --mode export                small fp16 file for inference (Model/tantra.pt)
  python main.py --mode data                  clean + mix all data -> Datasets/pretrain.jsonl, sft.jsonl, val_*.jsonl
  python main.py --mode tokenizer             build Model/tokenizer.json from your data (do this ONCE)
  python main.py --mode smriti                build the knowledge store Model/smriti.db (facts the model looks up)
  python main.py --mode dpo --prefs FILE      preference tuning from chosen/rejected pairs
  python main.py --mode adapter               list / install category specialist layers
  python main.py --mode hardware              show CPU / RAM / GPU

Folders:  Datasets/ (your .jsonl)   Model/ (tokenizer + checkpoints)   Tantra/ (engine)   WebUI/   Tests/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time

import torch

from Tantra.config import EOS_ID, NeuroCoreConfig
from Tantra.hardware import detect_hardware
from Tantra.model import NeuroCoreModel, load_model
from Tantra.tokenizer import build_tokenizer, load_tokenizer
from Tantra.utils import get_logger, print_banner, set_seed, unwrap_model

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "Model")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
DATA_DIR = os.path.join(ROOT, "Datasets")
log = get_logger("tantra")


def default_data(args) -> None:
    """Pick the cleaned files for the chosen stage unless --data / --val were given."""
    stage_file = os.path.join(DATA_DIR, "pretrain.jsonl" if args.stage == "pretrain" else "sft.jsonl")
    if not args.data:
        args.data = stage_file if os.path.isfile(stage_file) else os.path.join(DATA_DIR, "master_train.jsonl")
    if args.val is None:
        v = os.path.join(DATA_DIR, f"val_{args.stage}.jsonl")
        args.val = v if os.path.isfile(v) else os.path.join(DATA_DIR, "master_val.jsonl")


def ckpt_paths(model_dir: str) -> dict:
    # Only two training files: latest (to continue) and best (lowest validation loss).
    return {"latest": os.path.join(model_dir, "latest.pt"), "best": os.path.join(model_dir, "best.pt")}


def default_checkpoint(model_dir: str = MODEL_DIR) -> str:
    p = ckpt_paths(model_dir)
    for c in (os.path.join(model_dir, "tantra.pt"), p["best"], p["latest"]):
        if os.path.isfile(c):
            return c
    raise FileNotFoundError("No trained checkpoint yet. Train first: python main.py --mode train")


def archive_old_run(model_dir: str, reason: str) -> None:
    """Move old checkpoints aside (never delete) so a fresh run starts clean."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(model_dir, "_old", stamp)
    moved = []
    for name in ("latest.pt", "best.pt", "latest.pt.meta.json", "best.pt.meta.json", "tantra.pt",
                 "probe_history.jsonl", "training_status.json",
                 # leftovers from the old layout — moved aside once, automatically
                 "Latest", "Best", "Checkpoints", "Export", "Experts", "Adapters", "tokenizer_32k",
                 "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json",
                 "memory_bank.json", "corpus_sample.txt"):
        src = os.path.join(model_dir, name)
        if os.path.exists(src):
            os.makedirs(dest, exist_ok=True)
            shutil.move(src, os.path.join(dest, name))
            moved.append(name)
    if moved:
        log.warning(f"{reason} Moved {', '.join(moved)} -> {os.path.relpath(dest, ROOT)}")


def build_config(args, vocab_size: int) -> NeuroCoreConfig:
    if args.preset == "billion":
        cfg = NeuroCoreConfig.billion(vocab_size)
    elif args.preset == "tiny":
        cfg = NeuroCoreConfig.tiny()
        cfg.vocab.vocab_size = cfg.vocab.byte_bpe_vocab = vocab_size
    else:
        cfg = NeuroCoreConfig.small(vocab_size)
    if args.dim or args.layers or args.heads:
        cfg.set_shape(args.dim or cfg.block.alra.dim, args.layers or cfg.block.num_layers,
                      args.heads or cfg.block.alra.num_heads)
    if args.local_attn_every is not None:
        cfg.block.alra.local_attn_every = args.local_attn_every
    if args.local_window:
        cfg.block.alra.local_window = args.local_window
    return cfg


# Files from the old (v1) layout. On the first run they are MOVED (not deleted) to _old_code/.
# Check that folder, then delete it yourself.
LEGACY = [
    "ARCHITECTURE.md", "ROADMAP.md", "SECURITY.md", "CONTRIBUTING.md", "CHANGELOG.md", "pyproject.toml",
    "benchmark.py", "chat.py", "train.bat", "tantra.ps1", "tantra_kaggle_training.ipynb", "tools", ".benchmarks",
    # (Tantra/Smriti.py from v1 is NOT listed: Windows paths ignore case and v2 has Tantra/smriti.py)
    "Tantra/Chitta.py", "Tantra/CognitiveOS.py", "Tantra/Manas.py", "Tantra/Nirikshak.py",
    "Tantra/Vivek.py", "Tantra/config_0926b.py", "Tantra/benchmark.py", "Tantra/cli_hardware_dispatch.py",
    "Tantra/codec.py", "Tantra/moe.py", "Tantra/tool_router.py", "Tantra/ui.py", "Tantra/probe_eval.py",
    "Tests/test_checkpoint_and_chat_loader.py", "Tests/test_core_architecture.py", "Tests/test_export_benchmark.py",
    "Tests/test_omnimodal_tools.py", "Tests/test_real_learning_and_gradients.py", "Tests/test_system_integration.py",
    "Tests/test_training_alignment.py", "Tests/test_v2_core.py",
    "WebUI/install_webui_autostart.ps1", "WebUI/start_webui.ps1", "WebUI/TantraLLMWebUI.cmd",
    "Datasets/documents", "Assets/tantra_architecture.jpg",
    "Assets/tantra_hero_banner_v1.1_weaving_intelligence_20260807.jpg",
    "Assets/tantra_hero_banner_v1.2_bold_title_20260807.jpg",
]


def move_legacy_files() -> None:
    dest = os.path.join(ROOT, "_old_code")
    moved = []
    for rel in LEGACY:
        src = os.path.join(ROOT, rel)
        if os.path.exists(src):
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target):   # already archived once: leave it, never delete
                continue
            shutil.move(src, target)
            moved.append(rel)
    for cache in glob.glob(os.path.join(ROOT, "*", "__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)   # Python rebuilds these automatically
    if moved:
        log.warning(f"Moved {len(moved)} old v1 files to _old_code/ — delete that folder when you are happy.")


# ── train ────────────────────────────────────────────────────────────────────

def run_train(args, hw) -> None:
    from Tantra.dataset import JSONLDataset
    from Tantra.eval_suite import load_probe, run_probe
    from Tantra.evolution import AutoGrowthController
    from Tantra.train import NeuroTrainer

    if not os.path.isfile(TOKENIZER_PATH):
        sys.exit("Model/tokenizer.json is missing. Build it once: python main.py --mode tokenizer")
    tok = load_tokenizer(TOKENIZER_PATH)
    paths = ckpt_paths(args.model_dir)

    resume_from, model = None, None
    if os.path.isfile(paths["latest"]) and not args.fresh:
        model, _ = load_model(paths["latest"], device=hw.device)
        rows = model.embed.weight.shape[0]
        if rows != tok.vocab_size:
            archive_old_run(args.model_dir, f"Old checkpoint uses a {rows:,}-token vocab, tokenizer has {tok.vocab_size:,}.")
            model = None
        else:
            resume_from = paths["latest"]
            log.info(f"Continuing training from {os.path.relpath(resume_from, ROOT)}")
    elif args.fresh:
        archive_old_run(args.model_dir, "--fresh requested.")

    if model is None:
        if not args.fresh:
            archive_old_run(args.model_dir, "Starting a new model.")
        cfg = build_config(args, tok.vocab_size)
        model = NeuroCoreModel(cfg, use_mtp=args.mtp).to(hw.device)
        log.info(f"New model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params | "
                 f"dim {cfg.block.alra.dim} x {cfg.block.num_layers} layers | vocab {tok.vocab_size:,} | "
                 f"softmax window {cfg.block.alra.local_window} every {cfg.block.alra.local_attn_every} layers")

    if args.adapter:
        if args.adapter not in model.category_layers:
            model.add_category_layers([args.adapter], depth=1, clone_layer_index=len(model.layers) - 1)
        model.freeze_for_category(args.adapter)
        log.info(f"Training only the '{args.adapter}' category layer (base model frozen).")

    trainer = NeuroTrainer(model, lr=args.lr, weight_decay=args.weight_decay, optimizer_name=args.optimizer,
                           total_steps=args.steps, warmup_steps=args.warmup, grad_accumulation_steps=args.grad_accum,
                           use_mtp_loss=bool(unwrap_model(model).use_mtp), max_grad_norm=args.max_grad_norm)
    if resume_from:
        trainer.load_checkpoint(resume_from)
        trainer.lr, trainer.warmup_steps = args.lr, args.warmup
        trainer.total_steps = max(args.steps, trainer.step_count + 1)
        trainer._set_lr()
        if trainer.step_count >= args.steps:
            sys.exit(f"Already at step {trainer.step_count:,}. Raise --steps to train further.")

    train_files = [f for f in args.data.split(",") if f]
    train_ds = JSONLDataset(train_files, tok, seq_len=args.seq_len, stage=args.stage, seed=args.seed + trainer.step_count)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers,
                                         persistent_workers=args.workers > 0)
    val_loader = None
    if args.val and os.path.isfile(args.val):
        val_ds = JSONLDataset(args.val, tok, seq_len=args.seq_len, stage=args.stage, shuffle_buffer=1, loop=False)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size)
    else:
        log.warning("No validation file — validation loss will not be measured.")

    probe = load_probe(args.probe) if args.probe else []
    history = os.path.join(args.model_dir, "probe_history.jsonl")

    def on_eval(step: int, metrics: dict) -> None:
        if probe:
            run_probe(unwrap_model(model), tok, probe, step,
                      generate=(step % (args.eval_every * 4) == 0), history_path=history)
        trainer.save_checkpoint(paths["latest"])
        if metrics.get("is_best"):
            trainer.save_checkpoint(paths["best"], save_optimizer=False)

    growth = AutoGrowthController(plateau_patience=args.growth_patience, max_layers=args.max_layers) \
        if args.auto_growth else None
    try:
        trainer.fit(loader, max_steps=args.steps, log_every=args.log_every, eval_every=args.eval_every,
                    val_loader=val_loader, val_batches=args.val_batches, on_eval=on_eval,
                    growth=growth, early_stopping_patience=args.early_stop)
    except KeyboardInterrupt:
        log.warning("Stopped by user.")
    finally:
        trainer.save_checkpoint(paths["latest"])
        log.info("Saved. Run the same command again to continue.")


# ── chat / generate ──────────────────────────────────────────────────────────

def _reply(model, tok, prompt_text: str, args, device) -> str:
    ids = torch.tensor([tok.encode(prompt_text)], device=device)
    out = []
    for t in model.generate_stream(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                   top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                                   eos_token_id=EOS_ID):
        t = int(t)
        if t == EOS_ID:
            break
        out.append(t)
        piece = tok.decode([t])
        print(piece, end="", flush=True)
    print()
    return tok.decode(out)


def run_chat(args, hw) -> None:
    from Tantra.adapters import AdapterRegistry, RequestRouter
    from Tantra.dataset import chat_prompt
    tok = load_tokenizer(TOKENIZER_PATH)
    path = args.checkpoint or default_checkpoint(args.model_dir)
    model, _ = load_model(path, hw.device, int8=args.int8)
    router = RequestRouter(AdapterRegistry()) if model.category_layers else None
    print(f"Loaded {os.path.relpath(path, ROOT)}. Type your message. /reset clears history, /quit exits.\n")
    history = []
    while True:
        try:
            q = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/reset":
            history.clear()
            continue
        if router is not None:
            model.active_category = router.route(q)
        print("Tantra > ", end="", flush=True)
        t0 = time.time()
        answer = _reply(model, tok, chat_prompt(q, args.system, history[-args.history:]), args, hw.device)
        history.append((q, answer))
        log.debug(f"{time.time() - t0:.1f}s")


def run_generate(args, hw) -> None:
    from Tantra.dataset import chat_prompt
    tok = load_tokenizer(TOKENIZER_PATH)
    model, _ = load_model(args.checkpoint or default_checkpoint(args.model_dir), hw.device, int8=args.int8)
    _reply(model, tok, chat_prompt(args.prompt or "नमस्ते", args.system), args, hw.device)


# ── eval / export / dpo / tokenizer / adapter ────────────────────────────────

def run_eval(args, hw) -> None:
    from Tantra.dataset import JSONLDataset
    from Tantra.eval_suite import load_probe, run_probe, throughput, validation_metrics
    tok = load_tokenizer(TOKENIZER_PATH)
    path = args.checkpoint or default_checkpoint(args.model_dir)
    model, ckpt = load_model(path, hw.device, int8=args.int8)
    report = {"checkpoint": os.path.relpath(path, ROOT), "step": ckpt.get("step_count"),
              "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    if args.val and os.path.isfile(args.val):
        ds = JSONLDataset(args.val, tok, seq_len=args.seq_len, stage=args.stage, shuffle_buffer=1, loop=False)
        report["validation"] = validation_metrics(model, torch.utils.data.DataLoader(ds, batch_size=4),
                                                  max_batches=args.val_batches)
    if args.probe:
        report["probe"] = run_probe(model, tok, load_probe(args.probe), report["step"] or 0, generate=True)
    report["speed"] = throughput(model, tok.vocab_size)
    report["finished_at"] = time.time()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open(os.path.join(args.model_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)   # shown in the WebUI Model tab


def run_export(args) -> None:
    from Tantra.export import export_clean_checkpoint
    src = args.checkpoint or default_checkpoint(args.model_dir)
    export_clean_checkpoint(src, args.output or os.path.join(args.model_dir, "tantra.pt"))


def run_dpo(args, hw) -> None:
    from Tantra.dataset import DPODataset
    from Tantra.train import NeuroTrainer
    tok = load_tokenizer(TOKENIZER_PATH)
    path = args.checkpoint or default_checkpoint(args.model_dir)
    model, _ = load_model(path, hw.device)
    trainer = NeuroTrainer(model, lr=args.lr if args.lr != 3e-4 else 5e-6, warmup_steps=10,
                           total_steps=args.steps, grad_accumulation_steps=args.grad_accum)
    trainer.load_checkpoint(path, reset_optimizer=True)
    loader = torch.utils.data.DataLoader(DPODataset(args.prefs, tok, max_len=args.seq_len), batch_size=args.batch_size)
    paths = ckpt_paths(args.model_dir)
    out = paths["latest"]
    trainer.train_dpo(loader, max_steps=args.steps, on_checkpoint=lambda s: trainer.save_checkpoint(out))
    trainer.save_checkpoint(out)


def run_tokenizer(args) -> None:
    if os.path.isfile(TOKENIZER_PATH):
        backup = os.path.join(MODEL_DIR, "_old", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(backup, exist_ok=True)
        for name in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"):
            if os.path.isfile(os.path.join(MODEL_DIR, name)):
                shutil.move(os.path.join(MODEL_DIR, name), os.path.join(backup, name))
        log.warning(f"Old tokenizer moved to {os.path.relpath(backup, ROOT)}. Existing checkpoints will NOT work "
                    f"with the new tokenizer — the next training run starts fresh.")
    build_tokenizer([f for f in args.data.split(",") if f], MODEL_DIR, vocab_size=args.vocab_size)


def run_data(args) -> None:
    from Tantra.data_prep import build
    build(DATA_DIR)
    print("\nDone. Next: python main.py --mode tokenizer (once), then python main.py --mode train --stage pretrain")


def run_smriti(args) -> None:
    from Tantra.smriti import build as build_smriti
    files = [f for f in (args.data or "").split(",") if f] or \
        [p for p in (os.path.join(DATA_DIR, "sft.jsonl"), os.path.join(DATA_DIR, "pretrain.jsonl")) if os.path.isfile(p)]
    if not files:
        sys.exit("No data. Run python main.py --mode data first (or pass --data file.jsonl).")
    print(json.dumps(build_smriti(files, os.path.join(args.model_dir, "smriti.db")), indent=2, ensure_ascii=False))


def run_adapter(args) -> None:
    from Tantra.adapters import AdapterRegistry, build_adapter_checkpoint
    reg = AdapterRegistry()
    reg.seed_defaults()
    if args.adapter_action == "install":
        src = args.checkpoint or default_checkpoint(args.model_dir)
        dst = ckpt_paths(args.model_dir)["latest"]
        print(build_adapter_checkpoint(src, dst, vocab_size=load_tokenizer(TOKENIZER_PATH).vocab_size))
        return
    for c in reg.all():
        print(f"{c.name:18s} depth {c.depth}  {c.status:10s} {c.description}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Tantra LLM", formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--mode", default="train",
                   choices=["train", "chat", "generate", "eval", "serve", "export", "data", "tokenizer", "smriti",
                            "dpo", "adapter", "hardware"])
    # data
    p.add_argument("--data", help="training .jsonl, comma-separated for several (default: Datasets/pretrain.jsonl "
                                  "or sft.jsonl for the stage)")
    p.add_argument("--val", help="held-out .jsonl, never trained on (default: Datasets/val_<stage>.jsonl)")
    p.add_argument("--probe", default=os.path.join(DATA_DIR, "probe_50.jsonl"), help="fixed test questions ('' to disable)")
    p.add_argument("--prefs", default=os.path.join(DATA_DIR, "preference_pairs.jsonl"), help="DPO chosen/rejected pairs")
    p.add_argument("--stage", choices=["sft", "pretrain"], default="sft", help="sft = learn answers only; pretrain = learn all text")
    # model
    p.add_argument("--preset", choices=["small", "billion", "tiny"], default="small")
    p.add_argument("--dim", type=int)
    p.add_argument("--layers", type=int)
    p.add_argument("--heads", type=int)
    p.add_argument("--local-attn-every", type=int, help="every Nth layer uses exact sliding-window attention (0 = off)")
    p.add_argument("--local-window", type=int)
    p.add_argument("--mtp", action="store_true", help="extra head predicting 2 tokens ahead (slower on CPU)")
    p.add_argument("--vocab-size", type=int, default=64000, help="tokenizer mode only (keep 64000 forever)")
    # training
    p.add_argument("--fresh", action="store_true", help="start a new model (old checkpoints moved to Model/_old)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, help="warm-up steps (default: min(500, steps/10))")
    p.add_argument("--optimizer", choices=["adamw", "lion", "sgd"], default="adamw")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--val-batches", type=int, default=50)
    p.add_argument("--early-stop", type=int, default=0, help="stop after N evals without val improvement (0 = never)")
    p.add_argument("--auto-growth", action="store_true")
    p.add_argument("--growth-patience", type=int, default=1000)
    p.add_argument("--max-layers", type=int, default=24)
    p.add_argument("--adapter", help="train only this category's specialist layer")
    p.add_argument("--adapter-action", choices=["list", "install"], default="list")
    # runtime
    p.add_argument("--checkpoint", help="checkpoint to use (default: tantra.pt, else best.pt, else latest.pt)")
    p.add_argument("--model-dir", default=MODEL_DIR)
    p.add_argument("--output")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int)
    p.add_argument("--workers", type=int, default=0, help="data-loading worker processes")
    p.add_argument("--seed", type=int, default=42)
    # generation
    p.add_argument("--prompt")
    p.add_argument("--system", help="optional system prompt")
    p.add_argument("--history", type=int, default=3, help="chat turns remembered in the prompt")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--int8", action="store_true", help="CPU: run with 8-bit weights (~2x faster)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--lan", action="store_true", help="serve: also reachable from your phone on the same Wi-Fi (key required)")
    args = p.parse_args()

    print_banner()
    move_legacy_files()
    set_seed(args.seed)
    if args.mode == "data":
        return run_data(args)
    if args.mode == "smriti":
        return run_smriti(args)
    if args.mode == "tokenizer" and not args.data:   # learn words from ALL cleaned data
        args.data = ",".join(p for p in (os.path.join(DATA_DIR, "pretrain.jsonl"), os.path.join(DATA_DIR, "sft.jsonl"))
                             if os.path.isfile(p)) or None
    default_data(args)
    if args.warmup is None:
        args.warmup = min(500, max(1, args.steps // 10))
    if args.mode == "tokenizer":
        return run_tokenizer(args)
    if args.mode == "adapter":
        return run_adapter(args)
    if args.mode == "export":
        return run_export(args)
    hw = detect_hardware(args.device, args.threads)
    if args.mode == "hardware":
        print(json.dumps(hw.as_dict(), indent=2))
    elif args.mode == "train":
        run_train(args, hw)
    elif args.mode == "chat":
        run_chat(args, hw)
    elif args.mode == "generate":
        run_generate(args, hw)
    elif args.mode == "eval":
        run_eval(args, hw)
    elif args.mode == "dpo":
        run_dpo(args, hw)
    elif args.mode == "serve":
        os.environ["TANTRA_INT8"] = "1" if args.int8 else os.environ.get("TANTRA_INT8", "0")
        from WebUI.server import start_server
        start_server(host="0.0.0.0" if args.lan else "127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
