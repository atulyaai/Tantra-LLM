"""
main.py — Tantra-LLM CLI entry point.

Usage:
    python main.py                     # full pipeline test
    python main.py --mode probe        # hardware auto-detection & profiling
    python main.py --mode vocab        # build & save vocabulary
    python main.py --mode train        # run training steps with auto-growth & repair
    python main.py --mode dataset      # pre-train on real JSONL dataset
    python main.py --mode eval         # evaluate perplexity & throughput benchmark
    python main.py --mode compress     # run DNA compression benchmark
    python main.py --mode generate     # generate text tokens (MTP 2x speed)
    python main.py --mode serve        # start local Web UI & REST API server
    python main.py --mode status       # dashboard
    python main.py --mode experts      # list experts
    python main.py --mode chat         # interactive REPL
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import torch
import torch._dynamo

from Tantra.config import NeuroCoreConfig, VocabConfig, MoEConfig, CompressionConfig
from Tantra.utils import get_logger
from Tantra.hardware import HardwareDetector, Profiler, RuntimeConfigBuilder, AdaptiveScheduler
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.model import NeuroCoreModel
from Tantra.moe import ExpertRegistry, LazyExpertLoader
from Tantra.codec import DNACodec, CompressionBenchmark
from Tantra.train import NeuroTrainer
from Tantra.dataset import JSONLDataset, extract_corpus_sample, PretokenizedBinDataset, find_bin_cache
from Tantra.evolution import AutoGrowthController, SelfRepairEngine, CategoryGrowthController
from Tantra.eval import EvaluationEngine
from Tantra.adapters import AdapterRegistry, RequestRouter, DEFAULT_CATEGORIES

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress
    console = Console()
except ImportError:
    console = None

log = get_logger("tantra")

MODEL_DIR       = os.path.join(os.path.dirname(__file__), "Model")
BEST_DIR        = os.path.join(MODEL_DIR, "Best")
LATEST_DIR      = os.path.join(MODEL_DIR, "Latest")
CHECKPOINTS_DIR = os.path.join(MODEL_DIR, "Checkpoints")
EXPERTS_DIR     = os.path.join(MODEL_DIR, "Experts")
_datasets_dir = os.path.join(os.path.dirname(__file__), "Datasets")
has_topics = any(entry.is_dir() for entry in os.scandir(_datasets_dir)) if os.path.exists(_datasets_dir) else False

if has_topics:
    DEFAULT_DATASET = _datasets_dir
else:
    _default_cand = os.path.join(_datasets_dir, "train_pack_all_expanded_1040k.jsonl")
    if not os.path.exists(_default_cand):
        _found = glob.glob(os.path.join(_datasets_dir, "*.jsonl"))
        DEFAULT_DATASET = _found[0] if _found else os.path.join(_datasets_dir, "tantra_master_identity_safety.jsonl")
    else:
        DEFAULT_DATASET = _default_cand

def print_banner():
    is_tty = getattr(sys.stdout, "isatty", lambda: False)()
    if console and is_tty:
        try:
            console.print(Panel.fit(
                "  [bold cyan]तन्त्र[/bold cyan]  [bold]TANTRA LLM[/bold]\n"
                "  [dim]NeuroCore Architecture • CPU-First • Local AI[/dim]",
                title="[yellow]Initializing[/yellow]",
                border_style="cyan"
            ))
            return
        except Exception:
            pass
    print("╔══════════════════════════════════════════════════════╗")
    print("║  तन्त्र  TANTRA LLM                                ║")
    print("║  NeuroCore Architecture • CPU-First • Local AI      ║")
    print("╚══════════════════════════════════════════════════════╝")

def print_status_dashboard(model, trainer, expert_reg, rt):
    if console:
        table = Table(title="Tantra-LLM Status Dashboard", show_header=False)
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        total_params = sum(p.numel() for p in model.parameters())
        status = "FRESH" if trainer.step_count == 0 else f"RESUMING (Step {trainer.step_count})"
        
        table.add_row("Model", f"NeuroCore ({total_params/1e6:.1f}M params)")
        table.add_row("Device", f"{rt.device} | dtype: {rt.dtype}")
        table.add_row("Training Status", status)
        table.add_row("Best Loss", f"{trainer.best_loss:.4f}")
        table.add_row("Total Tokens", f"{trainer.total_tokens:,}")
        table.add_row("Experts", f"{len(expert_reg)} registered")
        table.add_row("Hardware", f"{rt.offload_strategy} | Batch: {rt.batch_size}")
        
        console.print(table)
    else:
        print("== TANTRA-LLM STATUS ==")
        print(f"Status: {'FRESH' if trainer.step_count == 0 else 'RESUMING (Step ' + str(trainer.step_count) + ')'}")

def print_expert_panel(expert_reg):
    if console:
        table = Table(title="Expert Registry", show_header=True, header_style="bold magenta")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Specialization")
        table.add_column("Params")
        table.add_column("Usage")
        table.add_column("Status")
        table.add_column("DNA File")
        
        for e_id, e_info in expert_reg.experts.items():
            spec = e_info.get("specialization", "unknown")
            emoji = "🧠"
            if "language" in spec: emoji = "💬"
            elif "code" in spec: emoji = "💻"
            elif "math" in spec: emoji = "🔢"
            
            table.add_row(
                str(e_id),
                e_info.get("name", f"expert_{e_id}"),
                f"{emoji} {spec}",
                f"{e_info.get('param_count', 0)/1e6:.1f}M",
                str(e_info.get("usage_count", 0)),
                "ACTIVE",
                e_info.get("dna_path", "None")
            )
        console.print(table)
    else:
        print("== EXPERT REGISTRY ==")
        for e_id, e_info in expert_reg.experts.items():
            print(f"Expert {e_id}: {e_info}")

def run_interactive_chat(model, tokenizer, device, temp=0.8, top_p=0.95, router=None):
    if console:
        console.print("[bold green]Tantra Interactive REPL[/bold green] (Type /help for commands, /quit to exit)")
    else:
        print("Tantra Interactive REPL (Type /help for commands, /quit to exit)")
        
    while True:
        try:
            if console:
                prompt_text = "[bold cyan]तन्त्र >[/bold cyan] "
                user_input = console.input(prompt_text)
            else:
                user_input = input("तन्त्र > ")
                
            if not user_input.strip():
                continue
                
            if user_input.startswith("/"):
                cmd = user_input.split()[0].lower()
                if cmd == "/quit":
                    break
                elif cmd == "/help":
                    msg = "Commands: /help, /status, /experts, /settings, /quit"
                    if console: console.print(msg)
                    else: print(msg)
                elif cmd == "/settings":
                    parts = user_input.split()
                    if len(parts) >= 3:
                        temp = float(parts[1])
                        top_p = float(parts[2])
                    msg = f"Settings: temp={temp}, top_p={top_p}"
                    if console: console.print(msg)
                    else: print(msg)
                elif cmd == "/status":
                    msg = "Run main.py --mode status for full dashboard."
                    if console: console.print(msg)
                    else: print(msg)
                elif cmd == "/experts":
                    msg = "Run main.py --mode experts for full panel."
                    if console: console.print(msg)
                    else: print(msg)
                continue
                
            # Request-level routing: pick ONE domain adapter, base as fallback.
            routed = None
            if router is not None:
                if hasattr(model, "category_layers") and model.category_layers:
                    routed = router.route(user_input)
                    model.active_category = routed
            if console and routed is not None:
                console.print(f"[dim]→ routed to adapter: {routed}[/dim]")
            elif console:
                console.print("[dim]→ base (no adapter)[/dim]")

            formatted_input = f"<|user|>\n{user_input}\n<|assistant|>\n"
            tokens = tokenizer.encode(formatted_input)
            prompt = torch.tensor([tokens], device=device)
            if console:
                console.print(f"[dim]Thinking...[/dim]")
            with torch.no_grad():
                out = model.generate(prompt, max_new_tokens=200, temperature=temp, use_mtp_speculation=False)
            # generate() returns prompt + continuation concatenated; only decode
            # the newly generated tail so the REPL doesn't echo the user's input.
            new_tokens = out[0, prompt.shape[1]:].tolist()
            response = tokenizer.decode(new_tokens)
            if console:
                console.print(f"[bold yellow]Assistant:[/bold yellow] {response}")
            else:
                print(f"Assistant: {response}")
                
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            if console:
                console.print(f"[red]Error: {str(e)}[/red]")
            else:
                print(f"Error: {str(e)}")
            continue
                
            # Request-level routing: pick ONE domain adapter, base as fallback.
            routed = None
            if router is not None:
                if hasattr(model, "category_layers") and model.category_layers:
                    routed = router.route(user_input)
                    model.active_category = routed
            if console and routed is not None:
                console.print(f"[dim]→ routed to adapter: {routed}[/dim]")
            elif console:
                console.print("[dim]→ base (no adapter)[/dim]")

            formatted_input = f"<s><|user|>\n{user_input}\n<|assistant|>\n"
            tokens = tokenizer.encode(formatted_input)
            prompt = torch.tensor([tokens], device=device)
            if console:
                console.print(f"[dim]Thinking...[/dim]")
            with torch.no_grad():
                out = model.generate(prompt, max_new_tokens=200, temperature=temp, use_mtp_speculation=False)
            # generate() returns prompt + continuation concatenated; only decode
            # the newly generated tail so the REPL doesn't echo the user's input.
            new_tokens = out[0, prompt.shape[1]:].tolist()
            response = tokenizer.decode(new_tokens)
            if console:
                console.print(f"[bold yellow]Assistant:[/bold yellow] {response}")
            else:
                print(f"Assistant: {response}")
                
        except (KeyboardInterrupt, EOFError):
            break

def detect_hardware():
    log.info("== [1] HARDWARE AUTO-DETECTION & PROACTIVE HEALTH ==")
    
    is_colab = ('google.colab' in sys.modules 
                or os.environ.get('COLAB_RELEASE_TAG') is not None
                or os.environ.get('COLAB_GPU') is not None
                or os.path.exists('/content'))
    is_non_tty = not getattr(sys.stdout, "isatty", lambda: False)()
    
    if is_colab or is_non_tty:
        log.info("  [INFO] Running in Container/Non-TTY mode. Skipping benchmarks for instant startup.")
        from Tantra.hardware import GPUInfo, RuntimeConfig
        
        # Check for GPU without blocking
        has_cuda = False
        gpus = []
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                gpus = [GPUInfo(0, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // (1024*1024), "8.0", "cuda")]
        except Exception:
            pass
            
        device = 'cuda:0' if gpus else 'cpu'
        log.info(f"  Detected Device: {device} | strategy: {'full_gpu' if gpus else 'cpu_only'}")
        
        rt = RuntimeConfig(
            device=device,
            dtype='bfloat16' if gpus else 'int8',
            use_bitnet=True,
            batch_size=4 if gpus else 1,
            max_seq_len=8192,
            active_experts=1,
            expert_cache_size=8,
            prefetch_depth=2,
            compression_level='high',
            offload_strategy='full_gpu' if gpus else 'cpu_only',
            ram_budget_mb=8192,
            vram_budget_mb=12000,
            expert_size_mb=500,
            num_threads=4,
            prefill_chunk_size=512,
            profile_name="COLAB-GPU" if gpus else "COLAB-CPU"
        )
        
        class FastScheduler:
            def start(self): pass
            def stop(self): pass
            
        return rt, FastScheduler()

    hw = HardwareDetector()
    profile = hw.detect()
    hw.print_profile(profile)
    perf = Profiler(profile).run()
    rt = RuntimeConfigBuilder().build(profile, perf)
    log.info(f"  Strategy   : {rt.offload_strategy}")
    log.info(f"  Device     : {rt.device} | dtype: {rt.dtype}")
    log.info(f"  Compression: {rt.compression_level}")
    log.info(f"  Expert Cache: {rt.expert_cache_size} in RAM | batch: {rt.batch_size}")
    
    sched = AdaptiveScheduler(rt)
    sched.start()
    return rt, sched


def build_vocab(cfg: VocabConfig, corpus_file: str | None = None) -> UnifiedTokenizer:
    log.info("== [2] UNIFIED VOCABULARY & TOKENIZER STATUS =======")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)
    os.makedirs(LATEST_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    tokenizer_json_path = os.path.join(MODEL_DIR, "tokenizer.json")
    status = "Cached Artifact"

    if os.path.exists(tokenizer_json_path):
        try:
            bpe = ByteBPETokenizer.load(tokenizer_json_path, cfg)
            status = f"Loaded real BPE tokenizer ({tokenizer_json_path}, {bpe.vocab_size:,} tokens)"
        except Exception as e:
            log.warning(f"Failed to load tokenizer.json ({e}). Retraining...")
            bpe = ByteBPETokenizer(cfg)
    else:
        bpe = ByteBPETokenizer(cfg)

    if (not os.path.exists(tokenizer_json_path) or bpe.vocab_size == 0 or bpe._tokenizer is None or bpe._tokenizer.get_vocab_size() == 0) and corpus_file and os.path.exists(corpus_file):
        # Dataset mode commonly receives the Datasets directory.  Select a
        # real JSONL corpus rather than attempting ``open(Datasets)`` when a
        # tokenizer has to be rebuilt.
        resolved_corpus = corpus_file
        if os.path.isdir(corpus_file):
            candidates = [
                path for path in glob.glob(os.path.join(corpus_file, "**", "*.jsonl"), recursive=True)
                if "_duplicates" not in os.path.normpath(path).split(os.sep)
            ]
            if not candidates:
                raise RuntimeError(f"No JSONL files found under dataset directory: {corpus_file}")
            resolved_corpus = max(candidates, key=os.path.getsize)
            log.info(f"  Tokenizer rebuild source selected from dataset directory: {resolved_corpus}")
        sample_txt = extract_corpus_sample(resolved_corpus, os.path.join(MODEL_DIR, "corpus_sample.txt"))
        special_toks = list(cfg.special_tokens.keys())
        bpe.train([sample_txt], vocab_size=cfg.vocab_size, special_tokens=special_toks)
        bpe.save(tokenizer_json_path)
        status = f"Trained fresh BPE tokenizer on {resolved_corpus} & saved to {tokenizer_json_path}"

    patcher = MegabytePatcher()
    tok = UnifiedTokenizer(cfg, bpe, patcher)
    log.info(f"  Vocab Size       : {bpe.vocab_size:,} tokens")
    log.info(f"  BPE Subword Merges: {bpe.vocab_size - len(cfg.special_tokens) - 256:,} merge rules")
    log.info(f"  Special Tokens   : {len(cfg.special_tokens)} (<pad>, <unk>, <s>, </s>, <|user|>, <|assistant|>, <|system|>)")
    log.info(f"  Byte Patching    : Megabyte Patching Unit Enabled (byte-fallback handling)")
    log.info(f"  Tokenizer Status : {status}")
    log.info(f"  Artifact Path    : {tokenizer_json_path}")
    return tok


def init_experts(moe_cfg, model_cfg, codec):
    log.info("== [3] EXPERT REGISTRY & LAZY LOADER =============")
    os.makedirs(EXPERTS_DIR, exist_ok=True)
    reg = ExpertRegistry(EXPERTS_DIR, moe_cfg.num_experts)
    reg.load()
    # Domain experts map 1:1 to the topic dataset folders in Datasets/.
    # This lets a topic's data route to a dedicated expert for specialization.
    DOMAIN_SPECS = [
        "general", "code", "math", "science", "reasoning",
        "creative_writing", "conversation", "multilingual", "instructions", "safety",
    ]
    if len(reg) == 0:
        for i, spec in enumerate(DOMAIN_SPECS):
            reg.register_new(i, spec, 2_000_000_000)
        log.info(f"  Registered {len(reg)} domain experts: {', '.join(DOMAIN_SPECS)}")
        
    sample_expert_weight = torch.randn(1024, 1024, dtype=torch.float32)
    dna_path = os.path.join(EXPERTS_DIR, "expert_0.dna")
    if not os.path.exists(dna_path):
        codec.compress(sample_expert_weight, dna_path)
        log.info(f"  Compressed expert_0 weight tensor -> {dna_path}")
        
    return reg, LazyExpertLoader(moe_cfg, model_cfg, reg, codec)


ADAPTER_ROOT = os.path.join(MODEL_DIR, "MoE2_32K")
ADAPTER_CHECKPOINT = os.path.join(ADAPTER_ROOT, "checkpoint_adapters.pt")


def run_adapter_mode(action: str, name: str | None = None, description: str = "", topics: str | None = None, rank: int = 32, keywords: str | None = None) -> None:
    """Manage routeable adapter categories (add/list/remove/init)."""
    from Tantra.adapters import build_adapter_checkpoint
    registry = AdapterRegistry()
    registry.seed_defaults()

    if action == "list":
        rows = registry.all()
        print(f"\nRegistered adapter categories ({len(rows)}):")
        for cat in rows:
            topic_str = ",".join(cat.topics)
            print(f"  - {cat.name:<18} [{cat.status}] rank={cat.rank} params={cat.params/1e6:.2f}M topics={topic_str}")
        print("Request router: one category per request, base as fallback.")
        return

    if action == "add":
        if not name:
            raise ValueError("--name is required to add a category.")
        topic_list = [t.strip() for t in (topics or name).split(",") if t.strip()]
        kw_list = [k.strip() for k in (keywords or "").split(",") if k.strip()]
        registry.add(name, description=description, topics=topic_list, rank=rank, keywords=kw_list)
        print(f"Added category '{name}'. Train it with: python main.py --mode dataset --adapter {name}")
        return

    if action == "remove":
        if not name:
            raise ValueError("--name is required to remove a category.")
        ok = registry.remove(name)
        print(f"{'Removed' if ok else 'Not found'}: {name}")
        return

    if action == "init":
        if not os.path.exists(os.path.join(ADAPTER_ROOT, "checkpoint_init.pt")):
            raise FileNotFoundError("Base checkpoint Model/MoE2_32K/checkpoint_init.pt not found. Run the profile converter first.")
        result = build_adapter_checkpoint(
            os.path.join(ADAPTER_ROOT, "checkpoint_init.pt"),
            ADAPTER_CHECKPOINT,
            vocab_size=32768,
        )
        print("Adapter checkpoint initialized with default categories.")
        return

    raise ValueError(f"Unknown adapter action: {action}")


def build_adapter_model(rt, vocab_size: int = 32768):
    """Load the MoE-2 / 32K base with installed specialist layers."""
    from Tantra.model import build_cpu_model
    if not os.path.exists(ADAPTER_CHECKPOINT):
        raise FileNotFoundError(f"Adapter checkpoint not found: {ADAPTER_CHECKPOINT}. Run: python main.py --mode adapter init")
    model = build_cpu_model("moe2", attention_kind="alra", vocab_size=vocab_size)
    # Submodules must exist before the checkpoint (which contains category_layers.*)
    # can be loaded; otherwise those keys are silently skipped.
    registry = AdapterRegistry()
    registry.seed_defaults()
    model.add_category_layers([c.name for c in registry.all()], clone_layer_index=model.config.adapter.clone_layer_index)
    ckpt = torch.load(ADAPTER_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.sync_category_gates_from_checkpoint(ckpt["model_state_dict"])
    model = model.to(rt.device)
    return model


def init_model(cfg, device):
    log.info("== [4] NEUROCORE MODEL ENGINE & PARAMETER DIAGNOSTICS ==")
    model = NeuroCoreModel(cfg, use_mtp=getattr(cfg, "use_mtp", True))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    log.info(f"  Total Parameters     : {total_params:,} ({total_params/1e6:.1f}M)")
    log.info(f"  Trainable Parameters : {trainable_params:,}")
    log.info(f"  Frozen Parameters    : {frozen_params:,}")
    log.info(f"  Model Architecture   : {cfg.block.num_layers} NeuroCore Blocks | {cfg.block.alra.dim} Embed Dim | {cfg.block.alra.num_heads} Attention Heads")
    log.info(f"  Attention Engine     : ALRA (Adaptive Linear Resonance Attention) [O(1) Memory Scan]")
    log.info(f"  Feed-Forward Engine  : SGP (Sparse Gated Projection) + BitNet 1.58-bit Ternary Quantization")
    log.info(f"  Speculative Engine   : Multi-Token Prediction (MTP 2x Acceleration)")
    log.info(f"  Target Device        : {device}")
    return model.to(device)


def restore_checkpoint_architecture(cfg, checkpoint_path: str) -> None:
    """Apply lightweight saved architecture metadata before model creation."""
    meta_path = checkpoint_path + ".meta.json"
    if not os.path.exists(meta_path):
        return
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        saved_layers = int(metadata.get("num_layers", cfg.block.num_layers))
        if saved_layers >= 1 and saved_layers != cfg.block.num_layers:
            cfg.block.num_layers = saved_layers
            log.info("Checkpoint architecture restored: %d layers.", saved_layers)
    except Exception as exc:
        log.warning("Could not read checkpoint architecture metadata (%s); using configured depth.", exc)


def run_forward(model, vcfg, batch_size, device):
    log.info("== [FORWARD PASS DIAGNOSTICS] ======================")
    x, y = generate_synthetic_batch(vcfg.vocab_size, batch_size=batch_size, seq_len=64)
    x, y = x.to(device), y.to(device)
    log.info(f"  Input Batch Shape  : {list(x.shape)}")
    with torch.no_grad():
        out = model(x, return_mtp=True)
    logits, mtp_logits = out[0]
    log.info(f"  Main Logits Shape  : {list(logits.shape)}  ✓")
    log.info(f"  MTP Logits Shape   : {list(mtp_logits.shape)}  ✓")


def run_training(model, vcfg, steps=30, resume=False):
    log.info("== [SYNTHETIC BENCHMARK TRAINING] ==================")
    trainer = NeuroTrainer(model, lr=1e-4, total_steps=steps)
    latest_ckpt = os.path.join(LATEST_DIR, "checkpoint_latest.pt")
    
    if resume and os.path.exists(latest_ckpt):
        log.info(f"RESUMING training from existing checkpoint: {latest_ckpt}")
        trainer.load_checkpoint(latest_ckpt)
    else:
        log.info("Starting FRESH synthetic training benchmark.")

    trainer.train_demo(steps=steps, vocab_size=vcfg.vocab_size)
    trainer.save_checkpoint(latest_ckpt, save_optimizer=True)


def run_dataset_training(model, tokenizer, dataset_path, steps=50, resume=False, eval_every=1000, log_every=50, checkpoint_every=500, batch_size=1, seq_len=128, grad_accumulation_steps=1, data_workers=0, use_latent_reasoning=True, use_mtp_loss=True, compile=False, lr=1e-4, warmup_steps=None, topic_weights=None, training_stage="sft", auto_growth=False, growth_patience=1000, growth_min_delta=0.005, max_layers=None, model_dir=None, adapter_name=None, archive_checkpoints=True):
    log.info("== [DATASET PRE-TRAINING MODE] =====================")
    if training_stage not in {"pretrain", "sft"}:
        raise ValueError(f"Unknown training stage: {training_stage}")
    mask_non_assistant = training_stage == "sft"
    stage_label = "full-token pretraining" if not mask_non_assistant else "assistant-only instruction tuning"
    log.info(f"Loading real dataset from: {dataset_path} ({stage_label})")
    if tokenizer.bpe.vocab_size == 0 or tokenizer.bpe._tokenizer is None or tokenizer.bpe._tokenizer.get_vocab_size() == 0:
        log.error(f"CRITICAL: BPE Tokenizer is untrained (vocab_size 0)! Aborting training to prevent raw-byte fallback.")
        raise RuntimeError("Tokenizer is falling back to raw bytes (no valid BPE merges found). Please generate a valid tokenizer.json before training.")
    repair = SelfRepairEngine()
    repair.scan_and_repair(model)

    # Per-category bidirectional growth: when a specialist stack plateaus it
    # grows (harder categories) and shrinks when converged but idle. The base
    # AutoGrowthController is disabled here because the base is frozen.
    cat_growth_ctrl = None
    cat_meta = None
    if adapter_name is not None and auto_growth:
        cat_growth_ctrl = CategoryGrowthController(
            plateau_patience=max(50, growth_patience), min_delta=growth_min_delta)
        reg = AdapterRegistry()
        cat_meta = reg.get(adapter_name)
        auto_growth = False  # handled per-category below, not on the frozen base

    warmup = warmup_steps if warmup_steps is not None else max(50, steps // 10)
    log.info(f"Learning rate: {lr:.2e}  |  Warmup steps: {warmup}")
    trainer = NeuroTrainer(model, lr=lr, total_steps=steps, warmup_steps=warmup, grad_accumulation_steps=grad_accumulation_steps, use_latent_reasoning=use_latent_reasoning, use_mtp_loss=use_mtp_loss)

    checkpoint_root = os.path.abspath(model_dir or MODEL_DIR)
    latest_dir = os.path.join(checkpoint_root, "Latest")
    best_dir = os.path.join(checkpoint_root, "Best")
    checkpoints_dir = os.path.join(checkpoint_root, "Checkpoints")
    for directory in (latest_dir, best_dir, checkpoints_dir):
        os.makedirs(directory, exist_ok=True)
    latest_ckpt = os.path.join(latest_dir, "checkpoint_latest.pt")
    best_ckpt = os.path.join(best_dir, "checkpoint_best.pt")
    
    # Resume only when explicitly requested.  Automatically restoring an
    # instruction-tuning checkpoint for a new broad pretraining stage carries
    # over a mismatched LR schedule and can erase/generalize poorly from an
    # already overfit state.
    resume_target = None
    if resume:
        step_checkpoints = sorted(
            glob.glob(os.path.join(checkpoints_dir, "checkpoint_step_*.pt")),
            key=os.path.getmtime,
            reverse=True,
        )
        candidates = [latest_ckpt, *step_checkpoints, best_ckpt]
        seen = set()
        for candidate in candidates:
            if candidate in seen or not os.path.isfile(candidate):
                continue
            seen.add(candidate)
            try:
                log.info(f"Trying recovery checkpoint: {candidate}")
                trainer.load_checkpoint(candidate)
                resume_target = candidate
                break
            except Exception as exc:
                log.warning(f"Skipping unreadable checkpoint {candidate}: {exc}")
        if resume_target is None:
            log.warning("--resume was requested, but no readable checkpoint was found in Model/. Starting fresh training run from step 1.")

    if resume_target:
        log.info(f"RESUMING training from recovered checkpoint: {resume_target}")
        if steps <= trainer.step_count:
            effective_target = trainer.step_count + steps
            log.info(f"  [Incremental Steps] Specified --steps {steps} <= checkpoint step {trainer.step_count}. "
                     f"Running +{steps} steps -> new target: {effective_target} steps.")
            steps = effective_target
        if training_stage == "sft":
            log.info(f"  [SFT Stage] Re-initializing optimizer & scheduler for instruction fine-tuning (LR={lr:.2e}, warmup={warmup}).")
            trainer.lr = lr
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
            sft_steps = max(steps - trainer.step_count, 100)
            actual_warmup = max(1, min(warmup, sft_steps // 5))
            warmup_sched = torch.optim.lr_scheduler.LinearLR(trainer.optimizer, start_factor=0.05, end_factor=1.0, total_iters=actual_warmup)
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=max(sft_steps - actual_warmup, 50), eta_min=1e-6)
            trainer.scheduler = torch.optim.lr_scheduler.SequentialLR(trainer.optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[actual_warmup])
    else:
        log.info("Starting fresh dataset training run.")

    if compile:
        torch._dynamo.config.suppress_errors = True
        has_compiler = shutil.which("g++") or shutil.which("cl") or shutil.which("gcc")
        if not has_compiler:
            log.info("  [PyTorch Inductor] Note: No C++ compiler (g++/cl) detected on Windows PATH.")
            log.info("  Running in fast eager mode. Install w64devkit (g++) or MSVC for Inductor C++ compilation.")
        else:
            log.info("  [PyTorch Inductor] Compiling model with torch.compile(backend='inductor')...")
            try:
                trainer.model = torch.compile(trainer.model, backend="inductor")
            except Exception as e:
                log.warning(f"  torch.compile failed ({e}), continuing uncompiled.")

    def eval_callback(step):
        # This is a qualitative generation sample, formatted with standard chat tokens.
        log.info("\n--- [ SAMPLE GENERATION @ Step %d ] ---" % step)
        prompt_text = "<|user|>\nWhat is Tantra?\n<|assistant|>\n"
        log.info("Prompt: User: What is Tantra?")
        prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], device=model.embed.weight.device)
        out = model.generate(prompt_ids, max_new_tokens=64, min_new_tokens=15, temperature=0.7, top_p=0.9, use_mtp_speculation=True)
        # Only decode the newly generated continuation, not the echoed prompt.
        new_tokens = out[0, prompt_ids.shape[1]:].tolist()
        response = tokenizer.decode(new_tokens)
        log.info("Output: %s" % response)
        log.info("----------------------------------\n")

        # Bidirectional per-category growth (adapter training only). During a
        # single-category run every token routes to this category, so usage is
        # always "high" and the controller will only GROW a plateauing stack.
        # SHRINK is exercised under multi-category routing (see unit tests and
        # the chat/serve eval path).
        if cat_growth_ctrl is not None and cat_meta is not None and trainer.ema_loss is not None:
            tokens_so_far = max(1, step) * 2048
            decision = cat_growth_ctrl.observe(
                adapter_name, float(trainer.ema_loss), cat_routed=tokens_so_far,
                total_routed=tokens_so_far, depth=model.category_depth(adapter_name),
                min_depth=cat_meta.min_depth, max_depth=cat_meta.max_depth)
            if decision == "grow" and model.grow_category(adapter_name, cat_meta.max_depth):
                model.freeze_for_category(adapter_name)
                trainer.refresh_optimizer()
                new_depth = model.category_depth(adapter_name)
                new_params = sum(p.numel() for p in model.category_layers[adapter_name].parameters())
                reg = AdapterRegistry()
                reg.update_depth(adapter_name, new_depth, new_params)
                reg.save()
                log.info(f"[CategoryGrowth] Grew '{adapter_name}' to depth {new_depth} (params={new_params}).")
            elif decision == "shrink" and model.shrink_category(adapter_name, cat_meta.min_depth):
                model.freeze_for_category(adapter_name)
                trainer.refresh_optimizer()
                new_depth = model.category_depth(adapter_name)
                new_params = sum(p.numel() for p in model.category_layers[adapter_name].parameters())
                reg = AdapterRegistry()
                reg.update_depth(adapter_name, new_depth, new_params)
                reg.save()
                log.info(f"[CategoryGrowth] Shrank '{adapter_name}' to depth {new_depth} (params={new_params}).")
        
        # Avoid saving immediately upon resuming (step == trainer.step_count when just loaded)
        # We only save if we have actually progressed.
        if step > 0:
            # Save to Latest (full state with optimizer for seamless resume)
            trainer.save_checkpoint(latest_ckpt, save_optimizer=True)
            
            if archive_checkpoints:
                # Optional archive copies; CPU profiles use only Latest by
                # default to avoid spending disk on repeated optimizer state.
                step_ckpt = os.path.join(checkpoints_dir, f"checkpoint_step_{step}.pt")
                trainer.save_checkpoint(step_ckpt, save_optimizer=True)
                if (trainer.ema_loss is not None and trainer.ema_loss <= trainer.best_loss) or step % (eval_every * 4) == 0 or step == steps:
                    version_name = f"Tantra_v1_step_{step}.pt"
                    trainer.save_checkpoint(os.path.join(best_dir, version_name), save_optimizer=False)
                    trainer.save_checkpoint(best_ckpt, save_optimizer=False)
            
            trainer._last_saved_step = step

    def checkpoint_callback(step):
        """Persist the exact resumable state without an expensive sample run."""
        if step == getattr(trainer, "_last_saved_step", -1):
            # The evaluation callback has just created the same exact latest
            # recovery state. Do not write another multi-gigabyte file.
            return
        trainer.save_checkpoint(latest_ckpt, save_optimizer=True)
        trainer._last_saved_step = step
        log.info("Recovery checkpoint saved at step %d.", step)

    # Record the starting step so we don't save it immediately
    trainer._last_saved_step = trainer.step_count
    
    # Generation is expensive on CPU and is not informative before base
    # pretraining. Keep the immediate sample for instruction tuning only.
    if training_stage == "sft":
        eval_callback(trainer.step_count)

    from Tantra.dataset import TopicMixedDataset
    
    # ``steps`` counts optimizer updates, while an IterableDataset yields
    # individual samples.  Gradient accumulation consumes multiple complete
    # batches per update, so the old cap stopped runs early when --grad-accum
    # was above one.
    max_samples = steps * batch_size * max(1, grad_accumulation_steps)

    if os.path.isdir(dataset_path):
        # Scan for topic subdirectories
        topic_paths = {}
        for entry in os.scandir(dataset_path):
            if entry.is_dir():
                topic = entry.name
                jsonls = glob.glob(os.path.join(entry.path, "*.jsonl"))
                if jsonls:
                    topic_paths[topic] = jsonls
        
        if topic_paths:
            log.info(f"  Topic directories found: {list(topic_paths.keys())}")
            # When training a single category, restrict to that category's
            # registered topic folders and freeze the shared base; only the
            # dedicated specialist layer for this category is trained.
            if adapter_name is not None:
                registry = AdapterRegistry()
                cat = registry.get(adapter_name)
                if cat is None:
                    raise ValueError(f"Adapter category '{adapter_name}' is not in the registry. Run: python main.py --mode adapter add --name {adapter_name}")
                if not hasattr(model, "category_layers") or adapter_name not in model.category_layers:
                    raise ValueError(f"Adapter checkpoint has no category layer '{adapter_name}'. Run: python main.py --mode adapter init")
                allowed = set(cat.topics)
                topic_paths = {t: p for t, p in topic_paths.items() if t in allowed}
                if not topic_paths:
                    raise ValueError(f"Category '{adapter_name}' maps to topics {sorted(allowed)} but none exist under {dataset_path}.")
                log.info(f"  Training category '{adapter_name}' on topics: {list(topic_paths.keys())}")
                model.freeze_for_category(adapter_name)
                dataset = TopicMixedDataset(topic_paths, {t: 1.0 for t in topic_paths}, tokenizer, seq_len=seq_len,
                                            max_samples=max_samples, mask_non_assistant=mask_non_assistant)
            else:
                # Default weights bias toward the biggest, most general corpus while
                # giving every domain expert meaningful exposure. Override with the
                # --topic-weights flag if you want a different mixture.
                DEFAULT_TOPIC_WEIGHTS = {
                "general": 40.0,
                "code": 15.0,
                "math": 15.0,
                "science": 8.0,
                "reasoning": 8.0,
                "creative_writing": 4.0,
                "conversation": 4.0,
                "multilingual": 3.0,
                "instructions": 2.0,
                "safety": 1.0,
            }
            if topic_weights:
                weights = {t: float(topic_weights.get(t, DEFAULT_TOPIC_WEIGHTS.get(t, 1.0))) for t in topic_paths.keys()}
                log.info(f"  Custom topic weights: {weights}")
            else:
                weights = {t: DEFAULT_TOPIC_WEIGHTS.get(t, 1.0) for t in topic_paths.keys()}
            dataset = TopicMixedDataset(topic_paths, weights, tokenizer, seq_len=seq_len,
                                        max_samples=max_samples, mask_non_assistant=mask_non_assistant)
        else:
            # Fallback to single file if no subdirectories with jsonl found
            fallback = glob.glob(os.path.join(dataset_path, "*.jsonl"))
            if fallback:
                dataset = JSONLDataset(fallback[0], tokenizer, seq_len=seq_len,
                                      max_samples=max_samples, mask_non_assistant=mask_non_assistant)
            else:
                dataset = JSONLDataset(dataset_path, tokenizer, seq_len=seq_len,
                                      max_samples=max_samples, mask_non_assistant=mask_non_assistant)
    else:
        bin_cache = find_bin_cache(dataset_path)
        if bin_cache:
            log.info(f"  Pre-tokenized cache found -> {bin_cache} (skipping BPE encode() at train time)")
            dataset = PretokenizedBinDataset(bin_cache, seq_len=seq_len,
                                             max_samples=max_samples,
                                             mask_non_assistant=mask_non_assistant)
        else:
            dataset = JSONLDataset(dataset_path, tokenizer, seq_len=seq_len,
                                  max_samples=max_samples, mask_non_assistant=mask_non_assistant)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=data_workers,
        persistent_workers=data_workers > 0, prefetch_factor=4 if data_workers > 0 else None,
    )
    try:
        trainer.train_dataset(dataloader, max_steps=steps, log_every=log_every, eval_every=eval_every, eval_callback=eval_callback, checkpoint_every=checkpoint_every, checkpoint_callback=checkpoint_callback, tokenizer=tokenizer, use_latent_reasoning=use_latent_reasoning, auto_growth=auto_growth, growth_patience=growth_patience, growth_min_delta=growth_min_delta, max_layers=max_layers)
    except KeyboardInterrupt:
        # Ctrl+C happens after an optimizer boundary in many practical runs.
        # Save that completed state before allowing the process to stop.
        trainer.save_checkpoint(latest_ckpt, save_optimizer=True)
        log.warning("Training interrupted; recovery checkpoint saved at step %d.", trainer.step_count)

    if trainer.step_count > trainer._last_saved_step:
        trainer.save_checkpoint(latest_ckpt, save_optimizer=True)


def run_evaluation(model, tokenizer, dataset_path):
    log.info("== [MODEL EVALUATION & BENCHMARK MODE] =============")
    engine = EvaluationEngine(model)
    dataset = JSONLDataset(dataset_path, tokenizer, seq_len=128, max_samples=20) if os.path.exists(dataset_path) else None
    report = engine.print_benchmark_report(dataset, vocab_size=tokenizer.vocab_size)
    return report


def run_compression_benchmark(comp_cfg):
    log.info("== [COMPRESSION BENCHMARK] =========================")
    bench = CompressionBenchmark(comp_cfg)
    sample_weight = torch.randn(1024, 1024, dtype=torch.float32)
    bench.run(sample_weight, output_dir=os.path.join(MODEL_DIR, "reports"))


def run_generation(model, vcfg, device):
    log.info("── [TEXT GENERATION MODE (MTP Speculation)] ───────")
    prompt = torch.randint(0, vcfg.vocab_size, (1, 4), device=device)
    log.info(f"  Prompt tokens: {prompt.tolist()[0]}")
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=10, temperature=0.8, use_mtp_speculation=True)
    log.info(f"  Generated sequence tokens: {out.tolist()[0]}  ✓")


def serve(model, tokenizer, port=8000, expert_dir=None):
    log.info("== [PRODUCTION WEB SERVER & DASHBOARD MODE] =========")
    log.info(f"  Launching Interactive Web UI & OpenAI REST API on http://localhost:{port}")
    try:
        from webui.server import start_server
        start_server(host="0.0.0.0", port=port)
    except Exception as e:
        log.error(f"Failed to start Tantra web server: {e}")


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="Tantra-LLM / NeuroCore CLI Engine")
    parser.add_argument("--mode", default="full",
                        choices=["full", "probe", "vocab", "train", "dataset", "eval", "compress", "generate", "serve", "status", "experts", "chat", "adapter"],
                        help="Execution mode")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="JSONL dataset path")
    parser.add_argument("--steps", type=int, default=30, help="Training steps")
    parser.add_argument("--seq-len", type=int, default=128, help="Context sequence length window")
    parser.add_argument("--use-mtp", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable Multi-Token Prediction (MTP)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling")
    parser.add_argument("--port", type=int, default=8000, help="Server port (serve mode)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    parser.add_argument("--eval-every", type=int, default=1000, help="Run a qualitative generation sample and archive checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=10, help="Print a rolling training summary every N optimizer steps")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Save a resumable recovery checkpoint every N optimizer steps (0 disables; default 500 minimizes I/O overhead)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (larger effective batch without more RAM; 1 = off)")
    parser.add_argument("--data-workers", type=int, default=0, help="Parallel data-loading/tokenization workers (overlaps tokenization with training compute; 0 = synchronous/main-thread, as before)")
    parser.add_argument("--training-stage", choices=["pretrain", "sft"], default="sft", help="pretrain uses full-token loss; sft supervises assistant replies only")
    parser.add_argument("--latent-reasoning", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable latent reasoning. Defaults off for pretraining and on for SFT.")
    parser.add_argument("--mtp-loss", action=argparse.BooleanOptionalAction, default=None, help="Train the MTP auxiliary head. Defaults off for pretraining and on for SFT.")
    parser.add_argument("--auto-growth", action="store_true", help="Add one layer only after a sustained loss plateau; saved layers resume correctly.")
    parser.add_argument("--growth-patience", type=int, default=1000, help="Optimizer steps to observe before auto-growth may add a layer")
    parser.add_argument("--growth-min-delta", type=float, default=0.005, help="Minimum EMA-loss improvement required to avoid auto-growth")
    parser.add_argument("--max-layers", type=int, default=None, help="Hard maximum depth when --auto-growth is enabled")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile(backend='inductor') for CPU/GPU kernel fusion")
    parser.add_argument("--device", type=str, default="auto", help="Target device (auto, cuda, cpu, mps)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4; use 5e-5 or lower for fine-tuning)")
    parser.add_argument("--warmup", type=int, default=None, help="LR warmup steps (default: steps // 10)")
    parser.add_argument("--topic-weights", type=str, default=None, help="JSON dict of topic weights, e.g. '{\"general\":40,\"code\":15}'")
    parser.add_argument("--model-dir", type=str, default=None, help="Custom root directory for model checkpoints (e.g. Google Drive)")
    parser.add_argument("--adapter-action", default="list", choices=["list", "add", "remove", "init"],
                        help="--mode adapter sub-action")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Category to train (dataset mode) or force for chat/generate. None routes per-request.")
    parser.add_argument("--adapter-desc", type=str, default="", help="Description when adding a category")
    parser.add_argument("--adapter-topics", type=str, default=None, help="Comma list of Datasets/<topic> folders for a new category")
    parser.add_argument("--adapter-keywords", type=str, default=None, help="Comma list of routing keywords for a new category")
    args = parser.parse_args()

    vcfg = VocabConfig()
    mcfg = NeuroCoreConfig()
    mcfg.use_mtp = args.use_mtp
    moe  = MoEConfig()
    ccfg = CompressionConfig()

    # Adapter management needs no model/hardware; handle it immediately.
    if args.mode == "adapter":
        run_adapter_mode(
            args.adapter_action,
            name=args.adapter,
            description=args.adapter_desc,
            topics=args.adapter_topics,
            rank=32,
            keywords=args.adapter_keywords,
        )
        return

    if args.mode == "probe":
        detect_hardware()
        return

    if args.mode == "vocab":
        build_vocab(vcfg, args.dataset)
        return

    if args.mode == "compress":
        run_compression_benchmark(ccfg)
        return

    rt, sched = detect_hardware()
    
    # ── Hybrid Device Selection: GPU if available, else CPU ──
    if args.device == "auto":
        # Auto-detect best available device
        if torch.cuda.is_available():
            rt.device = "cuda:0"
            log.info(f"  [HYBRID] CUDA GPU detected → using {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            rt.device = "mps"
            log.info(f"  [HYBRID] Apple MPS detected → using Metal GPU")
        else:
            rt.device = "cpu"
            log.info(f"  [HYBRID] No GPU found → using CPU ({os.cpu_count()} threads)")
    else:
        # Manual override with validation
        requested = args.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            log.warning(f"  [DEVICE] CUDA requested but not available! Falling back to CPU.")
            rt.device = "cpu"
        elif requested == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            log.warning(f"  [DEVICE] MPS requested but not available! Falling back to CPU.")
            rt.device = "cpu"
        else:
            rt.device = requested
            log.info(f"  [DEVICE OVERRIDE] Target device explicitly set to: {rt.device}")
    tok = build_vocab(vcfg, args.dataset)
    codec = DNACodec(ccfg)
    reg, loader = init_experts(moe, mcfg, codec)
    # The persisted registry is the authoritative MoE layout for local
    # checkpoints.  Keeping the model at ``small()``'s default of 10 while
    # the registry/checkpoint contains 8 experts makes checkpoint restore
    # fail on every router tensor.
    if len(reg) > 0:
        mcfg.moe.num_experts = len(reg)
    latest_ckpt_file = os.path.join(args.model_dir or MODEL_DIR, "Latest", "checkpoint_latest.pt")
    restore_checkpoint_architecture(mcfg, latest_ckpt_file)
    # When the latest checkpoint embeds its own architecture config (as the
    # CPU-profile trainer saves now), rebuild the model from it instead of
    # NeuroCoreConfig.small(). Otherwise a checkpoint trained as 38.6M causal /
    # 512-dim loads into the 178M ALRA skeleton and every tensor mismatches,
    # silently leaving the model at random weights (garbage chat output).
    _ckpt_path = latest_ckpt_file
    if os.path.exists(_ckpt_path) and os.path.getsize(_ckpt_path) > 10 * 1024 * 1024 and mcfg is not None:
        try:
            _ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(_ckpt, dict):
                _ckpt_cfg = _ckpt.get("config", None)
                if _ckpt_cfg is not None:
                    _ckpt_cfg.vocab.vocab_size = vcfg.vocab_size
                    mcfg = _ckpt_cfg
                    log.info("Rebuilt model architecture from checkpoint config "
                             f"(dim={_ckpt_cfg.block.alra.dim}, layers={_ckpt_cfg.block.num_layers}, vocab={_ckpt_cfg.vocab.vocab_size}).")
        except Exception as _exc:
            log.warning(f"Could not read checkpoint config: {_exc}; using default architecture.")
    model = init_model(mcfg, rt.device)

    # When a category is requested for dataset/chat/generate/serve, load the
    # MoE-2 / 32K adapter checkpoint (shared base + specialist layers) instead
    # of the 178M general model.
    if args.adapter is not None and args.mode in ("dataset", "chat", "generate", "serve"):
        model = build_adapter_model(rt)

    trainer = NeuroTrainer(model, lr=1e-4)
    # Check if a checkpoint exists for status — use LATEST_DIR constant (capital L)
    # not the literal "latest" path which never matches on Windows.
    if args.mode == "status":
        latest_ckpt_status = os.path.join(args.model_dir or MODEL_DIR, "Latest", "checkpoint_latest.pt")
        if os.path.exists(latest_ckpt_status):
            try:
                trainer.load_checkpoint(latest_ckpt_status)
            except Exception as e:
                log.warning(f"Could not load latest checkpoint for status: {e}")
        print_status_dashboard(model, trainer, reg, rt)
        sched.stop()
        return
        
    if args.mode == "experts":
        print_expert_panel(reg)
        sched.stop()
        return
        
    if args.mode == "chat":
        if args.adapter is not None:
            if args.adapter not in model.category_layers:
                log.warning(f"Category '{args.adapter}' not in adapter checkpoint; ignoring --adapter.")
            else:
                model.active_category = args.adapter
        else:
            router = RequestRouter(AdapterRegistry())
            router._model = model  # allow per-request routing to set active category
            run_interactive_chat(model, tok, rt.device, args.temperature, args.top_p, router=router)
        sched.stop()
        return

    if args.mode == "train":
        run_training(model, vcfg, steps=args.steps, resume=args.resume)
    elif args.mode == "dataset":
        topic_weights = None
        if args.topic_weights:
            import json as _json
            try:
                topic_weights = _json.loads(args.topic_weights)
            except Exception as e:
                log.warning(f"Could not parse --topic-weights ({e}); using defaults.")
        use_latent_reasoning = args.latent_reasoning
        if use_latent_reasoning is None:
            use_latent_reasoning = args.training_stage == "sft"
        use_mtp_loss = args.mtp_loss
        if use_mtp_loss is None:
            use_mtp_loss = args.training_stage == "sft"
        run_dataset_training(model, tok, args.dataset, steps=args.steps, resume=args.resume, eval_every=args.eval_every, log_every=args.log_every, checkpoint_every=args.checkpoint_every, batch_size=args.batch_size, seq_len=args.seq_len, grad_accumulation_steps=args.grad_accum, data_workers=args.data_workers, use_latent_reasoning=use_latent_reasoning, use_mtp_loss=use_mtp_loss, compile=args.compile, lr=args.lr, warmup_steps=args.warmup, topic_weights=topic_weights, training_stage=args.training_stage, auto_growth=args.auto_growth, growth_patience=args.growth_patience, growth_min_delta=args.growth_min_delta, max_layers=args.max_layers, adapter_name=args.adapter, model_dir=(ADAPTER_ROOT if args.adapter is not None else args.model_dir))
    elif args.mode == "eval":
        run_evaluation(model, tok, args.dataset)
    elif args.mode == "generate":
        run_generation(model, vcfg, rt.device)
    elif args.mode == "serve":
        serve(model, tok, port=args.port, expert_dir=EXPERTS_DIR)
    else:  # full mode
        run_forward(model, vcfg, rt.batch_size, rt.device)
        run_evaluation(model, tok, args.dataset)
        run_generation(model, vcfg, rt.device)

    log.info("Pipeline complete -- NeuroCore ready!")
    sched.stop()


if __name__ == "__main__":
    main()
