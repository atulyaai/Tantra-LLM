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
import time
import torch
import torch._dynamo

from Tantra.config import NeuroCoreConfig, VocabConfig, MoEConfig, CompressionConfig
from Tantra.utils import get_logger
from Tantra.hardware import HardwareDetector, Profiler, RuntimeConfigBuilder, AdaptiveScheduler
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.model import NeuroCoreModel, cpu_dense_config, build_cpu_model
from Tantra.moe import ExpertRegistry, LazyExpertLoader
from Tantra.codec import DNACodec, CompressionBenchmark
from Tantra.train import NeuroTrainer
from Tantra.dataset import JSONLDataset, extract_corpus_sample, PretokenizedBinDataset, find_bin_cache
from Tantra.evolution import AutoGrowthController, SelfRepairEngine, CategoryGrowthController
from Tantra.eval_suite import EvaluationEngine
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
        if _found:
            _found.sort(key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0, reverse=True)
            DEFAULT_DATASET = _found[0]
        else:
            DEFAULT_DATASET = os.path.join(_datasets_dir, "tantra_master_identity_safety.jsonl")
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

def run_interactive_chat(model, tokenizer, device, temp=0.7, top_p=0.9, router=None, use_mtp=False):
    model.eval()
    if console:
        console.print("[bold green]╭───────────────────────────────────────────────────╮[/bold green]")
        console.print("[bold green]│  तन्त्र  TANTRA LLM Interactive Terminal Playground │[/bold green]")
        console.print("[bold green]╰───────────────────────────────────────────────────╯[/bold green]")
        console.print("[dim]Commands: /temp <float>, /mtp <on|off>, /clear, /stats, /help, /quit[/dim]\n")
    else:
        print("== TANTRA LLM Interactive Terminal Playground ==")
        print("Commands: /temp <float>, /mtp <on|off>, /clear, /stats, /help, /quit\n")

    while True:
        try:
            if console:
                user_input = console.input("[bold cyan]You >[/bold cyan] ")
            else:
                user_input = input("You > ")

            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                parts = user_input.strip().split()
                cmd = parts[0].lower()
                if cmd in ("/quit", "/exit"):
                    break
                elif cmd == "/help":
                    msg = "Commands: /temp <float> (adjust creativity), /mtp <on|off> (speculative speedup), /stats (show parameters), /clear, /quit"
                    if console: console.print(f"[dim]{msg}[/dim]")
                    else: print(msg)
                elif cmd == "/temp":
                    if len(parts) >= 2:
                        try:
                            temp = max(0.0, min(2.0, float(parts[1])))
                            if console: console.print(f"[green]Temperature set to {temp:.2f}[/green]")
                            else: print(f"Temperature set to {temp:.2f}")
                        except ValueError:
                            if console: console.print("[red]Invalid temperature value[/red]")
                elif cmd == "/mtp":
                    if len(parts) >= 2:
                        use_mtp = parts[1].lower() in ("on", "true", "1", "yes")
                        if console: console.print(f"[green]MTP speculative acceleration: {'ON' if use_mtp else 'OFF'}[/green]")
                        else: print(f"MTP speculative acceleration: {'ON' if use_mtp else 'OFF'}")
                elif cmd == "/clear":
                    if os.name == "nt": os.system("cls")
                    else: os.system("clear")
                elif cmd == "/stats":
                    total_p = sum(p.numel() for p in model.parameters())
                    if console: console.print(f"[magenta]Model: {total_p/1e6:.1f}M params | Device: {device} | Temp: {temp} | MTP: {use_mtp}[/magenta]")
                    else: print(f"Model: {total_p/1e6:.1f}M params | Device: {device} | Temp: {temp} | MTP: {use_mtp}")
                continue

            # Request-level routing: pick ONE domain adapter, base as fallback.
            routed = None
            if router is not None:
                if hasattr(model, "category_layers") and model.category_layers:
                    routed = router.route(user_input)
                    model.active_category = routed
            if console and routed is not None:
                console.print(f"[dim]→ routed to adapter: {routed}[/dim]")

            formatted_input = f"<|system|>\nYou are Tantra, a helpful, polite, and intelligent AI assistant.\n<|user|>\n{user_input}\n<|assistant|>\n"
            tokens = tokenizer.encode(formatted_input)
            prompt = torch.tensor([tokens], device=device)

            if console:
                console.print("[bold yellow]Tantra >[/bold yellow] ", end="")
            else:
                print("Tantra > ", end="", flush=True)

            t0 = time.perf_counter()
            generated_tokens = []
            with torch.no_grad():
                for token_id in model.generate_stream(prompt, max_new_tokens=256, temperature=temp, top_p=top_p, use_mtp_speculation=use_mtp):
                    tid = int(token_id.item() if hasattr(token_id, "item") else token_id)
                    if tid in (0, 2):  # <pad> or </s> (EOS)
                        break
                    generated_tokens.append(tid)
                    piece = tokenizer.decode([tid])
                    if any(stop_tag in piece for stop_tag in ["<|user|>", "<|system|>", "<|assistant|>", "</s>", "<s>"]):
                        break
                    if console:
                        console.print(piece, end="")
                    else:
                        print(piece, end="", flush=True)

            elapsed = max(time.perf_counter() - t0, 1e-4)
            tok_speed = len(generated_tokens) / elapsed

            if console:
                console.print(f"\n[dim]({len(generated_tokens)} tokens, {tok_speed:.1f} tok/s)[/dim]\n")
            else:
                print(f"\n({len(generated_tokens)} tokens, {tok_speed:.1f} tok/s)\n")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            if console:
                console.print(f"[red]Error: {str(e)}[/red]")
            else:
                print(f"Error: {str(e)}")
            continue


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
    is_real_moe = getattr(cfg.moe, "real_top1", False)
    num_exp = getattr(cfg.moe, "num_experts", 1)
    model = NeuroCoreModel(
        cfg,
        use_mtp=getattr(cfg, "use_mtp", True),
        use_moe=(is_real_moe and num_exp > 1)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    log.info(f"  Total Parameters     : {total_params:,} ({total_params/1e6:.1f}M)")
    log.info(f"  Trainable Parameters : {trainable_params:,}")
    log.info(f"  Frozen Parameters    : {frozen_params:,}")
    moe_label = f" | {num_exp} Real Top-1 MoE Experts" if (is_real_moe and num_exp > 1) else " (Dense)"
    log.info(f"  Model Architecture   : {cfg.block.num_layers} NeuroCore Blocks | {cfg.block.alra.dim} Embed Dim | {cfg.block.alra.num_heads} Attention Heads{moe_label}")
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


def run_dataset_training(model, tokenizer, dataset_path, steps=50, resume=False, eval_every=1000, log_every=50, checkpoint_every=500, batch_size=1, seq_len=128, grad_accumulation_steps=1, data_workers=0, use_latent_reasoning=True, use_mtp_loss=True, compile=False, lr=1e-4, weight_decay=0.01, optimizer="adamw", warmup_steps=None, topic_weights=None, training_stage="sft", auto_growth=False, growth_patience=1000, growth_min_delta=0.005, max_layers=None, model_dir=None, adapter_name=None, archive_checkpoints=True, pack_sequences=True, checkpoint_path=None, max_grad_norm=1.0, mtp_loss_weight=0.3, track=None, curriculum_phase=None):

    log.info("== [DATASET PRE-TRAINING MODE] =====================")
    if training_stage not in {"pretrain", "sft"}:
        raise ValueError(f"Unknown training stage: {training_stage}")
    mask_non_assistant = training_stage == "sft"
    stage_label = "full-token pretraining" if not mask_non_assistant else "assistant-only instruction tuning"
    log.info(f"Loading real dataset from: {dataset_path} ({stage_label})")
    if os.path.isfile(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8", errors="ignore") as _f:
                _line_count = sum(1 for _ in _f)
            log.info(f"Dataset '{os.path.basename(dataset_path)}' loaded with {_line_count:,} items.")
            if _line_count < 1000:
                log.warning(f"[STUB DATASET WARNING] '{os.path.basename(dataset_path)}' contains only {_line_count} lines! Small dataset runs loop every {max(1, _line_count // 8)} steps. Ensure full corpus is used for production training.")
        except Exception:
            pass

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
    log.info(f"Optimizer: {optimizer.upper()}  |  Learning rate: {lr:.2e}  |  Warmup steps: {warmup}  |  Grad clip: {max_grad_norm}  |  MTP weight: {mtp_loss_weight}")
    trainer = NeuroTrainer(model, lr=lr, weight_decay=weight_decay, optimizer_name=optimizer, total_steps=steps, warmup_steps=warmup, grad_accumulation_steps=grad_accumulation_steps, use_latent_reasoning=use_latent_reasoning, use_mtp_loss=use_mtp_loss, mtp_loss_weight=mtp_loss_weight, max_grad_norm=max_grad_norm)

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
    resume_target = None
    if resume or checkpoint_path:
        candidates = []
        if checkpoint_path and os.path.isfile(checkpoint_path):
            candidates.append(checkpoint_path)

        search_dirs = [checkpoint_root, checkpoints_dir, latest_dir, best_dir, MODEL_DIR, os.path.join(MODEL_DIR, "Checkpoints"), os.path.join(MODEL_DIR, "Latest"), os.path.join(MODEL_DIR, "Best")]
        for d in search_dirs:
            if os.path.exists(d):
                candidates.extend(glob.glob(os.path.join(d, "*.pt")))
                candidates.extend(glob.glob(os.path.join(d, "**", "*.pt"), recursive=True))

        def _get_step_num(p: str) -> int:
            import re
            m = re.search(r'step_(\d+)', os.path.basename(p))
            if m: return int(m.group(1))
            meta = p + ".meta.json"
            if os.path.exists(meta):
                try:
                    with open(meta, "r") as mf:
                        return int(json.load(mf).get("step", 0))
                except Exception: pass
            if "latest" in os.path.basename(p).lower(): return 999999999
            if "best" in os.path.basename(p).lower(): return 999999998
            return 0

        # Sort candidates descending by step count so highest milestone (e.g. step 31000) is loaded first
        sorted_candidates = sorted(list(set(candidates)), key=_get_step_num, reverse=True)
        seen = set()
        for candidate in sorted_candidates:
            if candidate in seen or not os.path.isfile(candidate) or "sample" in candidate:
                continue
            seen.add(candidate)
            try:
                log.info(f"Loading recovery checkpoint: {candidate} ({os.path.getsize(candidate)/1e6:.1f} MB)...")
                trainer.load_checkpoint(candidate)
                resume_target = candidate
                break
            except Exception as exc:
                log.warning(f"Skipping unreadable checkpoint {candidate}: {exc}")
        if resume_target is None:
            log.warning(f"--resume was requested, but no readable checkpoint was found in {checkpoint_root} or {MODEL_DIR}. Starting fresh training run from step 1.")

    if resume_target:
        log.info(f"RESUMING training from recovered checkpoint: {resume_target}")
        if steps <= trainer.step_count:
            effective_target = trainer.step_count + steps
            log.info(f"  [Incremental Steps] Specified --steps {steps} <= checkpoint step {trainer.step_count}. "
                     f"Running +{steps} steps -> new target: {effective_target} steps.")
            steps = effective_target
        prev_stage = getattr(trainer, "training_stage", None)
        if training_stage == "sft" and prev_stage != "sft" and prev_stage is not None:
            log.info(f"  [SFT Stage Transition] Transitioning from {prev_stage} -> sft. Re-initializing optimizer & scheduler (LR={lr:.2e}, warmup={warmup}).")
            trainer.lr = lr
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8)
            sft_steps = max(steps - trainer.step_count, 100)
            actual_warmup = max(1, min(warmup, sft_steps // 5))
            from Tantra.train import create_lr_scheduler
            trainer.scheduler = create_lr_scheduler(trainer.optimizer, warmup_steps=actual_warmup, total_steps=sft_steps, min_lr_ratio=0.01)
            trainer.training_stage = "sft"
        else:
            stage_name = training_stage or prev_stage or "pretrain"
            trainer.training_stage = stage_name
            # ── FIX #1 (CRITICAL): Always rebuild the scheduler whenever the step horizon
            # changes on resume.  The trainer was constructed earlier with the raw
            # command-line --steps value (e.g. 10,000) as total_steps.  After the
            # incremental-target math above, `steps` is now trainer.step_count + 10,000
            # (e.g. 82,427).  Without this rebuild the cosine schedule's progress
            # = current_step / old_total_steps > 1.0, which immediately clamps the LR
            # to min_lr_ratio (5 % of base), and every single training step runs at
            # that near-zero floor for the entire session.
            remaining = max(steps - trainer.step_count, 1)
            actual_warmup = max(1, min(warmup or max(remaining // 10, 100), remaining // 5))
            from Tantra.train import create_lr_scheduler
            old_total = getattr(trainer, "total_steps", 0)
            if steps != old_total:
                trainer.total_steps = steps
                trainer.warmup_steps = actual_warmup
                trainer.scheduler = create_lr_scheduler(
                    trainer.optimizer,
                    warmup_steps=actual_warmup,
                    total_steps=remaining,          # schedule over the *remaining* steps
                    min_lr_ratio=0.05,
                )
                log.info(
                    f"  [Schedule Corrected ✅] LR horizon rebuilt: remaining={remaining:,} steps, "
                    f"warmup={actual_warmup}, lr≈{lr:.2e} (was pinned at floor due to stale horizon)"
                )
            else:
                log.info(f"  [{stage_name.upper()} Stage Resume] Preserved AdamW optimizer momentum buffers and LR scheduler position across resume boundary.")
    else:
        trainer.training_stage = training_stage or "pretrain"
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
        # Evaluates 4 diverse domain prompts to monitor multi-skill emergence
        log.info(f"┌── 🌐 [ MULTI-DOMAIN & ZERO-SHOT WORLD BENCHMARK @ Step {step:,} ] " + "─" * 20)
        test_prompts = [
            ("General", "💬", "<|user|>\nWhat is Tantra LLM?\n\n<|assistant|>\n", 0.4),
            ("Coding",  "💻", "<|user|>\nWrite a Python function to reverse a string.\n\n<|assistant|>\n```python\n", 0.2),
            ("Math",    "🔢", "<|user|>\nSolve for x in 2x + 6 = 14.\n\n<|assistant|>\n", 0.2),
            ("Science", "🔬", "<|user|>\nState Newton's First Law of Motion.\n\n<|assistant|>\n", 0.4)
        ]
        raw_model = getattr(model, "module", getattr(model, "_orig_mod", model))
        for domain, icon, prompt_text, temp in test_prompts:
            prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], device=raw_model.embed.weight.device)
            out = raw_model.generate(prompt_ids, max_new_tokens=64, min_new_tokens=1, temperature=temp, top_p=0.9, repetition_penalty=1.15)
            new_tokens = out[0, prompt_ids.shape[1]:].tolist()
            response = tokenizer.decode(new_tokens).strip()

            extra_tag = ""
            if domain == "Coding":
                import ast
                code_cand = response.replace("```python", "").replace("```", "").strip()
                try:
                    ast.parse(code_cand)
                    extra_tag = " (✅ Valid Python AST)"
                except Exception:
                    pass
            elif domain == "Math":
                if "x = 4" in response or "x=4" in response or "= 4" in response:
                    extra_tag = " (✅ Solved: x = 4)"

            clean_disp = response.replace("\n", " ")[:90]
            log.info(f"│ {icon} [{domain:7s}]: {clean_disp}{extra_tag}")
        
        # Zero-Shot World Knowledge MMLU Benchmark Evaluation
        try:
            from Tantra.world_eval import evaluate_zero_shot_world_knowledge
            world_res = evaluate_zero_shot_world_knowledge(raw_model, tokenizer)
            if world_res:
                log.info(f"│ 🌍 [World MMLU]: 🏆 {world_res['world_mmlu_accuracy']:.1f}% Zero-Shot Accuracy ({world_res['correct_samples']}/{world_res['total_samples']} correct)")
        except Exception:
            pass
        log.info("└" + "─" * 80)




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
            # Save to Latest asynchronously (full state with optimizer for seamless resume)
            trainer.save_checkpoint(latest_ckpt, save_optimizer=True, async_write=True)
            
            if archive_checkpoints:
                # Optional archive copies; CPU profiles use only Latest by
                # default to avoid spending disk on repeated optimizer state.
                step_ckpt = os.path.join(checkpoints_dir, f"checkpoint_step_{step}.pt")
                trainer.save_checkpoint(step_ckpt, save_optimizer=True, async_write=True)
                if getattr(trainer, "is_new_best", False):
                    trainer.save_checkpoint(best_ckpt, save_optimizer=False, async_write=True)
                    log.info(f"🏆 [NEW BEST CHECKPOINT] Val Loss: {trainer.best_val_loss:.4f} -> {os.path.basename(best_ckpt)}")
                    trainer.is_new_best = False


                if step % (eval_every * 4) == 0 or step == steps:
                    version_name = f"Tantra_v1_step_{step}.pt"
                    trainer.save_checkpoint(os.path.join(best_dir, version_name), save_optimizer=False, async_write=True)


            
            trainer._last_saved_step = step

    def checkpoint_callback(step):
        """Persist the exact resumable state without an expensive sample run."""
        if step == getattr(trainer, "_last_saved_step", -1):
            # The evaluation callback has just created the same exact latest
            # recovery state. Do not write another multi-gigabyte file.
            return
        trainer.save_checkpoint(latest_ckpt, save_optimizer=True, async_write=True)
        trainer._last_saved_step = step
        log.info("Recovery checkpoint queued at step %d.", step)

    # Record the starting step so we don't save it immediately
    trainer._last_saved_step = trainer.step_count
    
    # Generation is expensive on CPU and is not informative before base
    # pretraining. Keep the immediate sample for instruction tuning only.
    if training_stage == "sft":
        eval_callback(trainer.step_count)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from Tantra.dataset import TopicMixedDataset
    
    # ``steps`` counts optimizer updates, while an IterableDataset yields
    # individual samples.  Gradient accumulation consumes multiple complete
    # batches per update, so the old cap stopped runs early when --grad-accum
    # was above one.
    max_samples = steps * batch_size * max(1, grad_accumulation_steps)

    dataset = None
    if track and track.lower() not in ("all", "none"):
        t_norm = track.lower().strip()
        log.info(f"🎯 [EXPERT TRACK OVERRIDE] Training exclusively on '{t_norm}' category...")
        track_map = {
            "chitchat": ["chitchat_phase1_greetings.jsonl", "expert_conversation.jsonl"] if curriculum_phase == 1 else (["chitchat_phase2_short.jsonl", "expert_conversation.jsonl"] if curriculum_phase == 2 else ["chitchat_phase3_full.jsonl", "expert_conversation.jsonl", "conversation.jsonl"]),
            "chitchat-p1": ["chitchat_phase1_greetings.jsonl"],
            "chitchat-p2": ["chitchat_phase2_short.jsonl"],
            "chitchat-p3": ["chitchat_phase3_full.jsonl", "expert_conversation.jsonl"],
            "greetings": ["chitchat_phase1_greetings.jsonl"],
            "conversation": ["chitchat_phase3_full.jsonl", "expert_conversation.jsonl", "conversation.jsonl"],
            "chat": ["chitchat_phase3_full.jsonl", "expert_conversation.jsonl", "conversation.jsonl"],
            "identity": ["chitchat_phase1_greetings.jsonl", "gold_corpus.jsonl", "expert_conversation.jsonl"],
            "math": ["math_phase1_arithmetic.jsonl", "expert_math_science.jsonl"] if curriculum_phase == 1 else (["math_phase2_wordproblems.jsonl", "expert_math_science.jsonl"] if curriculum_phase == 2 else ["math_phase3_advanced.jsonl", "expert_math_science.jsonl", "math.jsonl"]),
            "math-p1": ["math_phase1_arithmetic.jsonl"],
            "math-p2": ["math_phase2_wordproblems.jsonl"],
            "math-p3": ["math_phase3_advanced.jsonl", "expert_math_science.jsonl"],
            "code": ["code_phase1_syntax.jsonl", "expert_code.jsonl"] if curriculum_phase == 1 else (["code_phase2_algorithms.jsonl", "expert_code.jsonl"] if curriculum_phase == 2 else ["code_phase3_systems.jsonl", "expert_code.jsonl", "code.jsonl"]),
            "code-p1": ["code_phase1_syntax.jsonl"],
            "code-p2": ["code_phase2_algorithms.jsonl"],
            "code-p3": ["code_phase3_systems.jsonl", "expert_code.jsonl"],
            "science": ["science_phase1_fundamentals.jsonl", "expert_math_science.jsonl"] if curriculum_phase == 1 else (["science_phase2_explanations.jsonl", "expert_math_science.jsonl"] if curriculum_phase == 2 else ["science_phase3_advanced.jsonl", "expert_math_science.jsonl", "science.jsonl"]),
            "science-p1": ["science_phase1_fundamentals.jsonl"],
            "science-p2": ["science_phase2_explanations.jsonl"],
            "science-p3": ["science_phase3_advanced.jsonl", "expert_math_science.jsonl"],
            "general": ["expert_general.jsonl", "general.jsonl"],
            "gold": ["gold_corpus.jsonl"]
        }
        cand_names = track_map.get(t_norm, [f"expert_{t_norm}.jsonl", f"{t_norm}.jsonl"])
        base_dir = dataset_path if os.path.isdir(dataset_path) else (os.path.dirname(dataset_path) or "Datasets")
        selected_file = None
        for c in cand_names:
            c_path = os.path.join(base_dir, c)
            if os.path.isfile(c_path) and os.path.getsize(c_path) > 0:
                selected_file = c_path
                break
        
        # On-demand builder for specific domains
        if selected_file is None:
            if "code" in t_norm:
                from Tantra.dataset import build_phased_code_curriculum
                build_phased_code_curriculum(base_dir)
            elif "math" in t_norm:
                from Tantra.dataset import build_phased_math_curriculum
                build_phased_math_curriculum(base_dir)
            elif "science" in t_norm:
                from Tantra.dataset import build_phased_science_curriculum
                build_phased_science_curriculum(base_dir)
            elif t_norm in ("chitchat", "conversation", "chat", "identity", "chitchat-p1", "chitchat-p2", "chitchat-p3", "greetings"):
                from Tantra.dataset import build_phased_chitchat_curriculum
                build_phased_chitchat_curriculum(base_dir)
            
            for c in cand_names:
                c_path = os.path.join(base_dir, c)
                if os.path.isfile(c_path) and os.path.getsize(c_path) > 0:
                    selected_file = c_path
                    break

        if selected_file and os.path.isfile(selected_file):
            log.info(f"📂 Routing to Expert Track File: {selected_file} ({os.path.getsize(selected_file)/1e6:.1f} MB)")
            dataset = JSONLDataset(selected_file, tokenizer, seq_len=seq_len,
                                  max_samples=max_samples, mask_non_assistant=mask_non_assistant, pack_sequences=pack_sequences)
        else:
            log.warning(f"Could not find track file for '{track}' under {base_dir}; falling back to multi-track.")

    if dataset is not None:
        pass
    elif os.path.isdir(dataset_path):
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
            # Multiple jsonl files directly under dataset_path
            direct_jsonls = [p for p in glob.glob(os.path.join(dataset_path, "*.jsonl")) if "preference" not in p and "sample" not in p]
            if len(direct_jsonls) > 1:
                topic_paths = {os.path.splitext(os.path.basename(p))[0].replace("expert_", ""): [p] for p in direct_jsonls}
                log.info(f"  Multi-track datasets detected: {list(topic_paths.keys())}")
                weights = {t: 1.0 for t in topic_paths.keys()}
                dataset = TopicMixedDataset(topic_paths, weights, tokenizer, seq_len=seq_len,
                                            max_samples=max_samples, mask_non_assistant=mask_non_assistant)
            elif len(direct_jsonls) == 1:
                dataset = JSONLDataset(direct_jsonls[0], tokenizer, seq_len=seq_len,
                                      max_samples=max_samples, mask_non_assistant=mask_non_assistant, pack_sequences=pack_sequences)
            else:
                dataset = JSONLDataset(dataset_path, tokenizer, seq_len=seq_len,
                                      max_samples=max_samples, mask_non_assistant=mask_non_assistant, pack_sequences=pack_sequences)
    else:
        bin_cache = find_bin_cache(dataset_path)
        if bin_cache:
            log.info(f"  Pre-tokenized cache found -> {bin_cache} (skipping BPE encode() at train time)")
            dataset = PretokenizedBinDataset(bin_cache, seq_len=seq_len,
                                             max_samples=max_samples,
                                             mask_non_assistant=mask_non_assistant)
        else:
            dataset = JSONLDataset(dataset_path, tokenizer, seq_len=seq_len,
                                  max_samples=max_samples, mask_non_assistant=mask_non_assistant, pack_sequences=pack_sequences)

    val_loader = None
    target_val_file = dataset_path if os.path.isfile(dataset_path) else None
    if target_val_file is None and os.path.isdir(dataset_path):
        candidates = [
            os.path.join(dataset_path, "master_corpus.jsonl"),
            os.path.join(dataset_path, "expert_general.jsonl"),
            os.path.join(dataset_path, "expert_conversation.jsonl"),
        ] + glob.glob(os.path.join(dataset_path, "**/*.jsonl"), recursive=True)
        for c in candidates:
            if os.path.isfile(c) and os.path.getsize(c) > 100:
                target_val_file = c
                break

    if target_val_file and os.path.isfile(target_val_file):
        try:
            val_dataset = JSONLDataset(target_val_file, tokenizer, seq_len=seq_len, max_samples=100,
                                       mask_non_assistant=mask_non_assistant, split="val", val_ratio=0.05,
                                       pack_sequences=pack_sequences)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
            log.info(f"  Held-out validation stream active from: {os.path.basename(target_val_file)}")
        except Exception as e:
            log.debug(f"Could not build validation dataset: {e}")



    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=data_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=data_workers > 0, prefetch_factor=4 if data_workers > 0 else None,
    )
    enrichment = 0.0 if training_stage == "sft" else 0.02
    try:
        trainer.train_dataset(dataloader, max_steps=steps, log_every=log_every, eval_every=eval_every, eval_callback=eval_callback, checkpoint_every=checkpoint_every, checkpoint_callback=checkpoint_callback, tokenizer=tokenizer, enrichment_rate=enrichment, use_latent_reasoning=use_latent_reasoning, auto_growth=auto_growth, growth_patience=growth_patience, growth_min_delta=growth_min_delta, max_layers=max_layers, val_loader=val_loader)

    except KeyboardInterrupt:
        # Ctrl+C happens after an optimizer boundary in many practical runs.
        # Save that completed state before allowing the process to stop.
        trainer.save_checkpoint(latest_ckpt, save_optimizer=True, async_write=False)
        log.warning("Training interrupted; recovery checkpoint saved at step %d.", trainer.step_count)
    finally:
        # Guarantee that the exact final milestone checkpoint is written synchronously
        final_step = trainer.step_count
        final_milestone = os.path.join(checkpoints_dir, f"checkpoint_step_{final_step}.pt")
        trainer.save_checkpoint(final_milestone, save_optimizer=True, async_write=False)
        trainer.save_checkpoint(latest_ckpt, save_optimizer=True, async_write=False)
        trainer.flush_checkpoint_writers()
        log.info("🏁 [FINAL CHECKPOINT FLUSHED] Step %d successfully written to disk: %s", final_step, final_milestone)


def run_dpo_training(
    model, tokenizer, dataset_path, steps=1000, eval_every=250, log_every=25,
    checkpoint_every=250, batch_size=4, grad_accumulation_steps=4, data_workers=2,
    lr=5e-6, beta=0.1, model_dir=None, checkpoint_path=None
):
    from Tantra.dataset import DPODataset
    from torch.utils.data import DataLoader
    
    checkpoints_dir = os.path.join(model_dir or MODEL_DIR, "Checkpoints")
    latest_dir = os.path.join(model_dir or MODEL_DIR, "Latest")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)
    latest_ckpt = os.path.join(latest_dir, "checkpoint_latest.pt")
    
    device = next(model.parameters()).device
    trainer = NeuroTrainer(
        model, lr=lr, weight_decay=0.01,
        total_steps=steps, warmup_steps=max(10, steps // 20),
        grad_accumulation_steps=grad_accumulation_steps
    )
    trainer.device = device
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
        log.info(f"Loaded baseline checkpoint for DPO: {checkpoint_path}")
        
    dpo_dataset = DPODataset(dataset_path, tokenizer, max_len=128)
    dpo_loader = DataLoader(
        dpo_dataset,
        batch_size=batch_size,
        num_workers=data_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=data_workers > 0,
        prefetch_factor=2 if data_workers > 0 else None,
    )
    
    def ckpt_cb(step, loss):
        milestone = os.path.join(checkpoints_dir, f"checkpoint_dpo_step_{step}.pt")
        trainer.save_checkpoint(milestone, save_optimizer=True, async_write=False)
        trainer.save_checkpoint(latest_ckpt, save_optimizer=True, async_write=False)
        
    def eval_cb(step):
        log.info(f"┌── 🌐 [ DPO PREFERENCE ALIGNMENT BENCHMARK @ Step {step:,} ] " + "─" * 20)
        test_prompts = [
            ("General", "💬", "<|user|>\nWhat is Tantra LLM?\n\n<|assistant|>\n"),
            ("Coding",  "💻", "<|user|>\nWrite a Python function to reverse a string.\n\n<|assistant|>\n"),
            ("Math",    "🔢", "<|user|>\nSolve for x in 2x + 6 = 14.\n\n<|assistant|>\n"),
            ("Science", "🔬", "<|user|>\nState Newton's First Law of Motion.\n\n<|assistant|>\n")
        ]
        raw_model = getattr(model, "module", getattr(model, "_orig_mod", model))
        for domain, icon, prompt_text in test_prompts:
            prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], device=raw_model.embed.weight.device)
            out = raw_model.generate(prompt_ids, max_new_tokens=48, min_new_tokens=1, temperature=0.7, top_p=0.9, repetition_penalty=1.2)
            new_tokens = out[0, prompt_ids.shape[1]:].tolist()
            response = tokenizer.decode(new_tokens).strip().replace("\n", " ")
            log.info(f"│ {icon} [{domain:7s}]: {response[:90]}")
        
        try:
            from Tantra.world_eval import evaluate_zero_shot_world_knowledge
            world_res = evaluate_zero_shot_world_knowledge(raw_model, tokenizer)
            if world_res:
                log.info(f"│ 🌍 [World MMLU]: 🏆 {world_res['world_mmlu_accuracy']:.1f}% Zero-Shot Accuracy ({world_res['correct_samples']}/{world_res['total_samples']} correct)")
        except Exception:
            pass
        log.info("└" + "─" * 80)
        
    try:
        trainer.train_dpo(
            dpo_loader,
            beta=beta,
            max_steps=steps,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            checkpoint_callback=ckpt_cb,
            eval_every=eval_every,
            eval_callback=eval_cb,
        )
    finally:
        final_step = trainer.step_count
        final_milestone = os.path.join(checkpoints_dir, f"checkpoint_dpo_step_{final_step}.pt")
        trainer.save_checkpoint(final_milestone, save_optimizer=True, async_write=False)
        trainer.save_checkpoint(latest_ckpt, save_optimizer=True, async_write=False)
        trainer.flush_checkpoint_writers()
        log.info("🏁 [DPO ALIGNMENT FINISHED] Checkpoint saved: %s", final_milestone)


def run_evaluation(model, tokenizer, dataset_path, device="cpu", max_batches=50):
    log.info("== [MODEL EVALUATION & BENCHMARK MODE] =============")
    engine = EvaluationEngine(model, device=str(device))
    if os.path.isdir(dataset_path):
        from Tantra.dataset import TopicMixedDataset
        dataset = TopicMixedDataset(dataset_path, tokenizer, seq_len=128, max_samples=500)
    else:
        dataset = JSONLDataset(dataset_path, tokenizer, seq_len=128, max_samples=500) if os.path.exists(dataset_path) else None

    if dataset is None:
        log.warning(f"Could not load evaluation dataset from: {dataset_path}")
        return {}

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    metrics = engine.evaluate_metrics(loader, max_batches=max_batches)
    print("\n" + "=" * 65)
    print(f"📊 EVALUATION METRICS REPORT ({dataset_path})")
    print("=" * 65)
    for k, v in metrics.items():
        print(f"  • {k:20s}: {v:.4f}")
    print("=" * 65 + "\n")
    return metrics


def run_compression_benchmark(comp_cfg):
    log.info("== [COMPRESSION BENCHMARK] =========================")
    bench = CompressionBenchmark(comp_cfg)
    sample_weight = torch.randn(1024, 1024, dtype=torch.float32)
    bench.run(sample_weight, output_dir=os.path.join(MODEL_DIR, "reports"))


def run_generation(model, tokenizer, vcfg, device, prompt_text=None, temperature=0.35, top_p=0.9, max_new_tokens=64, use_mtp=True):
    log.info("── [TEXT GENERATION MODE (MTP Speculation)] ───────")
    if prompt_text:
        log.info(f"  Prompt: {prompt_text!r}")
        prompt_ids = tokenizer.encode(prompt_text)
        if not prompt_ids:
            prompt_ids = [1]  # <bos>
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    else:
        prompt_tensor = torch.randint(0, vcfg.vocab_size, (1, 4), device=device)
        log.info(f"  Random Prompt tokens: {prompt_tensor.tolist()[0]}")

    model.eval()
    with torch.no_grad():
        out = model.generate(prompt_tensor, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, use_mtp_speculation=use_mtp)

    out_ids = out.tolist()[0]
    gen_ids = out_ids[prompt_tensor.size(1):] if prompt_text else out_ids
    decoded = tokenizer.decode(gen_ids)
    print("\n" + "=" * 60)
    print(f"🤖 [TANTRA RESPONSE]:\n{decoded}")
    print("=" * 60 + "\n")
    log.info(f"  Generated Tokens Count: {len(gen_ids)} ✓")
    return decoded


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
                        choices=["full", "probe", "vocab", "train", "dataset", "eval", "compress", "generate", "serve", "status", "experts", "chat", "adapter", "dpo", "auto-pilot", "benchmark", "export"],
                        help="Execution mode")
    parser.add_argument("--track", "--expert", "--domain", dest="track", type=str, default=None,
                        choices=["all", "chitchat", "conversation", "chat", "identity", "chitchat-p1", "chitchat-p2", "chitchat-p3", "greetings", "math", "math-p1", "math-p2", "math-p3", "code", "code-p1", "code-p2", "code-p3", "science", "science-p1", "science-p2", "science-p3", "general", "gold"],
                        help="Select specific expert track for focused training (e.g. --track chitchat-p1, --track code-p1, --track math-p1, --track science-p1)")
    parser.add_argument("--curriculum-phase", "--phase", dest="curriculum_phase", type=int, default=None, choices=[1, 2, 3],
                        help="Curriculum learning phase: 1=greetings & identity only, 2=short dialogues (<100 words), 3=full multi-turn dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to custom .pt model checkpoint to load (for chat, eval, serve, dpo, benchmark, export)")
    parser.add_argument("--pack-sequences", action=argparse.BooleanOptionalAction, default=True, help="Enable continuous document sequence packing (0% padding waste)")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="JSONL dataset path")
    parser.add_argument("--preference-dataset", type=str, default="Datasets/preference_pairs.jsonl", help="DPO pairwise preference dataset path")
    parser.add_argument("--dpo-beta", type=float, default=0.1, help="DPO temperature scaling hyperparameter beta (default: 0.1)")
    parser.add_argument("--steps", type=int, default=30, help="Training steps")
    parser.add_argument("--seq-len", type=int, default=128, help="Context sequence length window")
    parser.add_argument("--use-mtp", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable Multi-Token Prediction (MTP)")
    parser.add_argument("--temperature", type=float, default=0.35, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--port", type=int, default=8000, help="Server port (serve mode)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, cpu, cuda, mps")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    parser.add_argument("--fresh", action="store_true", help="Start fresh on official 38.6M architecture without reading previous checkpoints")
    parser.add_argument("--eval-every", type=int, default=500, help="Run a qualitative generation sample and archive checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=50, help="Print a rolling training summary every N optimizer steps")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Save a resumable recovery checkpoint every N optimizer steps (0 disables; default 500 minimizes I/O overhead)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (larger effective batch without more RAM; 1 = off)")
    parser.add_argument("--data-workers", type=int, default=0, help="Parallel data-loading/tokenization workers (overlaps tokenization with training compute; 0 = synchronous/main-thread, as before)")

    parser.add_argument("--training-stage", choices=["pretrain", "sft"], default="sft", help="pretrain uses full-token loss; sft supervises assistant replies only")
    parser.add_argument("--latent-reasoning", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable latent reasoning. Defaults off for pretraining and on for SFT.")
    parser.add_argument("--mtp-loss", action=argparse.BooleanOptionalAction, default=None, help="Train the MTP auxiliary head. Defaults off for pretraining and on for SFT.")
    parser.add_argument("--auto-growth", action=argparse.BooleanOptionalAction, default=True, help="Automatically add depth layers whenever loss plateaus (default: enabled)")
    parser.add_argument("--growth-patience", type=int, default=250, help="Optimizer steps to observe before auto-growth adds a layer (default: 250)")
    parser.add_argument("--growth-min-delta", type=float, default=0.003, help="Minimum EMA-loss improvement required to avoid auto-growth")
    parser.add_argument("--max-layers", type=int, default=16, help="Hard maximum depth when auto-growth is enabled (default: 16 layers)")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile(backend='inductor') for CPU/GPU kernel fusion")
    parser.add_argument("--optimizer", type=str, choices=["adamw", "adam", "lion", "sgd"], default="adamw", help="Optimizer choice (default: adamw)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-4 for AdamW, 5e-5 for Lion)")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay (default: 0.01 for AdamW, 0.05 for Lion)")
    parser.add_argument("--warmup", "--warmup-steps", dest="warmup", type=int, default=None, help="LR warmup steps (default: steps // 10)")
    parser.add_argument("--topic-weights", type=str, default=None, help="JSON dict of topic weights, e.g. '{\"general\":40,\"code\":15}'")
    parser.add_argument("--model-dir", "--checkpoint-dir", dest="model_dir", type=str, default=None, help="Custom root directory for model checkpoints (e.g. Kaggle/Google Drive)")
    parser.add_argument("--mask-non-assistant", action="store_true", default=None, help="Supervise assistant replies only during training")
    parser.add_argument("--adapter-action", default="list", choices=["list", "add", "remove", "init"],
                        help="--mode adapter sub-action")
    parser.add_argument(\"--adapter\", type=str, default=None,
                        help=\"Category to train (dataset mode) or force for chat/generate. None routes per-request.\")
    parser.add_argument(\"--adapter-desc\", type=str, default=\"\", help=\"Description when adding a category\")
    parser.add_argument(\"--adapter-topics\", type=str, default=None, help=\"Comma list of Datasets/<topic> folders for a new category\")
    parser.add_argument(\"--dim\", type=int, default=512, help=\"Embedding dimension (default: 512)\")
    parser.add_argument(\"--layers\", type=int, default=8, help=\"Number of NeuroCore layers (default: 8)\")
    parser.add_argument(\"--heads\", type=int, default=8, help=\"Number of attention heads (default: 8)\")
    # ── MoE Expert Configuration ──────────────────────────────────────────────
    parser.add_argument(\"--num-experts\", type=int, default=0,
                        help=\"Number of real Top-1 MoE experts per MoE layer (0 = dense, no MoE). "
                             "Recommended: 4 for Dual-T4 Kaggle runs. Odd layers get MoE blocks.\")
    parser.add_argument(\"--real-moe\", action=\"store_true\", default=False,
                        help=\"Enable real Top-1 MoE routing (requires --num-experts >= 2). "
                             "When off, the model is a dense transformer regardless of --num-experts.\")
    # ── Automatic Curriculum Sequencer ───────────────────────────────────────
    parser.add_argument(\"--curriculum-order\", action=\"store_true\", default=False,
                        help=\"Run the full phased curriculum in order automatically: "
                             "chitchat-p1 -> p2 -> p3 -> math-p1 -> p2 -> p3 -> code-p1 -> p2 -> p3 -> science-p1 -> p2 -> p3. "
                             "Each phase gets steps/12 of the total --steps budget. "
                             "Conversation phases get 3x weight to prioritize greetings & grammar first.\")
    parser.add_argument(\"--output\", type=str, default=None, help=\"Output path for model export mode\")
    parser.add_argument(\"--prompt\", type=str, default=None, help=\"Text prompt for --mode generate\")
    parser.add_argument(\"--max-new-tokens\", type=int, default=64, help=\"Max new tokens to generate\")
    parser.add_argument(\"--max-grad-norm\", type=float, default=1.0, help=\"Max gradient norm clipping threshold (default: 1.0)\")
    parser.add_argument(\"--mtp-weight\", type=float, default=0.3, help=\"Auxiliary MTP loss weight factor (default: 0.3)\")
    parser.add_argument(\"--seed\", type=int, default=42, help=\"Random seed for reproducibility\")
    args = parser.parse_args()

    from Tantra.utils import set_seed
    set_seed(args.seed)

    vcfg = VocabConfig()
    mcfg = NeuroCoreConfig()
    mcfg.block.alra.dim = args.dim
    mcfg.block.sgp.dim = args.dim
    mcfg.block.num_layers = args.layers
    mcfg.block.alra.num_heads = args.heads
    mcfg.block.alra.head_dim = max(1, args.dim // args.heads)
    mcfg.use_mtp = args.use_mtp

    # ── Wire MoE configuration from CLI flags ─────────────────────────────────
    # Default config has num_experts=10 but real_top1=False (dense transformer).
    # --num-experts N --real-moe enables real Top-1 conditional compute.
    # Without --real-moe the model stays dense regardless of --num-experts.
    _num_experts = getattr(args, "num_experts", 0) or 0
    _real_moe    = getattr(args, "real_moe", False)
    if _real_moe and _num_experts >= 2:
        mcfg.moe.num_experts = _num_experts
        mcfg.moe.real_top1   = True
        log.info(f"🧠 [Real MoE] Enabled: {_num_experts} Top-1 experts per MoE layer (odd layers only)")
    else:
        # Dense mode: num_experts=1 disables all MoE paths in NeuroCoreBlock
        mcfg.moe.num_experts = 1
        mcfg.moe.real_top1   = False
        if _real_moe and _num_experts < 2:
            log.warning("--real-moe requires --num-experts >= 2. Running dense (no MoE).")

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
    if getattr(args, "fresh", False):
        mcfg = cpu_dense_config(vocab_size=vcfg.vocab_size, attention_kind="causal")
        model = build_cpu_model("dense", attention_kind="causal", vocab_size=vcfg.vocab_size)
        log.info(f"Initialized fresh official 38.6M CPU profile model ({model.num_parameters:,} parameters).")
    else:
        ckpt_candidates = []
        if args.checkpoint and os.path.exists(args.checkpoint):
            ckpt_candidates.append(args.checkpoint)
        ckpt_candidates.extend([
            os.path.join(args.model_dir or MODEL_DIR, "Latest", "checkpoint_latest.pt"),
            os.path.join(args.model_dir or MODEL_DIR, "checkpoint_latest.pt"),
            os.path.join(MODEL_DIR, "Latest", "checkpoint_latest.pt"),
            os.path.join(MODEL_DIR, "checkpoint_latest.pt"),
        ])
        latest_ckpt_file = next((p for p in ckpt_candidates if os.path.exists(p) and os.path.getsize(p) > 10 * 1024 * 1024), ckpt_candidates[0])
        restore_checkpoint_architecture(mcfg, latest_ckpt_file)
        _ckpt_path = latest_ckpt_file
        if os.path.exists(_ckpt_path) and os.path.getsize(_ckpt_path) > 10 * 1024 * 1024 and mcfg is not None:
            try:
                log.info(f"Reading model config from checkpoint: {_ckpt_path} ({os.path.getsize(_ckpt_path)/1e6:.1f} MB)...")
                _ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(_ckpt, dict):
                    _ckpt_cfg = _ckpt.get("config", None)
                    if _ckpt_cfg is not None:
                        _ckpt_cfg.vocab.vocab_size = vcfg.vocab_size
                        mcfg = _ckpt_cfg
                    
                    # Also check state_dict layer keys for dynamically grown models
                    sdict = _ckpt.get("model_state_dict", {})
                    import re
                    layer_indices = [int(m.group(1)) for k in sdict.keys() for m in [re.search(r'layers\.(\d+)\.', k)] if m]
                    if layer_indices and mcfg is not None and hasattr(mcfg, "block"):
                        ckpt_num_layers = max(layer_indices) + 1
                        if ckpt_num_layers != mcfg.block.num_layers:
                            mcfg.block.num_layers = ckpt_num_layers
                            log.info(f"Detected {ckpt_num_layers} layers in checkpoint weights; initialized architecture accordingly.")

                    log.info("Rebuilt model architecture from checkpoint "
                             f"(dim={mcfg.block.alra.dim}, layers={mcfg.block.num_layers}, vocab={mcfg.vocab.vocab_size}).")
            except Exception as _exc:
                log.warning(f"Could not read checkpoint config: {_exc}; using default architecture.")
        model = init_model(mcfg, rt.device)
        dev_str = str(getattr(rt.device, "type", rt.device))
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and dev_str.startswith("cuda") and args.mode in ("train", "dataset", "auto-pilot", "dpo"):
            log.info(f"  [Multi-GPU DataParallel] Enabling {torch.cuda.device_count()}x GPUs for parallel batch execution.")
            model = torch.nn.DataParallel(model)

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
        # Load custom checkpoint if passed or automatically load highest available milestone
        ckpt_to_load = args.checkpoint
        if ckpt_to_load is None:
            cand_list = []
            for d in [os.path.join(MODEL_DIR, "Checkpoints"), os.path.join(MODEL_DIR, "Best"), os.path.join(MODEL_DIR, "Latest"), os.path.join(MODEL_DIR, "Archive")]:
                if os.path.exists(d):
                    cand_list.extend(glob.glob(os.path.join(d, "*.pt")))

            def _step_val(p):
                import re
                m = re.search(r'step_(\d+)', os.path.basename(p))
                return int(m.group(1)) if m else 0

            sorted_cands = sorted([p for p in cand_list if "sample" not in p], key=_step_val, reverse=True)
            if sorted_cands:
                ckpt_to_load = sorted_cands[0]

        if args.checkpoint is not None:
            if not os.path.exists(args.checkpoint):
                log.error(f"❌ Checkpoint file not found: '{args.checkpoint}'")
                # Try finding matching checkpoints
                cand_find = [p for p in glob.glob("**/*.pt", recursive=True) if "sample" not in p]
                if cand_find:
                    log.info(f"Available checkpoints found on disk: {cand_find[:5]}")
            else:
                ckpt_to_load = args.checkpoint

        if ckpt_to_load and os.path.exists(ckpt_to_load):
            try:
                trainer.load_checkpoint(ckpt_to_load)
                log.info(f"✅ Loaded checkpoint for chat: {ckpt_to_load} (Step {trainer.step_count:,})")
            except Exception as e:
                log.error(f"Failed to load checkpoint {ckpt_to_load}: {e}")
        else:
            log.warning("⚠️ No valid checkpoint loaded! Model is running on random untrained weights.")

        if args.adapter is not None:
            if args.adapter not in model.category_layers:
                log.warning(f"Category '{args.adapter}' not in adapter checkpoint; ignoring --adapter.")
            else:
                model.active_category = args.adapter
        else:
            router = RequestRouter(AdapterRegistry())
            router._model = model  # allow per-request routing to set active category
            run_interactive_chat(model, tok, rt.device, args.temperature, args.top_p, router=router, use_mtp=args.use_mtp)
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
            use_mtp_loss = True  # Enable MTP multi-token speculative loss for both pretrain and SFT

        # Optimizer-specific hyperparameter defaults
        resolved_optimizer = (args.optimizer or "adamw").lower().strip()
        if args.lr is not None:
            resolved_lr = args.lr
        else:
            resolved_lr = 5e-5 if resolved_optimizer == "lion" else 1e-4

        if args.weight_decay is not None:
            resolved_wd = args.weight_decay
        else:
            resolved_wd = 0.05 if resolved_optimizer == "lion" else 0.01

        if getattr(args, "curriculum_order", False):
            # ── AUTOMATED SEQUENTIAL CURRICULUM ORDER ──
            # Prioritizes Conversation / Greetings / Grammar FIRST (55% total budget),
            # followed by Code (24%), Math (21%), and Science (15% normalized).
            curriculum_stages = [
                ("chitchat-p1", 0.20, "💬 CONVERSATION PHASE 1: Pure Greetings, Pleasantries & Identity Reflexes"),
                ("chitchat-p2", 0.20, "💬 CONVERSATION PHASE 2: Short Natural Turn-Taking & Conversational Grammar"),
                ("chitchat-p3", 0.15, "💬 CONVERSATION PHASE 3: Deep Multi-Turn Conversations & Instruction Fluency"),
                ("code-p1",     0.08, "💻 CODE PHASE 1: Python Syntax & Standard Library Primitives"),
                ("code-p2",     0.08, "💻 CODE PHASE 2: Algorithmic Logic & Functional Implementation"),
                ("code-p3",     0.08, "💻 CODE PHASE 3: Full Software Systems & Debugging"),
                ("math-p1",     0.07, "🔢 MATH PHASE 1: Arithmetic & Linear Equations"),
                ("math-p2",     0.07, "🔢 MATH PHASE 2: GSM8K Multi-Step Reasoning & Word Problems"),
                ("math-p3",     0.07, "🔢 MATH PHASE 3: MetaMathQA Advanced Symbolic Math & Proofs"),
                ("science-p1",  0.05, "🔬 SCIENCE PHASE 1: Fundamental Physical Laws & Core Definitions"),
                ("science-p2",  0.05, "🔬 SCIENCE PHASE 2: Explanatory Natural Sciences & Biology/Physics"),
                ("science-p3",  0.05, "🔬 SCIENCE PHASE 3: Advanced Multidisciplinary Science & Logic"),
            ]
            total_target_steps = args.steps
            log.info("=" * 80)
            log.info(f"🚀 [AUTO CURRICULUM SEQUENCER] Running 12-Stage Phased Curriculum ({total_target_steps:,} total steps)")
            log.info("🎯 PRIORITY: Conversation / Greetings / Grammar (55% budget) FIRST -> Code -> Math -> Science")
            log.info("=" * 80)
            
            for stage_idx, (track_name, budget_ratio, stage_desc) in enumerate(curriculum_stages, 1):
                stage_steps = max(50, int(total_target_steps * budget_ratio))
                log.info(f"\n▶️ [{stage_idx}/12] Launching: {stage_desc}")
                log.info(f"   Track: '{track_name}' | Steps: +{stage_steps:,} ({budget_ratio*100:.0f}% of total budget)")
                
                run_dataset_training(
                    model, tok, args.dataset, steps=stage_steps, resume=True,
                    eval_every=args.eval_every, log_every=args.log_every,
                    checkpoint_every=args.checkpoint_every, batch_size=args.batch_size,
                    seq_len=args.seq_len, grad_accumulation_steps=args.grad_accum,
                    data_workers=args.data_workers, use_latent_reasoning=use_latent_reasoning,
                    use_mtp_loss=use_mtp_loss, compile=args.compile, lr=resolved_lr,
                    weight_decay=resolved_wd, optimizer=resolved_optimizer, warmup_steps=args.warmup,
                    topic_weights=topic_weights, training_stage=args.training_stage,
                    auto_growth=args.auto_growth, growth_patience=args.growth_patience,
                    growth_min_delta=args.growth_min_delta, max_layers=args.max_layers,
                    adapter_name=args.adapter, model_dir=(ADAPTER_ROOT if args.adapter is not None else args.model_dir),
                    pack_sequences=args.pack_sequences, checkpoint_path=args.checkpoint,
                    max_grad_norm=args.max_grad_norm, mtp_loss_weight=args.mtp_weight,
                    track=track_name, curriculum_phase=None
                )
        else:
            run_dataset_training(model, tok, args.dataset, steps=args.steps, resume=args.resume, eval_every=args.eval_every, log_every=args.log_every, checkpoint_every=args.checkpoint_every, batch_size=args.batch_size, seq_len=args.seq_len, grad_accumulation_steps=args.grad_accum, data_workers=args.data_workers, use_latent_reasoning=use_latent_reasoning, use_mtp_loss=use_mtp_loss, compile=args.compile, lr=resolved_lr, weight_decay=resolved_wd, optimizer=resolved_optimizer, warmup_steps=args.warmup, topic_weights=topic_weights, training_stage=args.training_stage, auto_growth=args.auto_growth, growth_patience=args.growth_patience, growth_min_delta=args.growth_min_delta, max_layers=args.max_layers, adapter_name=args.adapter, model_dir=(ADAPTER_ROOT if args.adapter is not None else args.model_dir), pack_sequences=args.pack_sequences, checkpoint_path=args.checkpoint, max_grad_norm=args.max_grad_norm, mtp_loss_weight=args.mtp_weight, track=args.track, curriculum_phase=args.curriculum_phase)

    elif args.mode == "dpo":
        dpo_ckpt = args.checkpoint
        if dpo_ckpt is None:
            cand_list = []
            for d in [os.path.join(args.model_dir or MODEL_DIR, "Checkpoints"), os.path.join(args.model_dir or MODEL_DIR, "Best"), os.path.join(args.model_dir or MODEL_DIR, "Latest")]:
                if os.path.exists(d):
                    cand_list.extend(glob.glob(os.path.join(d, "*.pt")))
            if cand_list:
                import re
                dpo_ckpt = max([p for p in cand_list if "sample" not in p], key=lambda p: int(re.search(r'step_(\d+)', p).group(1)) if re.search(r'step_(\d+)', p) else 0)

        run_dpo_training(
            model, tok, args.preference_dataset, steps=args.steps,
            eval_every=args.eval_every, log_every=args.log_every,
            checkpoint_every=args.checkpoint_every, batch_size=args.batch_size,
            grad_accumulation_steps=args.grad_accum, data_workers=args.data_workers,
            lr=args.lr or 5e-6, beta=args.dpo_beta, model_dir=args.model_dir,
            checkpoint_path=dpo_ckpt
        )

    elif args.mode == "auto-pilot":
        total_steps = args.steps
        sft_steps = int(total_steps * 0.90)
        dpo_steps = max(1, total_steps - sft_steps)
        
        log.info("=" * 80)
        log.info(f"🚀 [AUTO-PILOT PIPELINE] Total: {total_steps:,} Steps │ Phase 1 (SFT + Auto-Growth): {sft_steps:,} Steps │ Phase 2 (DPO Preference Alignment): {dpo_steps:,} Steps")
        log.info("=" * 80)
        
        # Ensure datasets are ready
        from Tantra.dataset import build_4track_curriculum, generate_gold_datasets
        if not os.path.exists(args.dataset):
            log.info(f"Dataset {args.dataset} not found locally. Auto-building 4-Track Domain Curriculum...")
            build_4track_curriculum(datasets_dir=os.path.dirname(args.dataset) or "Datasets")
        if args.preference_dataset and not os.path.exists(args.preference_dataset):
            log.info(f"Preference dataset {args.preference_dataset} not found. Auto-generating DPO pairs...")
            generate_gold_datasets(datasets_dir=os.path.dirname(args.preference_dataset) or "Datasets")

        # Phase 1: High-Density SFT with Dynamic Auto-Growth
        log.info("▶️ [AUTO-PILOT PHASE 1/2] Starting High-Density SFT & Auto-Growth...")
        resolved_optimizer = (args.optimizer or "adamw").lower().strip()
        resolved_lr = args.lr if args.lr is not None else (5e-5 if resolved_optimizer == "lion" else 1e-4)
        resolved_wd = args.weight_decay if args.weight_decay is not None else (0.05 if resolved_optimizer == "lion" else 0.01)
        
        run_dataset_training(
            model, tok, args.dataset, steps=sft_steps, resume=args.resume,
            eval_every=args.eval_every, log_every=args.log_every,
            checkpoint_every=args.checkpoint_every, batch_size=args.batch_size,
            seq_len=args.seq_len, grad_accumulation_steps=args.grad_accum,
            data_workers=args.data_workers,
            use_latent_reasoning=(args.latent_reasoning if args.latent_reasoning is not None else True),
            use_mtp_loss=(args.mtp_loss if args.mtp_loss is not None else True),
            compile=args.compile, lr=resolved_lr, weight_decay=resolved_wd,
            optimizer=resolved_optimizer, warmup_steps=args.warmup,
            training_stage="sft", auto_growth=args.auto_growth,
            growth_patience=args.growth_patience, growth_min_delta=args.growth_min_delta,
            max_layers=args.max_layers, model_dir=args.model_dir,
            pack_sequences=args.pack_sequences, checkpoint_path=args.checkpoint,
            max_grad_norm=args.max_grad_norm, mtp_loss_weight=args.mtp_weight,
            track=args.track
        )
        
        # Phase 2: DPO Alignment
        log.info("▶️ [AUTO-PILOT PHASE 2/2] Phase 1 complete! Autonomously starting Phase 2 (DPO Preference Alignment)...")
        latest_ckpt = os.path.join(args.model_dir or MODEL_DIR, "Latest", "checkpoint_latest.pt")
        run_dpo_training(
            model, tok, args.preference_dataset, steps=dpo_steps,
            eval_every=args.eval_every, log_every=args.log_every,
            checkpoint_every=args.checkpoint_every, batch_size=args.batch_size,
            grad_accumulation_steps=args.grad_accum, data_workers=args.data_workers,
            lr=5e-6, beta=args.dpo_beta, model_dir=args.model_dir,
            checkpoint_path=latest_ckpt if os.path.exists(latest_ckpt) else None
        )
        log.info("🏆 [AUTO-PILOT PIPELINE COMPLETE] Multi-Stage Autonomous Training & Alignment Finished!")

    elif args.mode == "benchmark":
        from Tantra.benchmark import run_benchmarks
        run_benchmarks(args.checkpoint, str(rt.device))
    elif args.mode == "export":
        from Tantra.export import export_clean_checkpoint
        export_clean_checkpoint(args.checkpoint, args.output or args.model_dir or "Model/Export/checkpoint_clean.pt")
    elif args.mode == "eval":
        ckpt_to_load = args.checkpoint
        if ckpt_to_load is None:
            cand_list = []
            for d in [os.path.join(MODEL_DIR, "Checkpoints"), os.path.join(MODEL_DIR, "Best"), os.path.join(MODEL_DIR, "Latest")]:
                if os.path.exists(d):
                    cand_list.extend(glob.glob(os.path.join(d, "*.pt")))
            if cand_list:
                import re
                ckpt_to_load = max([p for p in cand_list if "sample" not in p], key=lambda p: int(re.search(r'step_(\d+)', p).group(1)) if re.search(r'step_(\d+)', p) else 0)

        if ckpt_to_load and os.path.exists(ckpt_to_load):
            try:
                trainer.load_checkpoint(ckpt_to_load)
                log.info(f"Loaded checkpoint for evaluation: {ckpt_to_load}")
            except Exception as e:
                log.warning(f"Could not load checkpoint {ckpt_to_load}: {e}")
        run_evaluation(model, tok, args.dataset, device=rt.device)
    elif args.mode == "generate":
        ckpt_to_load = args.checkpoint
        if ckpt_to_load is None:
            cand_list = []
            for d in [os.path.join(MODEL_DIR, "Checkpoints"), os.path.join(MODEL_DIR, "Best"), os.path.join(MODEL_DIR, "Latest")]:
                if os.path.exists(d):
                    cand_list.extend(glob.glob(os.path.join(d, "*.pt")))
            if cand_list:
                import re
                ckpt_to_load = max([p for p in cand_list if "sample" not in p], key=lambda p: int(re.search(r'step_(\d+)', p).group(1)) if re.search(r'step_(\d+)', p) else 0)

        if ckpt_to_load and os.path.exists(ckpt_to_load):
            try:
                trainer.load_checkpoint(ckpt_to_load)
                log.info(f"Loaded checkpoint for generation: {ckpt_to_load}")
            except Exception as e:
                log.warning(f"Could not load checkpoint {ckpt_to_load}: {e}")

        run_generation(model, tok, vcfg, rt.device, prompt_text=args.prompt, temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens, use_mtp=args.use_mtp)
    elif args.mode == "serve":
        ckpt_to_load = args.checkpoint
        if ckpt_to_load is None:
            for cand in [
                os.path.join(MODEL_DIR, "Checkpoints", "checkpoint_step_30000.pt"),
                os.path.join(MODEL_DIR, "Best", "checkpoint_best.pt"),
                os.path.join(MODEL_DIR, "Latest", "checkpoint_latest.pt"),
            ]:
                if os.path.exists(cand):
                    ckpt_to_load = cand
                    break
        if ckpt_to_load and os.path.exists(ckpt_to_load):
            try:
                trainer.load_checkpoint(ckpt_to_load)
                log.info(f"Loaded checkpoint for serve: {ckpt_to_load}")
            except Exception as e:
                log.warning(f"Could not load checkpoint {ckpt_to_load}: {e}")
        serve(model, tok, port=args.port, expert_dir=EXPERTS_DIR)
    else:  # full mode
        run_forward(model, vcfg, rt.batch_size, rt.device)
        run_evaluation(model, tok, args.dataset)
        run_generation(model, vcfg, rt.device)

    log.info("Pipeline complete -- NeuroCore ready!")
    sched.stop()


if __name__ == "__main__":
    main()
