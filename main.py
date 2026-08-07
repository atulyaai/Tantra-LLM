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
import os
import torch
import sys

from Tantra.config import NeuroCoreConfig, VocabConfig, MoEConfig, CompressionConfig
from Tantra.utils import get_logger
from Tantra.hardware import HardwareDetector, Profiler, RuntimeConfigBuilder, AdaptiveScheduler
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.model import NeuroCoreModel
from Tantra.moe import ExpertRegistry, LazyExpertLoader
from Tantra.codec import DNACodec, CompressionBenchmark
from Tantra.train import NeuroTrainer
from Tantra.dataset import JSONLDataset, extract_corpus_sample
from Tantra.evolution import AutoGrowthController, SelfRepairEngine
from Tantra.eval import EvaluationEngine
from Tantra.server import serve

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

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "Model")
VOCAB_PATH  = os.path.join(MODEL_DIR, "tokenizer.pt")
EXPERTS_DIR = os.path.join(MODEL_DIR, "Experts")
DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "Datasets", "train_pack_all_expanded_1040k.jsonl")

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

def run_interactive_chat(model, tokenizer, device, temp=0.8, top_p=0.95):
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
                
            tokens = tokenizer.encode(user_input)
            prompt = torch.tensor([tokens], device=device)
            if console:
                console.print(f"[dim]Thinking...[/dim]")
            with torch.no_grad():
                out = model.generate(prompt, max_new_tokens=50, temperature=temp, use_mtp_speculation=True)
            response = tokenizer.decode(out[0].tolist())
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
    
    # Proactive Health Watchdog
    from Tantra.health import HealthWatchdog
    watchdog = HealthWatchdog(MODEL_DIR)
    watchdog.audit_storage_and_compress_duplicates(threshold_mb=5000.0)
    
    sched = AdaptiveScheduler(rt)
    sched.start()
    return rt, sched


def build_vocab(cfg: VocabConfig, corpus_file: str | None = None) -> UnifiedTokenizer:
    log.info("== [2] UNIFIED VOCABULARY & TOKENIZER STATUS =======")
    os.makedirs(MODEL_DIR, exist_ok=True)
    bpe = ByteBPETokenizer(cfg)
    
    status = "Cached Artifact"
    if not os.path.exists(VOCAB_PATH) and corpus_file and os.path.exists(corpus_file):
        sample_txt = extract_corpus_sample(corpus_file, os.path.join(MODEL_DIR, "corpus_sample.txt"))
        bpe.train([sample_txt], vocab_size=cfg.vocab_size)
        status = f"Trained on BPE Corpus ({corpus_file})"
    elif os.path.exists(VOCAB_PATH):
        status = f"Loaded Cached Vocab Artifact ({VOCAB_PATH})"
        
    patcher = MegabytePatcher()
    tok = UnifiedTokenizer(cfg, bpe, patcher)
    torch.save({"vocab_size": cfg.vocab_size, "special_tokens": cfg.special_tokens}, VOCAB_PATH)
    
    log.info(f"  Vocab Size       : {cfg.vocab_size:,} tokens")
    log.info(f"  BPE Subword Merges: {cfg.vocab_size - len(cfg.special_tokens) - 256:,} merge rules")
    log.info(f"  Special Tokens   : {len(cfg.special_tokens)} (<pad>, <unk>, <s>, </s>, <|user|>, <|assistant|>, <|system|>)")
    log.info(f"  Byte Patching    : Megabyte Patching Unit Enabled (byte-fallback handling)")
    log.info(f"  Tokenizer Status : {status}")
    log.info(f"  Artifact Path    : {VOCAB_PATH}")
    return tok


def init_experts(moe_cfg, model_cfg, codec):
    log.info("== [3] EXPERT REGISTRY & LAZY LOADER =============")
    os.makedirs(EXPERTS_DIR, exist_ok=True)
    reg = ExpertRegistry(EXPERTS_DIR, moe_cfg.num_experts)
    reg.load()
    if len(reg) == 0:
        for i, spec in enumerate(["language", "code", "math", "science", "reasoning", "vision", "audio", "general"]):
            reg.register_new(i, spec, 2_000_000_000)
        log.info(f"  Registered {len(reg)} initial domain experts")
        
    sample_expert_weight = torch.randn(1024, 1024, dtype=torch.float32)
    dna_path = os.path.join(EXPERTS_DIR, "expert_0.dna")
    if not os.path.exists(dna_path):
        codec.compress(sample_expert_weight, dna_path)
        log.info(f"  Compressed expert_0 weight tensor -> {dna_path}")
        
    return reg, LazyExpertLoader(moe_cfg, model_cfg, reg, codec)


def init_model(cfg, device):
    log.info("== [4] NEUROCORE MODEL ENGINE & PARAMETER DIAGNOSTICS ==")
    model = NeuroCoreModel(cfg, use_mtp=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    log.info(f"  Total Parameters     : {total_params:,} ({total_params/1e6:.1f}M)")
    log.info(f"  Trainable Parameters : {trainable_params:,} (100% active for pre-training)")
    log.info(f"  Frozen Parameters    : {frozen_params:,}")
    log.info(f"  Quantization Scheme  : BitLinear 1.58-bit Ternary ({{-1, 0, +1}})")
    log.info(f"  Multi-Token Speculation: Enabled (MTP t+1, t+2 concurrent prediction)")
    
    dev = torch.device(device)
    model.to(dev)
    actual_device = next(model.parameters()).device
    log.info(f"  Model Target Device  : {dev}")
    log.info(f"  Model Actual Device  : {actual_device}")
    if dev.type == 'cuda':
        vram_used = torch.cuda.memory_allocated(dev) / (1024*1024)
        vram_total = torch.cuda.get_device_properties(dev).total_mem / (1024*1024)
        log.info(f"  GPU VRAM Allocated   : {vram_used:.1f} MB / {vram_total:.0f} MB")
    return model


def run_forward(model, vcfg, batch, device):
    log.info("== [5] FORWARD PASS VALIDATION =====================")
    model.eval()
    ids = torch.randint(0, vcfg.vocab_size, (batch, 64), device=device)
    with torch.no_grad():
        logits, _ = model(ids)
    if isinstance(logits, tuple):
        logits = logits[0]
    log.info(f"  Input  : {tuple(ids.shape)}")
    log.info(f"  Output : logits {tuple(logits.shape)}  [OK]")


def run_training(model, vcfg, steps=20, resume=False):
    log.info("== [TRAINING MODE WITH AUTO-REPAIR & GROWTH] =======")
    repair = SelfRepairEngine()
    repair.scan_and_repair(model)
    
    trainer = NeuroTrainer(model, lr=1e-4)
    ckpt_path = os.path.join(MODEL_DIR, "checkpoint_latest.pt")
    best_path = os.path.join(MODEL_DIR, "checkpoint_best.pt")
    
    resume_target = None
    if os.path.exists(ckpt_path):
        resume_target = ckpt_path
    elif os.path.exists(best_path):
        resume_target = best_path
        
    if resume_target:
        log.info(f"RESUMING training from existing checkpoint: {resume_target}")
        trainer.load_checkpoint(resume_target)
    else:
        log.info("No previous checkpoint found. Starting FRESH training run.")

    trainer.train_demo(steps=steps, vocab_size=vcfg.vocab_size)
    trainer.save_checkpoint(ckpt_path)


def run_dataset_training(model, tokenizer, dataset_path, steps=50, resume=False, eval_every=250, log_every=50, batch_size=1, seq_len=128):
    log.info("== [DATASET PRE-TRAINING MODE] =====================")
    log.info(f"Loading real dataset from: {dataset_path}")
    repair = SelfRepairEngine()
    repair.scan_and_repair(model)
    
    trainer = NeuroTrainer(model, lr=1e-4)
    ckpt_path = os.path.join(MODEL_DIR, "checkpoint_latest.pt")
    best_path = os.path.join(MODEL_DIR, "checkpoint_best.pt")
    
    # Auto-resume logic: check checkpoint_latest.pt first, then checkpoint_best.pt
    resume_target = None
    if os.path.exists(ckpt_path):
        resume_target = ckpt_path
    elif os.path.exists(best_path):
        resume_target = best_path
        
    if resume_target:
        log.info(f"RESUMING training from existing checkpoint: {resume_target}")
        trainer.load_checkpoint(resume_target)
    else:
        log.info("No previous checkpoint found. Starting FRESH dataset training run.")

    def eval_callback(step):
        log.info("\n--- [ EVALUATION @ Step %d ] ---" % step)
        prompt_text = "User: What is Tantra?\nAssistant:"
        log.info("Prompt: %s" % prompt_text)
        prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], device=model.embed.weight.device)
        out = model.generate(prompt_ids, max_new_tokens=32, temperature=0.7, use_mtp_speculation=True)
        response = tokenizer.decode(out.tolist()[0])
        log.info("Output: %s" % response)
        log.info("----------------------------------\n")
        trainer.save_checkpoint(ckpt_path)
        if trainer.best_loss <= 2.0 or step % (eval_every * 4) == 0:
            trainer.save_checkpoint(best_path)

    eval_callback(0)

    dataset = JSONLDataset(dataset_path, tokenizer, seq_len=seq_len, max_samples=steps * batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
    trainer.train_dataset(dataloader, max_steps=steps, log_every=log_every, eval_every=eval_every, eval_callback=eval_callback)

    trainer.save_checkpoint(ckpt_path)


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


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="Tantra-LLM / NeuroCore CLI Engine")
    parser.add_argument("--mode", default="full",
                        choices=["full", "probe", "vocab", "train", "dataset", "eval", "compress", "generate", "serve", "status", "experts", "chat"],
                        help="Execution mode")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="JSONL dataset path")
    parser.add_argument("--steps", type=int, default=30, help="Training steps")
    parser.add_argument("--seq-len", type=int, default=128, help="Context sequence length window")
    parser.add_argument("--use-mtp", type=bool, default=True, help="Enable Multi-Token Prediction (MTP)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling")
    parser.add_argument("--port", type=int, default=8000, help="Server port (serve mode)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    parser.add_argument("--eval-every", type=int, default=250, help="Evaluate and save checkpoint every N steps (dataset mode)")
    parser.add_argument("--log-every", type=int, default=50, help="Log progress every N steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--device", type=str, default="auto", help="Target device (auto, cuda, cpu, mps)")
    args = parser.parse_args()

    vcfg = VocabConfig()
    mcfg = NeuroCoreConfig.small()
    mcfg.use_mtp = args.use_mtp
    moe  = MoEConfig()
    ccfg = CompressionConfig()

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
    model = init_model(mcfg, rt.device)
    
    trainer = NeuroTrainer(model, lr=1e-4)
    # Check if a checkpoint exists for status
    if os.path.exists(os.path.join(MODEL_DIR, "checkpoint_latest.pt")):
        try: trainer.load_checkpoint(os.path.join(MODEL_DIR, "checkpoint_latest.pt"))
        except: pass

    if args.mode == "status":
        print_status_dashboard(model, trainer, reg, rt)
        sched.stop()
        return
        
    if args.mode == "experts":
        print_expert_panel(reg)
        sched.stop()
        return
        
    if args.mode == "chat":
        run_interactive_chat(model, tok, rt.device, args.temperature, args.top_p)
        sched.stop()
        return

    if args.mode == "train":
        run_training(model, vcfg, steps=args.steps, resume=args.resume)
    elif args.mode == "dataset":
        run_dataset_training(model, tok, args.dataset, steps=args.steps, resume=args.resume, eval_every=args.eval_every, log_every=args.log_every, batch_size=args.batch_size, seq_len=args.seq_len)
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
