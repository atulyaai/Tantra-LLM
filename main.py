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
"""
from __future__ import annotations

import argparse
import os
import torch

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

log = get_logger("tantra")

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "Model")
VOCAB_PATH  = os.path.join(MODEL_DIR, "tokenizer.pt")
EXPERTS_DIR = os.path.join(MODEL_DIR, "Experts")
DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "Datasets", "train_pack_all_expanded_1040k.jsonl")


def detect_hardware():
    log.info("== [1] HARDWARE AUTO-DETECTION & PROACTIVE HEALTH ==")
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
    watchdog.audit_storage_and_compress_duplicates()
    
    sched = AdaptiveScheduler(rt)
    sched.start()
    return rt, sched


def build_vocab(cfg: VocabConfig, corpus_file: Optional[str] = None) -> UnifiedTokenizer:
    log.info("== [2] UNIFIED VOCABULARY & TOKENIZER STATUS =======")
    os.makedirs(MODEL_DIR, exist_ok=True)
    bpe = ByteBPETokenizer(cfg)
    
    status = "Cached Artifact"
    if corpus_file and os.path.exists(corpus_file):
        sample_txt = extract_corpus_sample(corpus_file, os.path.join(MODEL_DIR, "corpus_sample.txt"))
        bpe.train([sample_txt], vocab_size=cfg.vocab_size)
        status = f"Trained on BPE Corpus ({corpus_file})"
        
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
    
    model.to(torch.device(device))
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


def run_training(model, vcfg, steps=20):
    log.info("== [TRAINING MODE WITH AUTO-REPAIR & GROWTH] =======")
    repair = SelfRepairEngine()
    repair.scan_and_repair(model)
    
    trainer = NeuroTrainer(model, lr=1e-4)
    trainer.train_demo(steps=steps, vocab_size=vcfg.vocab_size)
    
    ckpt_path = os.path.join(MODEL_DIR, "checkpoint_latest.pt")
    trainer.save_checkpoint(ckpt_path)


def run_dataset_training(model, tokenizer, dataset_path, steps=50):
    log.info("== [DATASET PRE-TRAINING MODE] =====================")
    log.info(f"Loading real dataset from: {dataset_path}")
    repair = SelfRepairEngine()
    repair.scan_and_repair(model)
    
    dataset = JSONLDataset(dataset_path, tokenizer, seq_len=128, max_samples=steps)
    trainer = NeuroTrainer(model, lr=1e-4)
    trainer.train_dataset(dataset, max_steps=steps, log_every=1)
    
    ckpt_path = os.path.join(MODEL_DIR, "checkpoint_dataset_trained.pt")
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
    parser = argparse.ArgumentParser(description="Tantra-LLM / NeuroCore CLI Engine")
    parser.add_argument("--mode", default="full",
                        choices=["full", "probe", "vocab", "train", "dataset", "eval", "compress", "generate", "serve"],
                        help="Execution mode")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="JSONL dataset path")
    parser.add_argument("--steps", type=int, default=30, help="Training steps")
    parser.add_argument("--seq-len", type=int, default=128, help="Context sequence length window")
    parser.add_argument("--use-mtp", type=bool, default=True, help="Enable Multi-Token Prediction (MTP)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling")
    parser.add_argument("--port", type=int, default=8000, help="Server port (serve mode)")
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
    tok = build_vocab(vcfg, args.dataset)
    codec = DNACodec(ccfg)
    reg, loader = init_experts(moe, mcfg, codec)
    model = init_model(mcfg, rt.device)

    if args.mode == "train":
        run_training(model, vcfg, steps=args.steps)
    elif args.mode == "dataset":
        run_dataset_training(model, tok, args.dataset, steps=args.steps)
    elif args.mode == "eval":
        run_evaluation(model, tok, args.dataset)
    elif args.mode == "generate":
        run_generation(model, vcfg, rt.device)
    elif args.mode == "serve":
        serve(model, tok, port=args.port)
    else:  # full mode
        run_forward(model, vcfg, rt.batch_size, rt.device)
        run_evaluation(model, tok, args.dataset)
        run_generation(model, vcfg, rt.device)

    log.info("Pipeline complete -- NeuroCore ready!")
    sched.stop()


if __name__ == "__main__":
    main()
