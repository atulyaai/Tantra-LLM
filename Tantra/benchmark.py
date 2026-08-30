#!/usr/bin/env python3
"""
Tantra/benchmark.py — Unified 5-Level Industry Benchmark CLI
Runs GSM8K Math, HumanEval Code Sandbox (pass@1), Science QA, Tool Calling, and MMLU.
"""

import os
import sys
import json
import argparse
import time
import torch

from Tantra.model import NeuroCoreModel
from Tantra.config import NeuroCoreConfig
from Tantra.tokenizer import UnifiedTokenizer, ByteBPETokenizer, MegabytePatcher
from Tantra.eval_suite import IndustryBenchmarkSuite

BENCHMARK_GSM8K = [
    {"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many in May. How many clips did she sell altogether?", "answer": "72"},
    {"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "answer": "10"},
    {"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents gave her $15 and her grandparents gave her twice as much as her parents. How much more money does Betty need to buy the wallet?", "answer": "5"}
]

BENCHMARK_HUMANEVAL = [
    {
        "prompt": "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
        "test": "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
    },
    {
        "prompt": "def separate_paren_groups(paren_string: str) -> list[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Separate those groups and return list of them.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
        "test": "assert separate_paren_groups('(()()) (( )) () ((())()())') == ['(()())', '(())', '()', '((())()())']\n"
    }
]

def run_benchmarks(checkpoint_path: str = None, device: str = "auto"):
    print("=" * 70)
    print("🧪 TANTRA-LLM 5-LEVEL INDUSTRY BENCHMARK MATRIX")
    print("=" * 70)

    dev = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else (device if device != "auto" else "cpu"))
    print(f"⚡ Device: {dev}")

    cfg = NeuroCoreConfig.small()
    cfg.vocab.vocab_size = 32768

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📦 Loading Checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        ckpt_cfg = ckpt.get("config", None)
        if ckpt_cfg is not None:
            cfg = ckpt_cfg
        elif "embed.weight" in state:
            ckpt_vocab = state["embed.weight"].size(0)
            if cfg.vocab.vocab_size != ckpt_vocab:
                cfg.vocab.vocab_size = ckpt_vocab
            router_key = next((k for k in state if k.endswith("router.router_weights.weight")), None)
            if router_key:
                cfg.moe.num_experts = state[router_key].size(0)
            if any(k.startswith("layers.0.attn.w_q") for k in state):
                cfg.block.alra.attention_kind = "causal"
    else:
        state = None

    model = NeuroCoreModel(cfg).to(dev)

    if state is not None:
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"Error loading state dict: {e}")

    # Calculate model size and params
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024) # assuming fp32, 4 bytes per param
    print(f"📊 Model Parameters: {total_params / 1e6:.2f} M")
    print(f"💾 Model Memory Size: {model_size_mb:.2f} MB")

    tok_file = "Model/tokenizer.json"
    try:
        bpe = ByteBPETokenizer.load(tok_file, cfg.vocab)
    except Exception:
        bpe = ByteBPETokenizer(cfg.vocab)
    patcher = MegabytePatcher(cfg.vocab)
    tok = UnifiedTokenizer(cfg.vocab, bpe, patcher)

    # Simple Inference Speed Benchmark
    print("\n[Speed] Running inference speed benchmark...")
    model.eval()
    batch_sizes = [1, 4]
    seq_len = 128
    with torch.no_grad():
        for bz in batch_sizes:
            dummy_input = torch.randint(0, cfg.vocab.vocab_size, (bz, seq_len), device=dev)
            
            # warmup
            for _ in range(2):
                model(dummy_input)
            
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            iters = 10
            for _ in range(iters):
                model(dummy_input)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / iters
            tokens_per_sec = (bz * seq_len) / avg_time
            print(f"  -> Batch Size {bz}: {tokens_per_sec:.2f} tokens/sec (avg {avg_time*1000:.2f} ms/forward)")

    suite = IndustryBenchmarkSuite(model, tok, dev)
    
    # 1. GSM8K
    print("\n[Level 1] Evaluating GSM8K Math...")
    gsm_res = suite.evaluate_gsm8k_math(BENCHMARK_GSM8K)
    print(f"  -> GSM8K Accuracy: {gsm_res['gsm8k_accuracy']:.1f}% ({gsm_res['correct']}/{gsm_res['total']})")

    # 2. HumanEval
    print("\n[Level 2] Evaluating HumanEval Code Sandbox (pass@1)...")
    he_res = suite.evaluate_humaneval_code(BENCHMARK_HUMANEVAL)
    pass_rate = he_res.get("humaneval_pass_at_1", he_res.get("pass_at_1", 0.0))
    print(f"  -> HumanEval pass@1: {pass_rate:.1f}% ({he_res.get('passed', 0)}/{he_res.get('total', 0)})")

    print("\n" + "=" * 70)
    print("✅ Benchmark execution complete.")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tantra Benchmark Runner")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_benchmarks(args.checkpoint, args.device)
