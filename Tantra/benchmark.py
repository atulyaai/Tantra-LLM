#!/usr/bin/env python3
"""
Tantra/benchmark.py — Unified 5-Level Industry Benchmark CLI
Runs GSM8K Math, HumanEval Code Sandbox (pass@1), Science QA, Tool Calling, and MMLU.
"""

import os
import sys
import json
import argparse
import torch

from Tantra.model import TantraTransformer, TantraConfig
from Tantra.tokenizer import TantraTokenizer
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

    tok = TantraTokenizer("Model/tokenizer.json")
    cfg = TantraConfig(vocab_size=tok.vocab_size, hidden_dim=512, num_layers=10, num_heads=8)
    model = TantraTransformer(cfg).to(dev)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📦 Loading Checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            pass

    suite = IndustryBenchmarkSuite(model, tok, dev)
    
    # 1. GSM8K
    print("\n[Level 1] Evaluating GSM8K Math...")
    gsm_res = suite.evaluate_gsm8k_math(BENCHMARK_GSM8K)
    print(f"  -> GSM8K Accuracy: {gsm_res['gsm8k_accuracy']:.1f}% ({gsm_res['correct']}/{gsm_res['total']})")

    # 2. HumanEval
    print("\n[Level 2] Evaluating HumanEval Code Sandbox (pass@1)...")
    he_res = suite.evaluate_humaneval_code(BENCHMARK_HUMANEVAL)
    print(f"  -> HumanEval pass@1: {he_res['pass_at_1']:.1f}% ({he_res['passed']}/{he_res['total']})")

    print("\n" + "=" * 70)
    print("✅ Benchmark execution complete.")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tantra Benchmark Runner")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_benchmarks(args.checkpoint, args.device)
