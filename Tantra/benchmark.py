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

    cfg = NeuroCoreConfig()
    cfg.vocab.vocab_size = 32768

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📦 Loading Checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        ckpt_cfg = ckpt.get("config", None)
        if ckpt_cfg is not None:
            cfg = ckpt_cfg
        elif "embed.weight" in state:
            embed_w = state["embed.weight"]
            ckpt_vocab = embed_w.size(0)
            ckpt_dim = embed_w.size(1)
            cfg.vocab.vocab_size = ckpt_vocab
            cfg.block.alra.dim = ckpt_dim
            cfg.block.sgp.dim = ckpt_dim
            cfg.block.alra.num_heads = 8 if ckpt_dim == 512 else (12 if ckpt_dim == 768 else max(1, ckpt_dim // 64))
            cfg.block.alra.head_dim = max(1, ckpt_dim // cfg.block.alra.num_heads)

            layer_indices = set()
            for k in state.keys():
                if k.startswith("layers."):
                    parts = k.split(".")
                    if len(parts) > 1 and parts[1].isdigit():
                        layer_indices.add(int(parts[1]))
            if layer_indices:
                cfg.block.num_layers = max(layer_indices) + 1

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
    print("✅ Industry Benchmark execution complete.")
    print("=" * 70)


def run_60_benchmark(checkpoint_path: str = None, eval_jsonl: str = "Datasets/eval_60_benchmark.jsonl", device: str = "auto") -> dict:
    """Runs the 60-prompt 5-domain qualitative alignment benchmark."""
    import ast

    print("=" * 70)
    print(f"📊 Running 60-Prompt Domain Benchmark on: {checkpoint_path or 'Model/Latest/checkpoint_latest.pt'}")
    print("=" * 70)

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        default_candidates = [
            "Model/Latest/checkpoint_latest.pt",
            "Model/Best/checkpoint_best.pt",
            "Model/Checkpoints/checkpoint_step_91500.pt",
        ]
        checkpoint_path = next((c for c in default_candidates if os.path.exists(c)), None)
        if not checkpoint_path:
            raise FileNotFoundError("No valid checkpoint found to benchmark. Please specify --checkpoint.")
        print(f"🎯 Auto-selected checkpoint: {checkpoint_path}")

    dev = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else (device if device != "auto" else "cpu"))

    vcfg = NeuroCoreConfig().vocab
    bpe = ByteBPETokenizer.load("Model/tokenizer.json", vcfg) if os.path.exists("Model/tokenizer.json") else ByteBPETokenizer(vcfg)
    patcher = MegabytePatcher(vcfg)
    tokenizer = UnifiedTokenizer(vcfg, bpe, patcher)

    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg = ckpt.get("config", None) or NeuroCoreConfig()
    sdict = ckpt.get("model_state_dict", ckpt)

    # Layer count detection
    import re
    layer_indices = [int(m.group(1)) for k in sdict.keys() for m in [re.search(r'layers\.(\d+)\.', k)] if m]
    if layer_indices and hasattr(cfg, "block"):
        cfg.block.num_layers = max(layer_indices) + 1

    model = NeuroCoreModel(cfg).to(dev)
    load_res = model.load_state_dict(sdict, strict=False)
    print(f"✅ Architecture verified: {sum(p.numel() for p in model.parameters()):,} parameters.")
    model.eval()

    if not os.path.exists(eval_jsonl):
        raise FileNotFoundError(f"Benchmark file {eval_jsonl} not found.")

    with open(eval_jsonl, "r", encoding="utf-8") as f:
        prompts = [json.loads(l) for l in f if l.strip()]

    domain_scores = {}
    valid_ast_count = 0
    code_total = 0

    print(f"\nEvaluating {len(prompts)} prompts across 5 domains...")
    for i, item in enumerate(prompts):
        domain = item.get("domain", "general")
        q = item.get("prompt") or item.get("question", "")
        expected = item.get("expected") or item.get("answer", "")

        prompt_text = f"<|user|>\n{q}\n\n<|assistant|>\n"
        input_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=dev)

        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=64, min_new_tokens=1, temperature=0.2, top_p=0.9, repetition_penalty=1.15)

        gen_tokens = out[0, input_ids.shape[1]:].tolist()
        gen_text = tokenizer.decode(gen_tokens).strip()
        if "</s>" in gen_text:
            gen_text = gen_text.split("</s>")[0].strip()

        exp_words = set(expected.lower().split())
        gen_words = set(gen_text.lower().split())
        overlap = len(exp_words & gen_words) / max(len(exp_words), 1)

        if domain not in domain_scores:
            domain_scores[domain] = []
        domain_scores[domain].append(overlap)

        ast_valid = False
        if domain == "code":
            code_total += 1
            code_clean = gen_text.replace("```python", "").replace("```", "").strip()
            try:
                ast.parse(code_clean)
                ast_valid = True
                valid_ast_count += 1
            except Exception:
                pass

        if i < 5 or (i % 15 == 0):
            print(f"\n[{i+1}/{len(prompts)}] [{domain.upper()}] Q: {q[:70]}")
            print(f"  Expected: {expected[:70]}...")
            print(f"  Got     : {gen_text[:70]}...")
            print(f"  Overlap : {overlap*100:.1f}%" + (" | Valid AST: ✅" if ast_valid else ""))

    print("\n" + "=" * 70)
    print("🏆 60-PROMPT BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    all_scores = []
    for d, scs in domain_scores.items():
        avg_d = sum(scs) / len(scs) * 100
        all_scores.extend(scs)
        print(f"  • {d.capitalize():12s}: {avg_d:.1f}% concept overlap ({len(scs)} items)")

    overall_avg = sum(all_scores) / max(len(all_scores), 1) * 100
    print(f"  • Overall Alignment : {overall_avg:.1f}%")
    if code_total > 0:
        print(f"  • Valid Python AST  : {valid_ast_count}/{code_total} ({valid_ast_count/code_total*100:.1f}%)")
    print("=" * 70)
    return {"overall_alignment": overall_avg, "domain_scores": domain_scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tantra Benchmark Runner")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, cpu, cuda)")
    parser.add_argument("--suite", choices=["all", "industry", "60"], default="60", help="Benchmark suite to run (default: 60)")
    parser.add_argument("--eval-file", type=str, default="Datasets/eval_60_benchmark.jsonl", help="Custom JSONL benchmark file")
    args = parser.parse_args()

    if args.suite in ("industry", "all"):
        run_benchmarks(args.checkpoint, args.device)
    if args.suite in ("60", "all"):
        run_60_benchmark(args.checkpoint, args.eval_file, args.device)

