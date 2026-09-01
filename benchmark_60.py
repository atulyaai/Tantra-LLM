"""Reproducible held-out prompt benchmark for Tantra checkpoints.

The loader deliberately mirrors main.py's legacy-MoE detection and rejects
partial checkpoint loads.  A score produced from a silently incompatible model
is worse than no score at all.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from Tantra.config import VocabConfig
from Tantra.bitnet import BitLinear
from Tantra.model import NeuroCoreModel
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer


def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def _rouge_l(reference: str, hypothesis: str) -> float:
    """Small dependency-free ROUGE-L F1 implementation."""
    a, b = _words(reference), _words(hypothesis)
    if not a or not b:
        return 0.0
    row = [0] * (len(b) + 1)
    for token_a in a:
        previous = 0
        for j, token_b in enumerate(b, 1):
            saved = row[j]
            if token_a == token_b:
                row[j] = previous + 1
            else:
                row[j] = max(row[j], row[j - 1])
            previous = saved
    lcs = row[-1]
    precision, recall = lcs / len(b), lcs / len(a)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def load_checkpoint_model(checkpoint_path: Path) -> tuple[NeuroCoreModel, UnifiedTokenizer]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict.")
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint has no saved model config; refusing an unsafe benchmark load.")

    state = checkpoint["model_state_dict"]
    has_legacy_router = any(".router." in key for key in state)
    is_real_top1 = bool(getattr(config.moe, "real_top1", False) and getattr(config.moe, "num_experts", 1) > 1)
    legacy_compat = has_legacy_router and not is_real_top1 and getattr(config.moe, "num_experts", 1) > 1
    model = NeuroCoreModel(
        config,
        use_mtp=getattr(config, "use_mtp", True),
        use_moe=is_real_top1 or legacy_compat,
        compatibility_legacy_moe=legacy_compat,
    )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unsafe checkpoint load: "
            f"missing={incompatible.missing_keys[:8]}, unexpected={incompatible.unexpected_keys[:8]}"
        )
    model.eval()
    # Checkpoints contain trainable FP32 shadow weights.  Benchmarks must not
    # quantize those 110M weights once per generated token; convert each
    # BitLinear exactly once and reuse its cached ternary matrix.
    bitlinear_count = 0
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.to_inference_mode()
            bitlinear_count += 1

    vocab_config = VocabConfig()
    tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = Path("Model/tokenizer.json")
    bpe = ByteBPETokenizer.load(str(tokenizer_path), vocab_config)
    tokenizer = UnifiedTokenizer(vocab_config, bpe, MegabytePatcher())
    print(f"Loaded {checkpoint_path}: legacy_compat={legacy_compat}; zero incompatible tensors; cached {bitlinear_count} BitLinear layers for inference.")
    return model, tokenizer


def run_benchmark(checkpoint_path: str, eval_jsonl: str = "Datasets/eval_60_benchmark.jsonl", max_new_tokens: int = 96) -> dict:
    model, tokenizer = load_checkpoint_model(Path(checkpoint_path))
    items = [json.loads(line) for line in Path(eval_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(items) != 60:
        raise ValueError(f"Expected 60 held-out prompts, found {len(items)} in {eval_jsonl}.")

    scores = defaultdict(list)
    exact = defaultdict(int)
    code_valid = code_total = 0
    for item in items:
        domain, prompt, expected = item["domain"], item["prompt"], item["expected"]
        text = f"<|user|>\n{prompt}\n\n<|assistant|>\n"
        input_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
        with torch.inference_mode():
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, min_new_tokens=1,
                                    temperature=0.2, top_p=0.9, repetition_penalty=1.15)
        generated = tokenizer.decode(output[0, input_ids.shape[1]:].tolist()).split("</s>", 1)[0].strip()
        score = _rouge_l(expected, generated)
        scores[domain].append(score)
        exact[domain] += int(generated.strip().lower() == expected.strip().lower())
        if domain == "code":
            code_total += 1
            try:
                ast.parse(generated.replace("```python", "").replace("```", "").strip())
                code_valid += 1
            except SyntaxError:
                pass
        print(f"[{domain}] {prompt}\n  expected: {expected[:100]}\n  generated: {generated[:100]}\n  ROUGE-L: {score:.3f}")

    all_scores = [score for domain_scores in scores.values() for score in domain_scores]
    result = {
        "overall_rouge_l": sum(all_scores) / len(all_scores),
        "by_domain_rouge_l": {domain: sum(values) / len(values) for domain, values in scores.items()},
        "exact_matches": dict(exact),
        "valid_python_ast": {"valid": code_valid, "total": code_total},
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--eval", default="Datasets/eval_60_benchmark.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    run_benchmark(args.checkpoint, args.eval, args.max_new_tokens)
