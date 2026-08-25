"""
Tantra/world_eval.py — Zero-Shot World Knowledge Evaluation Suite (MMLU / Multi-Subject).
Evaluates true generalization on real-world multi-domain benchmarks using standard logit scoring.
"""

import os
import json
import torch
from typing import Dict, Any, List, Optional

BENCHMARK_PATH = os.path.join("Datasets", "benchmarks", "world_mmlu.jsonl")


def evaluate_zero_shot_world_knowledge(model: torch.nn.Module, tokenizer: Any, benchmark_path: str = BENCHMARK_PATH) -> Dict[str, float]:
    """
    Evaluates zero-shot accuracy on standardized multi-subject MMLU questions.
    Uses logit-comparison across option choices (A, B, C, D) without requiring text generation.
    """
    if not os.path.exists(benchmark_path):
        return {}

    questions: List[Dict[str, Any]] = []
    with open(benchmark_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    questions.append(json.loads(line))
                except Exception:
                    pass

    if not questions:
        return {}

    raw_m = getattr(model, "_orig_mod", model)
    device = next(raw_m.parameters()).device
    was_training = raw_m.training
    raw_m.eval()

    candidate_tokens = {}
    for letter in ["A", "B", "C", "D"]:
        toks = tokenizer.encode(f" {letter}") or tokenizer.encode(letter)
        if toks:
            candidate_tokens[letter] = toks[-1]

    if len(candidate_tokens) < 4:
        if was_training:
            raw_m.train()
        return {}

    correct = 0
    total = 0

    with torch.no_grad():
        for item in questions:
            q_text = item.get("question", "")
            options = item.get("options", {})
            correct_ans = item.get("answer", "").strip().upper()

            opt_str = "\n".join([f"({k}) {v}" for k, v in options.items()])
            prompt = f"<|user|>\nQuestion: {q_text}\n{opt_str}\nWhat is the correct option letter?\n\n<|assistant|>\n("

            p_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
            if p_ids.size(1) == 0:
                continue

            out = raw_m(p_ids)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            last_logits = logits[0, -1, :]

            scores = {letter: last_logits[tok_id].item() for letter, tok_id in candidate_tokens.items()}
            best_letter = max(scores, key=scores.get)

            if best_letter == correct_ans:
                correct += 1
            total += 1

    if was_training:
        raw_m.train()
    acc = (correct / total * 100.0) if total > 0 else 0.0
    return {
        "world_mmlu_accuracy": acc,
        "correct_samples": correct,
        "total_samples": total,
    }
