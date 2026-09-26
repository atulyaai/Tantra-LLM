"""
Tantra/eval_suite.py — Honest evaluation.

1. validation_metrics(): loss / perplexity / next-token accuracy on held-out data.
2. throughput(): tokens per second on this machine.
3. run_probe(): the fixed 50-question recall test (Datasets/probe_50.jsonl).
     answer_loss  how surprised the model is by the right answer (lower = better,
                  moves early in training)
     hit_rate     greedy answer contains an expected keyword ("remembered X/50")
   Same questions every time -> numbers are comparable across steps and runs.
   Results are appended to Model/probe_history.jsonl.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from Tantra.config import EOS_ID
from Tantra.utils import get_logger

log = get_logger("tantra.eval")
IGNORE_INDEX = -100


def _logits(out: Any) -> torch.Tensor:
    first = out[0] if isinstance(out, (tuple, list)) else out
    return first[0] if isinstance(first, (tuple, list)) else first


@torch.no_grad()
def validation_metrics(model: torch.nn.Module, loader: Any, max_batches: int = 50) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    loss_sum = 0.0
    n_tok = top1 = top5 = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        logits = _logits(model(x, use_latent_reasoning=False)).float()
        flat, tgt = logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        valid = tgt != IGNORE_INDEX
        if not valid.any():
            continue
        loss_sum += F.cross_entropy(flat[valid], tgt[valid], reduction="sum").item()
        n_tok += int(valid.sum())
        top1 += int((flat[valid].argmax(-1) == tgt[valid]).sum())
        top5 += int((flat[valid].topk(5, dim=-1).indices == tgt[valid, None]).any(-1).sum())
    if was_training:
        model.train()
    loss = loss_sum / max(n_tok, 1)
    return {"loss": round(loss, 4), "perplexity": round(math.exp(min(loss, 20.0)), 2),
            "top1_accuracy_percent": round(100.0 * top1 / max(n_tok, 1), 2),
            "top5_accuracy_percent": round(100.0 * top5 / max(n_tok, 1), 2), "tokens": n_tok}


@torch.no_grad()
def throughput(model: torch.nn.Module, vocab_size: int, seq_len: int = 256, runs: int = 5) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    x = torch.randint(0, vocab_size, (1, seq_len), device=device)
    model(x, use_latent_reasoning=False)
    t = time.perf_counter()
    for _ in range(runs):
        model(x, use_latent_reasoning=False)
    dt = time.perf_counter() - t
    return {"forward_tokens_per_sec": round(seq_len * runs / dt, 1)}


# ── Fixed 50-question probe ──────────────────────────────────────────────────

USER_TAG = "<|user|>\n"
ASSISTANT_TAG = "<|assistant|>\n"


def build_prompt(question: str) -> str:
    # Must match Tantra/dataset.py chat formatting exactly.
    return f"{USER_TAG}{question.strip()}\n\n{ASSISTANT_TAG}"


def load_probe(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _encode(tokenizer: Any, text: str) -> List[int]:
    try:
        return list(tokenizer.encode(text, modality="text"))
    except TypeError:
        return list(tokenizer.encode(text))


@torch.no_grad()
def run_probe(model: torch.nn.Module, tokenizer: Any, items: List[Dict[str, Any]],
              step: int, generate: bool = True, max_new_tokens: int = 40,
              history_path: Optional[str] = None) -> Dict[str, Any]:
    if not items:
        return {}
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    t0 = time.time()

    losses: List[float] = []
    per_cat: Dict[str, List[int]] = {}
    hits = 0
    samples: List[Dict[str, str]] = []

    for it in items:
        prompt_ids = _encode(tokenizer, build_prompt(it["question"]))
        ans_ids = _encode(tokenizer, it["reference"]) + [EOS_ID]
        ids = torch.tensor([prompt_ids + ans_ids], device=device)
        logits = _logits(model(ids, use_latent_reasoning=False)).float()
        # positions predicting the answer tokens
        start = len(prompt_ids) - 1
        pred = logits[0, start:start + len(ans_ids)]
        tgt = ids[0, len(prompt_ids):len(prompt_ids) + len(ans_ids)]
        losses.append(F.cross_entropy(pred, tgt).item())

        if generate:
            p = torch.tensor([prompt_ids], device=device)
            out = model.generate(p, max_new_tokens=max_new_tokens, temperature=0.0,
                                 top_p=1.0, repetition_penalty=1.1, no_repeat_ngram_size=3,
                                 eos_token_id=EOS_ID, min_new_tokens=1)
            text = tokenizer.decode(out[0, len(prompt_ids):].tolist())
            ok = any(k.lower() in text.lower() for k in it["keywords"])
            hits += int(ok)
            per_cat.setdefault(it.get("category", "all"), []).append(int(ok))
            if len(samples) < 3 or ok:
                samples.append({"q": it["question"][:60], "a": text.replace("\n", " ")[:80], "hit": ok})

    if was_training:
        model.train()

    result: Dict[str, Any] = {
        "step": int(step),
        "answer_loss": round(sum(losses) / len(losses), 4),
        "answer_ppl": round(math.exp(min(20.0, sum(losses) / len(losses))), 2),
        "n": len(items),
        "seconds": round(time.time() - t0, 1),
        "time": int(time.time()),
    }
    if generate:
        result["hit_rate"] = round(hits / len(items), 4)
        result["hits"] = hits
        result["by_category"] = {k: f"{sum(v)}/{len(v)}" for k, v in per_cat.items()}

    msg = f"  [PROBE-50 @ {step:,}] answer_loss={result['answer_loss']:.3f}"
    if generate:
        msg += f"  remembered={hits}/{len(items)}  {result['by_category']}"
    log.info(msg)
    if generate:
        for s in samples[:5]:
            log.info(f"     {'✓' if s['hit'] else '·'} {s['q']} → {s['a']}")

    if history_path:
        try:
            os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning(f"Could not write probe history: {exc}")
    return result
