"""Real-data bilingual conversation experiment for Tantra (10M parameters).

Uses the revision-pinned, human-reviewed OpenAssistant/oasst1 conversation trees.
Splits by complete conversation family before tokenization/training. No synthetic
training examples are added. This is a small experiment, not a capable assistant.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

import torch

from Tantra import ModelConfig, ROOT, sha256
from Tantra.pretrain import sources, location
from Tantra.tokenizer import TantraTokenizer
from Tantra.train import EncodedDataset, TantraModel, TrainArgs, _loop, eval_loss


def accepted(node):
    if node.get("deleted") or node.get("synthetic") or node.get("review_result") is not True:
        return False
    labels = node.get("labels") or {}
    if any((labels.get(k) or {}).get("value", 0) > 0
           for k in ("spam", "pii", "lang_mismatch", "not_appropriate")):
        return False
    return isinstance(node.get("text"), str) and bool(node["text"].strip())


def walk(node, history=(), language=None, ancestors=frozenset()):
    mid = node.get("message_id")
    if mid in ancestors or len(history) >= 12 or not accepted(node):
        return
    lang = node.get("lang")
    if lang not in ("hi", "en") or (language and language != lang):
        return
    role = {"prompter": "user", "assistant": "assistant"}.get(node.get("role"))
    if role is None or (not history and role != "user") or (history and history[-1]["role"] == role):
        return
    messages = history + ({"role": role, "content": node["text"].strip()},)
    if role == "assistant" and node.get("rank") in (None, 0):
        yield lang, messages
    for child in node.get("replies", []):
        yield from walk(child, messages, lang, ancestors | {mid})


def split_name(family):
    bucket = int(hashlib.sha256(family.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


def acquire_oasst():
    source = next(s for s in sources() if s["id"] == "oasst1")
    path = location(source)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = (f"https://huggingface.co/datasets/{source['repo']}/resolve/"
               f"{source['revision']}/{source['path']}?download=true")
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    if path.stat().st_size != source["size"] or sha256(path) != source["sha256"]:
        raise ValueError("OASST1 source size/checksum does not match pinned data lock")
    return path, source


def load_rows(path, source):
    splits = {s: [] for s in ("train", "validation", "test")}
    seen = set()
    counts = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            tree = json.loads(line)
            family = "oasst1:" + tree["message_tree_id"]
            for lang, messages in walk(tree.get("prompt") or {}):
                if any(len(m["content"]) > 4000 for m in messages):
                    counts[f"{lang}_too_long"] += 1
                    continue
                key = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                split = split_name(family)
                splits[split].append({
                    "messages": list(messages), "language": lang, "family_id": family,
                    "source": "OpenAssistant/oasst1", "source_revision": source["revision"],
                    "license": source["license"],
                })
                counts[f"{lang}_{split}"] += 1
    return splits, counts


def make_prompts(tok, model, output):
    prompts = [
        ("en", "Hello! How are you today?"),
        ("en", "Hi, what can you help me with?"),
        ("en", "I had a difficult day. Can we talk?"),
        ("hi", "नमस्ते! आप कैसे हैं?"),
        ("hi", "आज मेरा दिन अच्छा नहीं रहा। क्या हम बात कर सकते हैं?"),
        ("hi", "आप मेरी किस तरह मदद कर सकते हैं?"),
    ]
    model.eval()
    results = []
    for lang, prompt in prompts:
        ids, _ = tok.encode_chat([{"role": "user", "content": prompt}], add_generation_prompt=True)
        x = torch.tensor([ids[-model.cfg.max_seq_len:]], dtype=torch.long)
        generated = list(model.generate(x, max_new_tokens=64, temperature=0.7,
                                        top_k=30, top_p=0.9, stop_ids=[tok.end_id],
                                        repetition_penalty=1.12, no_repeat_ngram=3))
        results.append({"language": lang, "prompt": prompt, "response": tok.decode(generated)})
    return results


def run(output, steps=300, seed=2026):
    torch.set_num_threads(2)
    random.seed(seed)
    torch.manual_seed(seed)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new; use a new --output path")
    output.mkdir(parents=True, exist_ok=True)
    raw, source = acquire_oasst()
    splits, counts = load_rows(raw, source)
    for lang in ("hi", "en"):
        if counts[f"{lang}_train"] < 25 or counts[f"{lang}_validation"] < 2 or counts[f"{lang}_test"] < 2:
            raise RuntimeError(f"Insufficient reviewed {lang} conversations for honest train/validation/test: {dict(counts)}")
    tok_path = ROOT / "Model" / "tokenizer.json"
    tok = TantraTokenizer.load(tok_path)
    train = EncodedDataset(tok, splits["train"], max_len=192, assistant_only=False)
    validation = EncodedDataset(tok, splits["validation"], max_len=192, assistant_only=False)
    if len(train) < 50 or len(validation) < 4:
        raise RuntimeError("Too few encoded conversation examples after filtering")
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=384, n_layers=4,
                      n_heads=6, n_kv_heads=2, mlp_mult=2.75,
                      max_seq_len=192, dropout=0.05, lora_rank=4)
    model = TantraModel(cfg)
    params = list(model.base_parameters())
    initial_val = eval_loss(model, validation, batch_size=2, max_batches=100)
    _loop(model, params, train, TrainArgs(
        steps=steps, batch_size=2, lr=0.0003, warmup=max(5, steps // 20),
        weight_decay=0.01, max_len=192, log_every=max(1, steps // 10),
        seed=seed, gradient_accumulation=2,
    ), "bilingual-oasst1-10m")
    final_val = eval_loss(model, validation, batch_size=2, max_batches=100)
    test_data = EncodedDataset(tok, splits["test"], max_len=192, assistant_only=False)
    test_loss = eval_loss(model, test_data, batch_size=2, max_batches=100)
    model.save(output / "base.pt")
    tok.save(output / "tokenizer.json")
    generations = make_prompts(tok, model, output)
    report = {
        "experiment": "real bilingual human-reviewed conversation training",
        "source": {"dataset": "OpenAssistant/oasst1", "revision": source["revision"],
                   "license": source["license"], "sha256": source["sha256"]},
        "split": "90/5/5 by complete message_tree_id; no family crosses splits",
        "counts": dict(counts),
        "encoded_examples": {"train": len(train), "validation": len(validation), "test": len(test_data)},
        "model": {"parameters": model.num_params(), "config": cfg.to_dict(),
                  "tokenizer_vocab": tok.vocab_size, "device": "cpu", "seed": seed},
        "training": {"steps": steps, "initial_validation_loss": initial_val,
                     "final_validation_loss": final_val, "held_out_test_loss": test_loss},
        "unseen_prompt_generations": generations,
        "limitations": [
            "OASST1 is a community conversation dataset, not a professionally curated Hindi corpus.",
            "Only reviewed, non-deleted, non-synthetic, non-flagged same-language conversation paths are included.",
            "A 10M model trained on this small corpus is an experiment, not a production assistant.",
            "Generated responses are qualitative probes; validation/test loss and blind human review remain necessary.",
            "The repository tokenizer is reused; the experiment trains model weights from random initialization.",
        ],
    }
    (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="sample_output/bilingual_10m")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    run(args.output, args.steps)
