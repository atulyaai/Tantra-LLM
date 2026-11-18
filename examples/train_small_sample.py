"""Train a tiny, CPU-only Tantra model on a transparent toy corpus.

This is a mechanics smoke sample, NOT a useful general-purpose language model.
Run: python examples/train_small_sample.py --output sample_output/small_cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from Tantra import ModelConfig
from Tantra.tokenizer import TantraTokenizer
from Tantra.train import EncodedDataset, TantraModel, TrainArgs, _loop, eval_loss


TRAIN_TEXTS = [
    "The sun rises in the east.",
    "The moon appears at night.",
    "Water is important for life.",
    "Plants need sunlight and water.",
    "A small bird can fly.",
    "The sky looks blue on a clear day.",
    "We learn by reading and practice.",
    "A river flows toward the sea.",
    "The earth moves around the sun.",
    "Good questions help us understand.",
    "The wind moves the leaves.",
    "Books contain words and ideas.",
]


def train_sample(output: str | Path, steps: int = 35) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = TantraTokenizer.train(TRAIN_TEXTS, vocab_size=384, min_frequency=1)
    tokenizer.save(output / "tokenizer.json")
    rows = [{"text": text} for text in TRAIN_TEXTS]
    dataset = EncodedDataset(tokenizer, rows, max_len=48, assistant_only=False)
    if len(dataset) < 2:
        raise RuntimeError("Sample corpus did not produce enough trainable rows")

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        mlp_mult=2.0,
        max_seq_len=48,
        dropout=0.0,
        lora_rank=2,
    )
    model = TantraModel(cfg)
    params = list(model.base_parameters())
    initial_loss = eval_loss(model, dataset, batch_size=4, max_batches=20)
    final_train_loss = _loop(
        model,
        params,
        dataset,
        TrainArgs(
            steps=steps,
            batch_size=4,
            lr=0.025,
            warmup=0,
            weight_decay=0.0,
            max_len=48,
            log_every=steps,
            seed=2026,
            gradient_accumulation=1,
        ),
        "small-cpu-sample",
    )
    final_loss = eval_loss(model, dataset, batch_size=4, max_batches=20)
    checkpoint = output / "base.pt"
    model.save(checkpoint)
    reloaded = TantraModel.load(checkpoint, map_location="cpu").eval()
    model.eval()
    probe = torch.tensor([tokenizer.encode("The sun rises")], dtype=torch.long)
    with torch.no_grad():
        expected = model(probe)
        actual = reloaded(probe)
    reload_max_abs_error = float((expected - actual).abs().max())
    if not torch.isfinite(torch.tensor([initial_loss, final_train_loss, final_loss])).all():
        raise RuntimeError("Non-finite loss in sample training")
    if not final_loss < initial_loss:
        raise RuntimeError(
            f"Sample did not learn on its toy corpus: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )
    if reload_max_abs_error > 1e-6:
        raise RuntimeError(f"Checkpoint reload changed logits: max_abs_error={reload_max_abs_error}")

    report = {
        "sample_type": "toy-corpus CPU training smoke test; not a production model",
        "seed": 2026,
        "device": "cpu",
        "steps": steps,
        "train_examples": len(dataset),
        "parameters": model.num_params(),
        "vocab_size": tokenizer.vocab_size,
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "reported_loop_loss": final_train_loss,
        "checkpoint": checkpoint.name,
        "tokenizer": "tokenizer.json",
        "reload_max_abs_logit_error": reload_max_abs_error,
        "limitations": [
            "Tiny synthetic toy corpus; demonstrates training and checkpoint mechanics only",
            "Does not demonstrate language understanding, factual knowledge, Hindi quality, or generalization",
            "Not suitable for deployment or user conversations",
        ],
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="sample_output/small_cpu")
    parser.add_argument("--steps", type=int, default=35)
    args = parser.parse_args()
    print(json.dumps(train_sample(args.output, args.steps), indent=2))


if __name__ == "__main__":
    main()
