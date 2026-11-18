"""Generate a fixed held-out Qwen3.5 evaluation set for human review.

Run once without --adapter-dir for the untouched Base checkpoint and again with
a saved adapter. Keep suite, decoding options, hardware and revision identical.
This script records outputs; it does not assign quality scores or promote models.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from Tantra.qwen35_finetune import DEFAULT_REVISION, MODEL_ID, read_reviewed_jsonl


def evaluate(args):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the Qwen3.5 fine-tuning extra: pip install -e '.[finetune]'") from exc

    if len(args.revision) != 40:
        raise ValueError("--revision must be a full immutable 40-character Hub commit SHA")
    rows, dataset_hash = read_reviewed_jsonl(args.test_jsonl)
    output = Path(args.output_jsonl)
    if output.exists():
        raise ValueError(f"Refusing to overwrite evaluation output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.revision, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
    model.to(device)
    model.eval()

    language_counts = Counter()
    latencies = []
    total_generated = 0
    with output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            messages = row["messages"]
            target_index = next(
                (i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "assistant"),
                None,
            )
            if target_index is None:
                raise ValueError(f"test row {index} has no assistant reference")
            context = messages[:target_index]
            if not context or not any(m["role"] == "user" for m in context):
                raise ValueError(f"test row {index} has no user prompt")
            inputs = tokenizer.apply_chat_template(
                context, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            if isinstance(inputs, dict):
                input_ids = inputs["input_ids"].to(device)
            else:
                input_ids = inputs.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            new_ids = generated[0, input_ids.shape[-1]:]
            prediction = tokenizer.decode(new_ids, skip_special_tokens=True)
            generated_tokens = int(new_ids.numel())
            total_generated += generated_tokens
            latencies.append(elapsed)
            language = str(row["language"])
            language_counts[language] += 1
            record = {
                "case_id": str(row.get("id", index)),
                "family_id": str(row["family_id"]),
                "language": language,
                "prompt_messages": context,
                "reference": messages[target_index]["content"],
                "prediction": prediction,
                "latency_seconds": elapsed,
                "generated_tokens": generated_tokens,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index + 1}/{len(rows)}] {language} {elapsed:.2f}s {generated_tokens} tokens", flush=True)

    summary = {
        "model_id": args.model_id,
        "model_revision": args.revision,
        "adapter_dir": str(Path(args.adapter_dir).resolve()) if args.adapter_dir else None,
        "test_file": str(Path(args.test_jsonl).resolve()),
        "test_sha256": dataset_hash,
        "output_jsonl": str(output.resolve()),
        "cases": len(rows),
        "language_counts": dict(sorted(language_counts.items())),
        "mean_latency_seconds": sum(latencies) / len(latencies),
        "total_generated_tokens": total_generated,
        "device": str(device),
        "dtype": str(dtype),
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "human_quality_review": "required; not scored by this script",
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
