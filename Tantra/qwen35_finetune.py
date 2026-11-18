"""Text-only Qwen3.5-0.8B LoRA SFT pipeline for Tantra.

Input JSONL follows docs/DATASET_SPEC.md. This script intentionally trains an
adapter only; it never modifies the downloaded base checkpoint or Model/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_REVISION = "5c8a1b97ddef11f79b47ab9d07bf82b9117413f6"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_reviewed_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    families: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                raise ValueError(f"{path}:{line_no}: expected a messages array")
            if not row.get("family_id"):
                raise ValueError(f"{path}:{line_no}: family_id is required")
            if not str(row.get("language", "")).strip():
                raise ValueError(f"{path}:{line_no}: language is required for per-language reporting")
            review = row.get("review") or {}
            if review.get("status") not in {"reviewed", "verified_auto"}:
                raise ValueError(f"{path}:{line_no}: review status must be reviewed or verified_auto")
            source = row.get("source") or {}
            if not all(str(source.get(k, "")).strip() for k in ("id", "revision", "license")):
                raise ValueError(f"{path}:{line_no}: source id, revision and license are required")
            if str(source["license"]).strip().lower() in {"unknown", "none", "n/a"}:
                raise ValueError(f"{path}:{line_no}: unknown source license is not trainable")
            messages = row["messages"]
            if len(messages) < 2:
                raise ValueError(f"{path}:{line_no}: at least one user and one assistant message are required")
            for message in messages:
                if (not isinstance(message, dict)
                    or message.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(message.get("content"), str)
                    or not message["content"].strip()):
                    raise ValueError(f"{path}:{line_no}: invalid message; expected system/user/assistant and non-empty content")
            if not any(m["role"] == "assistant" for m in messages):
                raise ValueError(f"{path}:{line_no}: no assistant response")
            family = str(row["family_id"])
            families.add(family)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no usable rows")
    return rows, sha256_file(path)


def check_family_disjoint(train_rows: list[dict], validation_rows: list[dict]) -> None:
    train_families = {str(row.get("split_group", row["family_id"])) for row in train_rows}
    validation_families = {str(row.get("split_group", row["family_id"])) for row in validation_rows}
    overlap = train_families & validation_families
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"train/validation family leakage ({len(overlap)} families): {examples}")


def load_prepared_manifest(manifest_path: str | Path, experts: list[str]):
    """Load and checksum-verify selected workbench train/val splits.

    Test split files are checksum-verified but never parsed or used for selection.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("experts"), dict):
        raise ValueError("Prepared manifest has no experts mapping")
    if not experts:
        raise ValueError("Select at least one expert from the prepared manifest")
    train_rows, validation_rows = [], []
    split_hashes = {"train": {}, "val": {}, "test": {}}
    seen_train, seen_validation = set(), set()
    for expert in experts:
        if expert not in manifest["experts"]:
            raise ValueError(f"Unknown expert in prepared manifest: {expert}")
        record = manifest["experts"][expert]
        files = record.get("files") or {}
        for split in ("train", "val", "test"):
            item = files.get(split)
            if not item:
                raise ValueError(f"Expert {expert!r} has no frozen {split} split")
            path = manifest_path.parent / item["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = sha256_file(path)
            if digest.lower() != str(item.get("sha256", "")).lower():
                raise ValueError(f"Prepared {split} checksum mismatch: {path}")
            split_hashes[split][expert] = digest
            if split == "test":
                continue
            if path.stat().st_size == 0:
                continue
            rows, _ = read_reviewed_jsonl(path)
            for row in rows:
                group = str(row.get("split_group", row["family_id"]))
                expected_split = "train" if split == "train" else "val"
                if row.get("split") and row["split"] != expected_split:
                    raise ValueError(f"Row split label mismatch in {path}: expected {expected_split}")
                key = str(row.get("record_id") or hashlib.sha256(
                    json.dumps(row["messages"], ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest())
                if split == "train":
                    if key not in seen_train:
                        train_rows.append(row)
                        seen_train.add(key)
                else:
                    if key not in seen_validation:
                        validation_rows.append(row)
                        seen_validation.add(key)
    if not train_rows or not validation_rows:
        raise ValueError("Selected experts do not provide non-empty combined train and validation splits")
    check_family_disjoint(train_rows, validation_rows)
    return train_rows, validation_rows, sha256_file(manifest_path), split_hashes


def _ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in value]


def encode_rows(rows: list[dict], tokenizer, max_length: int) -> list[dict[str, list[int]]]:
    """Turn every assistant turn into a context-conditioned SFT example.

    Prompt tokens are masked from loss. Only the corresponding assistant turn,
    including its end marker, contributes to the objective.
    """
    examples: list[dict[str, list[int]]] = []
    for row in rows:
        messages = row["messages"]
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            context = messages[:index]
            if not context or not any(m["role"] == "user" for m in context):
                continue
            prompt_ids = _ids(tokenizer.apply_chat_template(
                context, tokenize=True, add_generation_prompt=True
            ))
            full_ids = _ids(tokenizer.apply_chat_template(
                messages[:index + 1], tokenize=True, add_generation_prompt=False
            ))
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError("Tokenizer chat template does not preserve prompt prefix; refusing unsafe label masking")
            if len(full_ids) > max_length:
                overflow = len(full_ids) - max_length
                # Keep the entire supervised answer; trim only the oldest context.
                if len(full_ids) - len(prompt_ids) >= max_length:
                    full_ids = full_ids[-max_length:]
                    prompt_len = 0
                else:
                    full_ids = full_ids[overflow:]
                    prompt_len = max(0, len(prompt_ids) - overflow)
            else:
                prompt_len = len(prompt_ids)
            if prompt_len >= len(full_ids):
                continue
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            examples.append({"input_ids": full_ids, "labels": labels})
    if not examples:
        raise ValueError("No assistant-turn examples could be encoded from the dataset")
    return examples


def collate(examples: list[dict], pad_id: int, torch):
    width = max(len(x["input_ids"]) for x in examples)
    ids, masks, labels = [], [], []
    for example in examples:
        padding = width - len(example["input_ids"])
        ids.append(example["input_ids"] + [pad_id] * padding)
        masks.append([1] * len(example["input_ids"]) + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def evaluate(model, examples, tokenizer, batch_size, device, torch) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = collate(examples[start:start + batch_size], tokenizer.pad_token_id, torch)
            batch = {key: value.to(device) for key, value in batch.items()}
            count = int((batch["labels"] != -100).sum().item())
            if not count:
                continue
            loss = model(**batch).loss
            total_loss += float(loss.detach().float().cpu()) * count
            total_tokens += count
    model.train()
    if not total_tokens:
        raise ValueError("Validation set has no supervised assistant tokens")
    return total_loss / total_tokens


def train(args) -> dict:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install Qwen3.5 fine-tuning dependencies: pip install -e '.[finetune]'") from exc

    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision.lower()):
        raise ValueError("--revision must be a full immutable 40-character Hub commit SHA")
    if min(args.max_steps, args.max_length, args.batch_size, args.grad_accum, args.rank) < 1:
        raise ValueError("steps, max_length, batch_size, grad_accum and rank must be positive")
    if args.learning_rate <= 0 or args.eval_steps < 1 or args.log_steps < 1:
        raise ValueError("learning_rate, eval_steps and log_steps must be positive")
    if args.alpha < 1 or not 0 <= args.dropout < 1:
        raise ValueError("LoRA alpha must be positive and dropout must be in [0, 1)")
    if not 0 <= args.warmup_ratio < 1 or args.max_grad_norm <= 0:
        raise ValueError("warmup_ratio must be in [0, 1) and max_grad_norm must be positive")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {output}")

    if args.manifest:
        if args.train_jsonl or args.validation_jsonl or not args.experts:
            raise ValueError("With --manifest, provide --experts and omit explicit train/validation files")
        train_rows, validation_rows, manifest_hash, split_hashes = load_prepared_manifest(
            args.manifest, [name.strip() for name in args.experts.split(",") if name.strip()]
        )
        train_hash = None
        validation_hash = None
    else:
        if not args.train_jsonl or not args.validation_jsonl:
            raise ValueError("Provide either --manifest with --experts or both explicit split files")
        train_rows, train_hash = read_reviewed_jsonl(args.train_jsonl)
        validation_rows, validation_hash = read_reviewed_jsonl(args.validation_jsonl)
        check_family_disjoint(train_rows, validation_rows)
        manifest_hash, split_hashes = None, {}

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32
    )
    print(f"Loading {args.model_id}@{args.revision} on {device} ({dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.revision, torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
    )
    model = get_peft_model(model, lora)
    model.to(device)
    model.print_trainable_parameters()

    train_examples = encode_rows(train_rows, tokenizer, args.max_length)
    validation_examples = encode_rows(validation_rows, tokenizer, args.max_length)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    warmup = max(1, int(args.max_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, max(0.0, (args.max_steps - step) / max(1, args.max_steps - warmup)))
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and dtype == torch.float16))
    output.mkdir(parents=True, exist_ok=True)
    best_dir = output / "best_adapter"
    best_val = math.inf
    best_step = 0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()

    while global_step < args.max_steps:
        order = list(range(len(train_examples)))
        random.Random(seed + global_step).shuffle(order)
        batches = [
            order[start:start + args.batch_size]
            for start in range(0, len(order), args.batch_size)
        ]
        # Include a short final accumulation group so small datasets still make progress.
        for group_start in range(0, len(batches), args.grad_accum):
            group = batches[group_start:group_start + args.grad_accum]
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for indices in group:
                selected = [train_examples[i] for i in indices]
                batch = collate(selected, tokenizer.pad_token_id, torch)
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                    raw_loss = model(**batch).loss
                    loss = raw_loss / len(group)
                scaler.scale(loss).backward()
                group_loss += float(raw_loss.detach().float().cpu())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), args.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            if global_step % args.log_steps == 0:
                print(f"step={global_step} train_loss={group_loss / len(group):.4f}", flush=True)
            if global_step % args.eval_steps == 0 or global_step == args.max_steps:
                val_loss = evaluate(model, validation_examples, tokenizer, args.batch_size, device, torch)
                print(f"step={global_step} validation_loss={val_loss:.5f}", flush=True)
                if val_loss < best_val:
                    best_val, best_step = val_loss, global_step
                    if best_dir.exists():
                        import shutil
                        shutil.rmtree(best_dir)
                    model.save_pretrained(best_dir, safe_serialization=True)
                    tokenizer.save_pretrained(best_dir)
                model.train()
            if global_step >= args.max_steps:
                break

    if not best_dir.is_dir():
        raise RuntimeError("Training ended without a saved validation checkpoint")
    from collections import Counter
    train_language_counts = dict(sorted(Counter(str(row["language"]) for row in train_rows).items()))
    validation_language_counts = dict(sorted(Counter(str(row["language"]) for row in validation_rows).items()))
    metadata = {
        "schema_version": 1,
        "candidate_status": "pilot_unvalidated",
        "model_id": args.model_id,
        "model_revision": args.revision,
        "training_method": "supervised_finetuning_lora",
        "task": "text_only_causal_lm",
        "train_file": str(Path(args.train_jsonl).resolve()) if args.train_jsonl else None,
        "train_sha256": train_hash,
        "validation_file": str(Path(args.validation_jsonl).resolve()) if args.validation_jsonl else None,
        "validation_sha256": validation_hash,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_language_counts": train_language_counts,
        "validation_language_counts": validation_language_counts,
        "train_supervised_turns": len(train_examples),
        "validation_supervised_turns": len(validation_examples),
        "max_steps": args.max_steps,
        "best_step": best_step,
        "best_validation_loss": best_val,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "learning_rate": args.learning_rate,
        "lora_rank": args.rank,
        "lora_alpha": args.alpha,
        "lora_target_modules": "all-linear",
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype),
        "test_data_used_for_training_or_selection": False,
    }
    (output / "tantra_qwen35_sft.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--validation-jsonl", default=None)
    parser.add_argument("--manifest", default=None, help="Prepared workbench manifest.json; verifies split hashes")
    parser.add_argument("--experts", default=None, help="Comma-separated expert IDs to combine from the manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--log-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
