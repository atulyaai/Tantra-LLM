"""Qwen3.5 environment and multilingual dataset preflight.

This command does not train, modify model weights, or read held-out test rows
for model selection. Use --check-model for an explicit checkpoint load/generate
smoke test after reviewing the hardware report.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

from Tantra.qwen35_finetune import (
    DEFAULT_REVISION,
    MODEL_ID,
    load_prepared_manifest,
    load_manifest_test_rows,
)


def environment_report() -> dict:
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "model_id": MODEL_ID,
        "model_revision": DEFAULT_REVISION,
        "packages": {},
        "cuda_available": False,
        "gpu": None,
        "gpu_total_vram_gib": None,
        "system_ram_gib": None,
    }
    try:
        import importlib.metadata as metadata
        for package in ("torch", "transformers", "peft", "accelerate", "safetensors"):
            try:
                report["packages"][package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                report["packages"][package] = None
    except Exception as exc:
        report["package_check_error"] = str(exc)

    try:
        import psutil
        report["system_ram_gib"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        report["system_ram_gib"] = "psutil not installed"

    try:
        import torch
        report["torch_runtime"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = props.name
            report["gpu_total_vram_gib"] = round(props.total_memory / (1024 ** 3), 2)
            report["gpu_free_vram_gib"] = round(torch.cuda.mem_get_info(0)[0] / (1024 ** 3), 2)
            report["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except ImportError:
        report["torch_runtime_error"] = "PyTorch is not installed"
    except Exception as exc:
        report["torch_runtime_error"] = str(exc)
    return report


def dataset_report(manifest_path: str, experts: list[str]) -> dict:
    train, validation, manifest_hash, split_hashes = load_prepared_manifest(
        manifest_path, experts, allow_undersized=False
    )
    # Verify the sealed test artifacts and their checksums without using their
    # contents for training or model selection.
    test_rows, test_manifest_hash, test_hashes = load_manifest_test_rows(manifest_path, experts)
    def summarize(rows):
        languages = Counter(str(row["language"]).strip().lower() for row in rows)
        families = {str(row.get("split_group", row["family_id"])) for row in rows}
        return {
            "rows": len(rows),
            "unique_families": len(families),
            "language_rows": dict(sorted(languages.items())),
        }
    return {
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": manifest_hash,
        "test_manifest_sha256": test_manifest_hash,
        "selected_experts": experts,
        "train": summarize(train),
        "validation": summarize(validation),
        "test": summarize(test_rows),
        "verified_split_sha256": split_hashes,
        "verified_test_sha256": test_hashes,
        "data_ready": True,
        "note": "Manifest data_ready status and hashes verified; language quality still requires fluent review.",
    }


def model_smoke(revision: str, allow_cpu: bool) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -e '.[finetune]'") from exc
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("No CUDA GPU detected. Re-run with --allow-cpu-model-check for a slow CPU-only load test.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "नमस्ते। Reply briefly."}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    if isinstance(prompt, dict):
        input_ids = prompt["input_ids"].to(device)
    else:
        input_ids = prompt.to(device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(generated[0, input_ids.shape[-1]:], skip_special_tokens=True)
    if not answer.strip():
        raise RuntimeError("Checkpoint loaded but smoke generation returned an empty answer")
    return {
        "checkpoint_loaded": True,
        "generation_nonempty": True,
        "device": str(device),
        "dtype": str(dtype),
        "smoke_prompt": "नमस्ते। Reply briefly.",
        "smoke_output": answer,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None, help="Prepared workbench manifest.json")
    parser.add_argument("--experts", default=None, help="Comma-separated expert IDs")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Immutable 40-character Hub commit SHA")
    parser.add_argument("--check-model", action="store_true", help="Download/load checkpoint and generate a short smoke response")
    parser.add_argument("--allow-cpu-model-check", action="store_true", help="Permit slow CPU-only checkpoint load")
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision.lower()):
        parser.error("--revision must be a full immutable 40-character Hub commit SHA")
    report = {"environment": environment_report(), "model_revision": args.revision}
    if args.manifest or args.experts:
        if not args.manifest or not args.experts:
            parser.error("--manifest and --experts must be supplied together")
        selected = [name.strip() for name in args.experts.split(",") if name.strip()]
        if not selected:
            parser.error("At least one expert ID is required")
        report["dataset"] = dataset_report(args.manifest, selected)
    if args.check_model:
        report["model_smoke"] = model_smoke(args.revision, args.allow_cpu_model_check)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
