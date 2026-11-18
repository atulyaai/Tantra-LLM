"""Repeatable multilingual evaluation harness for pretrained Tantra candidates.

This records model outputs and systems metrics. Semantic quality remains a
human-review task; the report preserves candidate provenance for reproducibility.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = []
    seen = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_no}") from exc
        required = ("id", "language", "category", "prompt")
        if any(not isinstance(item.get(key), str) or not item[key].strip() for key in required):
            raise ValueError(f"line {line_no} requires non-empty id/language/category/prompt")
        if item["id"] in seen:
            raise ValueError(f"duplicate benchmark case id: {item['id']}")
        seen.add(item["id"])
        cases.append(item)
    if not cases:
        raise ValueError("benchmark suite is empty")
    return cases


def run_benchmark(
    gateway,
    cases: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    max_new_tokens: int = 128,
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = []
    by_language = defaultdict(list)
    by_category = defaultdict(list)
    for case in cases:
        start = time.perf_counter()
        output = gateway.generate(
            [{"role": "user", "content": case["prompt"]}],
            max_new_tokens=max_new_tokens,
        )
        elapsed = (time.perf_counter() - start) * 1000
        output = output if isinstance(output, str) else str(output)
        identity_ok = ("tantra" in output.casefold()) if case.get("identity_check") else None
        row = {
            **case,
            "output": output,
            "latency_ms": round(elapsed, 3),
            "output_chars": len(output),
            "output_words": len(output.split()),
            "automatic_identity_check": identity_ok,
            "human_review_status": "pending",
        }
        rows.append(row)
        by_language[case["language"]].append(row)
        by_category[case["category"]].append(row)

    def aggregate(groups):
        result = {}
        for key, items in sorted(groups.items()):
            checked = [item for item in items if item["automatic_identity_check"] is not None]
            result[key] = {
                "cases": len(items),
                "mean_latency_ms": round(statistics.mean(item["latency_ms"] for item in items), 3),
                "nonempty_rate": round(sum(bool(item["output"].strip()) for item in items) / len(items), 4),
                "identity_pass_rate": (
                    round(sum(item["automatic_identity_check"] is True for item in checked) / len(checked), 4)
                    if checked else None
                ),
            }
        return result

    spec = getattr(gateway, "spec", None)
    summary = {
        "schema_version": 2,
        "model": getattr(spec, "model_id", "unknown"),
        "revision": getattr(spec, "revision", "unknown"),
        "candidate": candidate or {"kind": "base", "adapter_path": None, "residual_candidate_path": None},
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "cases": len(rows),
        "mean_latency_ms": round(statistics.mean(item["latency_ms"] for item in rows), 3),
        "nonempty_rate": round(sum(bool(item["output"].strip()) for item in rows) / len(rows), 4),
        "identity_checks_passed": sum(
            item["automatic_identity_check"] is True
            for item in rows if item["automatic_identity_check"] is not None
        ),
        "identity_checks_total": sum(item["automatic_identity_check"] is not None for item in rows),
        "by_language": aggregate(by_language),
        "by_category": aggregate(by_category),
        "semantic_quality_status": "not_scored_human_review_required",
        "results": rows,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m Tantra.pretrained_benchmark")
    parser.add_argument("--registry", default="configs/models/registry.json")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--residual-candidate")
    parser.add_argument("--source-commit", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--suite", default="Tests/fixtures/pretrained_benchmark.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(argv)

    if args.adapter and args.residual_candidate:
        parser.error("--adapter and --residual-candidate cannot be combined in one candidate run")

    from Tantra.model_registry import ModelRegistry
    from Tantra.model_gateway import ModelGateway

    spec = ModelRegistry.from_file(args.registry).get(args.model_id)
    gateway = ModelGateway(
        spec, args.checkpoint, device=args.device, adapter_path=args.adapter
    ).load()
    if args.residual_candidate:
        from Tantra.pretrained_expansion import load_residual_candidate
        load_residual_candidate(gateway._model, args.residual_candidate)

    candidate_kind = "residual" if args.residual_candidate else "lora" if args.adapter else "base"
    candidate = {
        "kind": candidate_kind,
        "checkpoint_path": args.checkpoint,
        "adapter_path": args.adapter,
        "residual_candidate_path": args.residual_candidate,
        "source_commit": args.source_commit,
    }
    try:
        report = run_benchmark(
            gateway,
            load_cases(args.suite),
            args.output,
            max_new_tokens=args.max_new_tokens,
            candidate=candidate,
        )
    finally:
        gateway.unload()
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
