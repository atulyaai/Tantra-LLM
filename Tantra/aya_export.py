"""Export only the pinned Aya training split into Tantra's pending-review JSONL format."""
from __future__ import annotations
import argparse
import hashlib
import heapq
import json
from pathlib import Path

LANGUAGE_CODES = {"eng": "en", "hin": "hi"}
DEFAULT_REPOSITORY = "CohereLabs/aya_dataset"
DEFAULT_REVISION = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"


def export_aya_training_rows(dataset, output_jsonl, *, split="train", language_codes=None,
                             repository=DEFAULT_REPOSITORY, revision=DEFAULT_REVISION,
                             max_rows_per_language=2000):
    if split != "train":
        raise ValueError("only Aya train split may be exported; keep source test split reserved")
    mapping = language_codes or LANGUAGE_CODES
    output = Path(output_jsonl)
    if output.exists():
        raise ValueError(f"refusing to overwrite existing export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if max_rows_per_language is not None and max_rows_per_language < 1:
        raise ValueError("max_rows_per_language must be positive or None")
    heaps = {language: [] for language in mapping.values()}
    seen = {language: 0 for language in mapping.values()}
    for index, row in enumerate(dataset):
        code = str(row.get("language_code", "")).strip().lower()
        if code not in mapping:
            continue
        user = row.get("inputs")
        assistant = row.get("targets")
        if not isinstance(user, str) or not user.strip() or not isinstance(assistant, str) or not assistant.strip():
            continue
        language = mapping[code]
        seen[language] += 1
        priority = int(hashlib.sha256(f"{revision}:{index}".encode()).hexdigest(), 16)
        item = (-priority, -index, user.strip(), assistant.strip(), code, row.get("annotation_type", "human_annotation"))
        heap = heaps[language]
        if max_rows_per_language is None or len(heap) < max_rows_per_language:
            heapq.heappush(heap, item)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, item)

    selected = []
    for language, heap in heaps.items():
        selected.extend((language, item) for item in heap)
    if not selected:
        raise ValueError("no supported English/Hindi Aya training rows were found")
    selected.sort(key=lambda pair: (-pair[1][0], -pair[1][1]))
    counts = {}
    output = Path(output_jsonl)
    if output.exists():
        raise ValueError(f"refusing to overwrite existing export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for language, item in selected:
            _, negative_index, user, assistant, code, annotation_type = item
            index = -negative_index
            record = {
                "id": f"aya-train-{index:08d}",
                "family_id": f"aya-train-family-{index:08d}",
                "language": language,
                "expert": "lang_en" if language == "en" else "lang_hi",
                "domain": "multilingual",
                "template_id": f"aya:{code}:{annotation_type}",
                "user": user,
                "assistant": assistant,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\\n")
            counts[language] = counts.get(language, 0) + 1
    manifest = {
        "schema_version": 1, "repository": repository, "revision": revision,
        "license": "Apache-2.0", "source_split": split, "rows": sum(counts.values()), "candidate_rows_seen": seen,
        "max_rows_per_language": max_rows_per_language,
        "by_language": dict(sorted(counts.items())),
        "output": str(output), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "training_eligible": False,
        "note": "Rows still require Tantra human review; Hinglish is not supplied by this exporter.",
    }
    Path(str(output) + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows-per-language", type=int, default=2000)
    args = parser.parse_args(argv)
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision.lower()):
        parser.error("--revision must be a full immutable 40-character commit SHA")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the finetune extra: pip install -e '.[finetune]'") from exc
    dataset = load_dataset(args.repository, split=args.split, revision=args.revision)
    result = export_aya_training_rows(
        dataset, args.out, split=args.split, repository=args.repository, revision=args.revision,
        max_rows_per_language=args.max_rows_per_language
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
