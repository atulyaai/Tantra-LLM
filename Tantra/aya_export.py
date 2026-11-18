"""Export only the pinned Aya training split into Tantra's pending-review JSONL format."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

LANGUAGE_CODES = {"eng": "en", "hin": "hi"}
DEFAULT_REPOSITORY = "CohereForAI/aya_dataset"
DEFAULT_REVISION = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"


def export_aya_training_rows(dataset, output_jsonl, *, split="train", language_codes=None):
    if split != "train":
        raise ValueError("only Aya train split may be exported; keep source test split reserved")
    mapping = language_codes or LANGUAGE_CODES
    output = Path(output_jsonl)
    if output.exists():
        raise ValueError(f"refusing to overwrite existing export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            code = str(row.get("language_code", "")).strip().lower()
            if code not in mapping:
                continue
            user = row.get("inputs")
            assistant = row.get("targets")
            if not isinstance(user, str) or not user.strip() or not isinstance(assistant, str) or not assistant.strip():
                continue
            record = {
                "id": f"aya-train-{index:08d}",
                "family_id": f"aya-train-family-{index:08d}",
                "language": mapping[code],
                "template_id": f"aya:{code}:{row.get('annotation_type', 'human_annotation')}",
                "user": user.strip(),
                "assistant": assistant.strip(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[mapping[code]] = counts.get(mapping[code], 0) + 1
            written += 1
    if not written:
        output.unlink(missing_ok=True)
        raise ValueError("no supported English/Hindi Aya training rows were found")
    return {"split": split, "rows": written, "by_language": dict(sorted(counts.items())),
            "output": str(output), "training_eligible": False,
            "note": "Rows still require Tantra human review; Hinglish is not supplied by this exporter."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default="train")
    args = parser.parse_args(argv)
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision.lower()):
        parser.error("--revision must be a full immutable 40-character commit SHA")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the finetune extra: pip install -e '.[finetune]'") from exc
    dataset = load_dataset(args.repository, split=args.split, revision=args.revision)
    result = export_aya_training_rows(dataset, args.out, split=args.split)
    result.update({"repository": args.repository, "revision": args.revision, "license": "Apache-2.0"})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
