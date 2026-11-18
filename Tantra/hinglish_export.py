"""Export the pinned synthetic Hinglish source into Tantra's pending-review queue format."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

DEFAULT_REPOSITORY = "DSMJ910/hinglish-instruct-10k"
DEFAULT_REVISION = "39686808306eb6d898c8f7e86dfa93549e4ef23b"
EXPERT_BY_CATEGORY = {"casual": "smalltalk", "customer_support": "qa", "question": "qa", "sentiment": "classification"}
DOMAIN_BY_EXPERT = {"smalltalk": "conversation", "qa": "tasks", "classification": "tasks"}
HINDI_ROMAN_MARKERS = re.compile(r"\b(yaar|bhai|hai|hain|kya|ka|ki|ke|ko|mera|mujhe|aap|nahi|nahin|kar|chahiye|raha|rahi|ho|tha|the|na|matlab|arre|haan)\b", re.I)


def export_hinglish_rows(dataset, output_jsonl, *, split="train", repository=DEFAULT_REPOSITORY, revision=DEFAULT_REVISION):
    if split != "train":
        raise ValueError("only source train split may be exported; reserve any source evaluation split")
    output=Path(output_jsonl)
    if output.exists():
        raise ValueError(f"refusing to overwrite existing export: {output}")
    output.parent.mkdir(parents=True,exist_ok=True)
    counts={}
    written=0
    with output.open("w",encoding="utf-8") as handle:
        for index,row in enumerate(dataset):
            if str(row.get("script","roman")).lower()!="roman":
                continue
            user=row.get("user_message")
            assistant=row.get("assistant_response")
            scenario=row.get("scenario")
            persona=row.get("persona")
            category=row.get("category","general")
            if not all(isinstance(v,str) and v.strip() for v in (user,assistant,scenario,persona)):
                continue
            if not HINDI_ROMAN_MARKERS.search(user+" "+assistant):
                continue
            # Rows sharing a scenario/persona are variants; keep them in one split family.
            family=f"hinglish-scenario:{scenario.strip().casefold()}:{persona.strip().casefold()}"
            expert = EXPERT_BY_CATEGORY.get(str(category).strip().lower())
            if expert is None:
                continue
            record={
                "id":f"hinglish-synthetic-{index:08d}",
                "family_id":family,
                "language":"hinglish",
                "expert":expert,
                "domain":DOMAIN_BY_EXPERT[expert],
                "template_id":f"synthetic-hinglish:{category}",
                "user":user.strip(),
                "assistant":assistant.strip(),
            }
            handle.write(json.dumps(record,ensure_ascii=False)+"\n")
            counts[category]=counts.get(category,0)+1
            written+=1
    if not written:
        output.unlink(missing_ok=True)
        raise ValueError("no usable Roman-script Hinglish rows found")
    manifest={
        "schema_version":1,"repository":repository,"revision":revision,"license":"Apache-2.0",
        "source_split":split,"rows":written,"by_category":dict(sorted(counts.items())),
        "output":str(output),"output_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "synthetic":True,"training_eligible":False,
        "limitations":["100% Roman script; no Devanagari code-mixing in this source",
                      "synthetic GPT-4o-mini generations; review for fabricated anecdotes and factual errors",
                      "scenario/persona variants share a family to reduce leakage"],
    }
    Path(str(output)+".manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True)
    parser.add_argument("--repository",default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision",default=DEFAULT_REVISION)
    parser.add_argument("--split",default="train")
    args=parser.parse_args(argv)
    if len(args.revision)!=40 or any(c not in "0123456789abcdef" for c in args.revision.lower()):
        parser.error("--revision must be a full immutable 40-character commit SHA")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the finetune extra: pip install -e '.[finetune]'") from exc
    dataset=load_dataset(args.repository,split=args.split,revision=args.revision)
    result=export_hinglish_rows(dataset,args.out,split=args.split,repository=args.repository,revision=args.revision)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
