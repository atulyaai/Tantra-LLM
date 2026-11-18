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
HINDI_ROMAN_MARKERS = re.compile(r"\b(yaar|bhai|hai|hain|kya|ka|ki|ke|ko|mera|mujhe|aap|nahi|nahin|kar|chahiye|raha|rahi|ho|tha|na|matlab|arre|haan)\b", re.I)


def export_hinglish_rows(dataset, output_jsonl, *, split="train", repository=DEFAULT_REPOSITORY,
                         revision=DEFAULT_REVISION, max_rows_per_expert=2000):
    if split != "train":
        raise ValueError("only source train split may be exported; reserve any source evaluation split")
    output=Path(output_jsonl)
    if output.exists():
        raise ValueError(f"refusing to overwrite existing export: {output}")
    output.parent.mkdir(parents=True,exist_ok=True)
    if max_rows_per_expert is not None and max_rows_per_expert < 1:
        raise ValueError("max_rows_per_expert must be positive or None")
    candidates={}
    candidates_seen={}
    for index,row in enumerate(dataset):
        if str(row.get("script","roman")).lower()!="roman":
            continue
        user=row.get("user_message")
        assistant=row.get("assistant_response")
        scenario=row.get("scenario")
        persona=row.get("persona")
        category=str(row.get("category","general")).strip().lower()
        if not all(isinstance(v,str) and v.strip() for v in (user,assistant,scenario,persona)):
            continue
        if not HINDI_ROMAN_MARKERS.search(user+" "+assistant):
            continue
        expert=EXPERT_BY_CATEGORY.get(category)
        if expert is None:
            continue
        family=f"hinglish-scenario:{scenario.strip().casefold()}:{persona.strip().casefold()}"
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
        candidates.setdefault(expert,[]).append((family,record))
        candidates_seen[expert]=candidates_seen.get(expert,0)+1
    if not candidates:
        raise ValueError("no usable Roman-script Hinglish rows found")

    selected=[]
    family_counts={}
    for expert,items in sorted(candidates.items()):
        families={}
        for family,record in items:
            families.setdefault(family,[]).append(record)
        ordered=sorted(families, key=lambda family: hashlib.sha256(
            f"{revision}:{expert}:{family}".encode()).hexdigest())
        chosen=[]
        for family in ordered:
            if max_rows_per_expert is not None and len(chosen)>=max_rows_per_expert:
                break
            chosen.extend(families[family])
        selected.extend(chosen)
        family_counts[expert]=len({row["family_id"] for row in chosen})

    output=Path(output_jsonl)
    if output.exists():
        raise ValueError(f"refusing to overwrite existing export: {output}")
    output.parent.mkdir(parents=True,exist_ok=True)
    selected.sort(key=lambda row: row["id"])
    with output.open("w",encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record,ensure_ascii=False)+"\n")
    counts={}
    for record in selected:
        category=record["template_id"].split(":",1)[-1]
        counts[category]=counts.get(category,0)+1
    written=len(selected)
    manifest={
        "schema_version":1,"repository":repository,"revision":revision,"license":"Apache-2.0",
        "source_split":split,"rows":written,"by_category":dict(sorted(counts.items())),
        "by_expert":{key:sum(1 for row in selected if row["expert"]==key) for key in sorted(candidates)},
        "families_by_expert":family_counts,"candidate_rows_seen":candidates_seen,
        "max_rows_per_expert":max_rows_per_expert,
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
    parser.add_argument("--max-rows-per-expert",type=int,default=2000)
    args=parser.parse_args(argv)
    if len(args.revision)!=40 or any(c not in "0123456789abcdef" for c in args.revision.lower()):
        parser.error("--revision must be a full immutable 40-character commit SHA")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the finetune extra: pip install -e '.[finetune]'") from exc
    dataset=load_dataset(args.repository,split=args.split,revision=args.revision)
    result=export_hinglish_rows(dataset,args.out,split=args.split,repository=args.repository,revision=args.revision,max_rows_per_expert=args.max_rows_per_expert)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
