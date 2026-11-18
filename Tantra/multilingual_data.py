"""Conservative multilingual data intake. Imported rows remain pending until human approval."""
from __future__ import annotations
import argparse, csv, hashlib, json, re
from collections import Counter
from pathlib import Path

LANGUAGES = {"en", "hi", "hinglish"}
SOURCE_FIELDS = ("id", "revision", "license", "url")

def rows(path):
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: yield line_no, json.loads(line)
        except json.JSONDecodeError as exc: raise ValueError(f"invalid JSON at {path}:{line_no}") from exc

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def script_issue(language, user, assistant):
    letters = [c for c in user + " " + assistant if c.isalpha()]
    if not letters: return "no_letters"
    devanagari = sum("\u0900" <= c <= "\u097f" for c in letters) / len(letters)
    latin = sum(c.isascii() and c.isalpha() for c in letters) / len(letters)
    if language == "hi" and devanagari < .45: return "hindi_script_mismatch_check_manually"
    if language == "en" and latin < .65: return "english_script_mismatch_check_manually"
    if language == "hinglish" and (latin < .35 or devanagari > .65): return "hinglish_script_mismatch_check_manually"
    return None

def build_review_queue(input_jsonl, output_dir, *, expert, source, domain="conversation"):
    if not isinstance(expert, str) or not expert.strip(): raise ValueError("expert must be a Tantra catalog ID")
    if any(not isinstance(source.get(k), str) or not source[k].strip() for k in SOURCE_FIELDS): raise ValueError("source requires id, revision, license and url")
    if source["license"].strip().casefold() in {"unknown", "n/a", "none"}: raise ValueError("source license must be identified")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()): raise ValueError("review queue output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    accepted, quarantine, seen = [], [], set()
    counts = Counter()
    for line_no, raw in rows(input_jsonl):
        if not isinstance(raw, dict):
            quarantine.append({"line": line_no, "reason": "row_not_object"}); continue
        required = ("id", "family_id", "language", "user", "assistant")
        missing = [k for k in required if not isinstance(raw.get(k), str) or not raw[k].strip()]
        if missing:
            quarantine.append({"line": line_no, "reason": "missing_or_empty:" + ",".join(missing)}); continue
        lang, user, answer = raw["language"].strip().lower(), raw["user"].strip(), raw["assistant"].strip()
        problems = []
        if lang not in LANGUAGES: problems.append("unsupported_language")
        if len(user) < 2 or len(answer) < 2: problems.append("too_short")
        if len(user) > 12000 or len(answer) > 12000: problems.append("message_too_long")
        if re.search(r"(?:sample|placeholder)\s+\d+\s+of\s+\d+", user + " " + answer, re.I): problems.append("placeholder_content")
        if lang in LANGUAGES and (issue := script_issue(lang, user, answer)): problems.append(issue)
        key = digest({"language": lang, "user": user, "assistant": answer})
        if key in seen: problems.append("exact_duplicate_content")
        if problems:
            quarantine.append({"line": line_no, "reason": ";".join(problems), "raw": raw}); continue
        seen.add(key)
        rid = digest({"source": source["id"], "revision": source["revision"], "record": raw["id"], "family": raw["family_id"], "content": key})
        accepted.append({
            "record_id": rid, "schema_version": 1, "expert": raw.get("expert", expert), "domain": raw.get("domain", domain), "language": lang,
            "family_id": raw["family_id"].strip(), "template_id": raw.get("template_id", f"source:{source['id']}"),
            "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
            "source": {"id": source["id"], "revision": source["revision"], "license": source["license"],
                       "kind": source.get("kind", "public_dataset_candidate"), "url": source["url"], "original_record_id": raw["id"]},
            "review": {"status": "pending", "reviewer": None, "reason": "Verify answer, language, safety, attribution and rights", "fluent_language_review": False},
        })
        counts[lang] += 1
    for name, values in (("candidates.jsonl", accepted), ("quarantine.jsonl", quarantine)):
        (out / name).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")
    summary = {"schema_version": 1, "source": {**{k: source[k] for k in SOURCE_FIELDS}, "kind": source.get("kind", "public_dataset_candidate")}, "expert": expert, "domain": domain,
               "input_rows": len(accepted)+len(quarantine), "pending_review_rows": len(accepted),
               "quarantined_rows": len(quarantine), "by_language": dict(sorted(counts.items())),
               "training_ready": False, "policy": "No imported row is training-eligible until explicit human approval."}
    (out / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return summary

def approve_review_queue(queue_dir, decisions_jsonl, incoming_dir, *, domain="conversation"):
    """Publish explicitly approved rows to incoming/<domain>/<expert>.jsonl."""
    queue = Path(queue_dir)
    candidates = [v for _, v in rows(queue / "candidates.jsonl")]
    by_id = {v["record_id"]: v for v in candidates}
    decisions = {}
    for line_no, decision in rows(decisions_jsonl):
        if not isinstance(decision, dict) or decision.get("record_id") not in by_id:
            raise ValueError(f"unknown review record at line {line_no}")
        rid = decision["record_id"]
        if rid in decisions:
            raise ValueError(f"duplicate review decision: {rid}")
        if decision.get("decision") not in {"approve", "reject"}:
            raise ValueError(f"decision must be approve/reject at line {line_no}")
        if not isinstance(decision.get("reviewer"), str) or not decision["reviewer"].strip():
            raise ValueError("reviewer is required")
        if not isinstance(decision.get("note"), str) or not decision["note"].strip():
            raise ValueError("review note is required")
        decisions[rid] = decision
    if set(decisions) != set(by_id):
        raise ValueError(f"review decisions missing for {len(set(by_id)-set(decisions))} candidates")

    groups = {}
    rejected = 0
    for rid, decision in decisions.items():
        row = dict(by_id[rid])
        if decision["decision"] == "reject":
            rejected += 1
            continue
        fluent = decision.get("fluent_language_review") is True
        if row["language"] in {"hi", "hinglish"} and not fluent:
            raise ValueError(f"fluent_language_review=true required for {row['language']} record")
        row["review"] = {
            "status": "reviewed", "reviewer": decision["reviewer"].strip(),
            "rationale": decision["note"].strip(), "fluent_language_review": fluent,
        }
        target_domain = row.pop("domain", domain)
        target_expert = row["expert"]
        if not isinstance(target_domain, str) or not target_domain.strip():
            raise ValueError(f"missing workbench domain for {rid}")
        groups.setdefault((target_domain.strip(), target_expert), []).append(row)

    output = Path(incoming_dir)
    destinations = {(d, e): output / d / f"{e}.jsonl" for d, e in groups}
    for destination in destinations.values():
        if destination.exists():
            raise ValueError(f"refusing to overwrite existing workbench input: {destination}")
    for key, records in groups.items():
        destination = destinations[key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in records),
            encoding="utf-8",
        )
    report = {
        "approved": sum(len(v) for v in groups.values()), "rejected": rejected,
        "published_paths": [str(v) for v in destinations.values()],
        "training_ready": False,
        "next_step": "Run python -m Tantra workbench_cli prepare; inspect family leakage, split hashes and language coverage.",
    }
    (queue / "review_result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command", required=True)
    intake=sub.add_parser("intake"); intake.add_argument("--input",required=True); intake.add_argument("--out",required=True); intake.add_argument("--expert",required=True); intake.add_argument("--domain",default="conversation")
    for key in SOURCE_FIELDS: intake.add_argument("--source-"+key,required=True)
    intake.add_argument("--source-kind",default="public_dataset_candidate")
    approve=sub.add_parser("approve"); approve.add_argument("--queue",required=True); approve.add_argument("--decisions",required=True); approve.add_argument("--incoming",default="Datasets/workbench/incoming"); approve.add_argument("--domain",default="conversation")
    a=p.parse_args(argv)
    if a.command=="intake":
        src={k:getattr(a,"source_"+k) for k in SOURCE_FIELDS}
        src["kind"]=a.source_kind
        result=build_review_queue(a.input,a.out,expert=a.expert,source=src,domain=a.domain)
    else: result=approve_review_queue(a.queue,a.decisions,a.incoming,domain=a.domain)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
