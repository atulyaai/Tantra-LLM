"""
tools/compile_master_dataset.py — Cleans, deduplicates, and compiles all domain datasets
into a single, balanced, high-density master curriculum: `Datasets/master_curriculum/master_train.jsonl`.
"""
import os
import sys
import json
import hashlib
from typing import Dict, Any, Optional

def normalize_sample(data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Normalizes any JSONL sample format to standard {system, user, assistant}."""
    system = data.get("system", "You are Tantra, a helpful, precise AI assistant created by Atulya AI.")
    user = data.get("user", "")
    assistant = data.get("assistant", "")

    # Format 2: OpenAI 'messages' array
    if "messages" in data and isinstance(data["messages"], list):
        for m in data["messages"]:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                user = content
            elif role == "assistant":
                assistant = content

    # Format 3: 'instruction' / 'input' / 'output'
    if not user and "instruction" in data:
        user = data["instruction"]
        if data.get("input"):
            user += f"\nInput: {data['input']}"
        assistant = data.get("output", "")

    # Format 4: 'prompt' / 'response'
    if not user and "prompt" in data:
        user = data["prompt"]
        assistant = data.get("response", "")

    # Format 5: raw 'text' string (split if ChatML tags present)
    if not user and "text" in data:
        raw = data["text"]
        if "<|user|>" in raw and "<|assistant|>" in raw:
            parts = raw.split("<|user|>")
            if len(parts) > 1:
                sub = parts[1].split("<|assistant|>")
                user = sub[0].strip()
                if len(sub) > 1:
                    assistant = sub[1].replace("</s>", "").replace("<|end|>", "").strip()
        else:
            user = "Complete the following knowledge text:"
            assistant = raw.strip()

    # Validation
    if not user.strip() or not assistant.strip():
        return None

    # Length guard (ignore massive or tiny single-word noise)
    if len(assistant.strip()) < 5:
        return None

    return {
        "system": system.strip(),
        "user": user.strip(),
        "assistant": assistant.strip()
    }

def compile_master_curriculum():
    os.makedirs("Datasets/master_curriculum", exist_ok=True)
    out_file = "Datasets/master_curriculum/master_train.jsonl"
    manifest_file = "Datasets/master_curriculum/manifest.json"

    # Prioritized source directories
    sources = [
        ("Synthetic Textbooks", "Datasets/synthetic_textbooks/textbooks_10k.jsonl"),
        ("Tool Calling", "Datasets/tool_calling/tool_calling.jsonl"),
        ("Curated SFT", "Datasets/curated_sft/curated_sft.jsonl"),
        ("Code", "Datasets/code/code.jsonl"),
        ("Math", "Datasets/math/math.jsonl"),
        ("Multilingual", "Datasets/multilingual/multilingual.jsonl"),
        ("Science", "Datasets/science/science.jsonl"),
        ("Safety", "Datasets/safety/safety.jsonl"),
        ("Instructions", "Datasets/instructions/instructions.jsonl"),
        ("Creative Writing", "Datasets/creative_writing/creative_writing.jsonl"),
    ]

    seen_hashes = set()
    domain_stats = {}
    master_samples = []

    print("=" * 65)
    print("      TANTRA MASTER DATASET CLEANUP & COMPILATION")
    print("=" * 65)

    for domain_name, path in sources:
        if not os.path.exists(path):
            continue
        
        valid_in_domain = 0
        dupes_in_domain = 0

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_data = json.loads(line)
                    norm = normalize_sample(raw_data)
                    if not norm:
                        continue

                    # Deduplicate via MD5
                    h = hashlib.md5((norm["user"] + "||" + norm["assistant"]).encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        dupes_in_domain += 1
                        continue

                    seen_hashes.add(h)
                    master_samples.append(norm)
                    valid_in_domain += 1

                except Exception:
                    continue

        domain_stats[domain_name] = {
            "valid_samples": valid_in_domain,
            "deduplicated_removed": dupes_in_domain,
            "source_file": path
        }
        print(f"[OK] {domain_name:22}: {valid_in_domain:,} samples added (Deduplicated: {dupes_in_domain})")

    # Write out unified master dataset
    with open(out_file, "w", encoding="utf-8") as f:
        for s in master_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    total_mb = os.path.getsize(out_file) / (1024 * 1024)

    manifest = {
        "dataset_name": "Tantra Unified Master Curriculum",
        "version": "2.0-Omni",
        "total_samples": len(master_samples),
        "total_size_mb": round(total_mb, 2),
        "format": "ChatML (system, user, assistant)",
        "domains": domain_stats
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 65)
    print(f"MASTER DATASET CREATED: {out_file}")
    print(f"TOTAL CLEAN SAMPLES   : {len(master_samples):,}")
    print(f"TOTAL FILE SIZE       : {total_mb:.2f} MB")
    print(f"MANIFEST GENERATED    : {manifest_file}")
    print("=" * 65)

if __name__ == "__main__":
    compile_master_curriculum()
