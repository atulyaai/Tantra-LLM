"""
tools/consolidate_datasets_flat.py — Consolidates and deduplicates all datasets
into clean, flat files directly under `Datasets/` and removes all subfolders.
"""
import os
import shutil
import json
import hashlib

def normalize(data):
    system = data.get("system", "You are Tantra, a helpful, precise AI assistant created by Atulya AI.")
    user = data.get("user", "")
    assistant = data.get("assistant", "")

    if "messages" in data and isinstance(data["messages"], list):
        for m in data["messages"]:
            if m.get("role") == "system":
                system = m.get("content", "")
            elif m.get("role") == "user":
                user = m.get("content", "")
            elif m.get("role") == "assistant":
                assistant = m.get("content", "")

    if not user and "instruction" in data:
        user = data["instruction"]
        inp = data.get("input")
        if inp:
            user += f"\nInput: {inp}"
        assistant = data.get("output", "")

    if not user and "prompt" in data:
        user = data["prompt"]
        assistant = data.get("response", "")

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

    if not user.strip() or not assistant.strip() or len(assistant.strip()) < 5:
        return None

    return {"system": system.strip(), "user": user.strip(), "assistant": assistant.strip()}

def main():
    print("=" * 65)
    print("       TANTRA FLAT DATASET CONSOLIDATION & DEDUPLICATION")
    print("=" * 65)

    sources = []
    for root, dirs, files in os.walk("Datasets"):
        for f in files:
            if f.endswith(".jsonl"):
                sources.append(os.path.join(root, f))

    print(f"Scanning {len(sources)} source files across Datasets/...")

    seen_hashes = set()
    master_samples = []

    priority_order = sorted(sources, key=lambda p: (
        0 if "synthetic_textbooks" in p else
        1 if "tool_calling" in p else
        2 if "master_curriculum" in p else
        3 if "curated_sft" in p else 4
    ))

    for src in priority_order:
        print(f"Processing: {src}")
        with open(src, "r", encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    norm = normalize(raw)
                    if not norm:
                        continue
                    h = hashlib.md5((norm["user"] + "||" + norm["assistant"]).encode("utf-8")).hexdigest()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        master_samples.append(norm)
                except Exception:
                    continue

    print(f"\n[OK] Found {len(master_samples):,} unique clean samples after hash-deduplication.")

    # 2. Stage to temp directory
    temp_dir = "temp_flat_datasets"
    os.makedirs(temp_dir, exist_ok=True)
    temp_master = os.path.join(temp_dir, "master_train.jsonl")

    with open(temp_master, "w", encoding="utf-8") as fp:
        for s in master_samples:
            fp.write(json.dumps(s, ensure_ascii=False) + "\n")

    total_mb = os.path.getsize(temp_master) / (1024 * 1024)

    temp_manifest = os.path.join(temp_dir, "manifest.json")
    manifest = {
        "dataset_name": "Tantra Unified Flat Master Dataset",
        "version": "2.0-Omni-Flat",
        "total_unique_samples": len(master_samples),
        "total_size_mb": round(total_mb, 2),
        "format": "ChatML (system, user, assistant)",
        "deduplicated": True,
        "files": ["master_train.jsonl", "manifest.json"]
    }
    with open(temp_manifest, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    # 3. Clean up all subfolders inside Datasets/
    print("\nRemoving subfolders and obsolete shards inside Datasets/...")
    for item in os.listdir("Datasets"):
        item_path = os.path.join("Datasets", item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path, ignore_errors=True)
        elif os.path.isfile(item_path):
            try:
                os.remove(item_path)
            except Exception:
                pass

    # 4. Move consolidated flat files directly into Datasets/
    shutil.move(temp_master, os.path.join("Datasets", "master_train.jsonl"))
    shutil.move(temp_manifest, os.path.join("Datasets", "manifest.json"))
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("=" * 65)
    print("DATASET CONSOLIDATION COMPLETE!")
    print(f"Direct Datasets/ directory contents:")
    for f in os.listdir("Datasets"):
        fp = os.path.join("Datasets", f)
        size = os.path.getsize(fp) / (1024 * 1024)
        print(f"  - Datasets/{f} ({size:.2f} MB)")
    print("=" * 65)

if __name__ == "__main__":
    main()
