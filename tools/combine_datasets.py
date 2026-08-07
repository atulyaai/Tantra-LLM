"""
Combine identity + safety datasets into a single training file, then create
a LoRA training script using peft + transformers + trl.
"""
import json
import os
import random

random.seed(42)

DATASETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Datasets')
MERGED_PATH = os.path.join(DATASETS_DIR, 'tantra_identity_safety_mixed.jsonl')

sources = [
    os.path.join(DATASETS_DIR, 'tantra_identity_safety_expanded.jsonl'),
    os.path.join(DATASETS_DIR, 'tantra_identity_safety_large.jsonl'),
]

merged = []
for src in sources:
    if not os.path.exists(src):
        print(f"  Skipping {os.path.basename(src)} (not found)")
        continue
    with open(src, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                merged.append(json.loads(line))
    print(f"  Loaded {os.path.basename(src)}: {len(merged)} total so far")

random.shuffle(merged)
print(f"\nTotal merged conversations: {len(merged):,}")

with open(MERGED_PATH, 'w', encoding='utf-8') as f:
    for conv in merged:
        f.write(json.dumps(conv, ensure_ascii=False) + '\n')

size_mb = os.path.getsize(MERGED_PATH) / (1024 * 1024)
print(f"Written -> {MERGED_PATH}")
print(f"File size: {size_mb:.1f} MB")
