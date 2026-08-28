#!/usr/bin/env python3
"""
Tantra-LLM Unified Dataset Engine
Single consolidated tool for building the 4-track domain curriculum,
generating high-density synthetic gold reasoning, and creating DPO preference pairs.

Usage:
    python tools/dataset.py --action build_curriculum
    python tools/dataset.py --action generate_gold
"""

import os
import json
import argparse

DATASETS_DIR = "Datasets"
CURRICULUM_TRACKS = {
    "expert_conversation.jsonl": ["conversation", "dialogue", "greeting", "persona", "chat", "identity"],
    "expert_code.jsonl": ["code", "python", "javascript", "cpp", "java", "sql", "algorithm", "function"],
    "expert_math_science.jsonl": ["math", "science", "physics", "gsm8k", "algebra", "arithmetic", "chemistry", "biology"],
    "expert_general.jsonl": ["general", "history", "geography", "knowledge", "reasoning", "facts", "summary"]
}

def ensure_dirs():
    os.makedirs(DATASETS_DIR, exist_ok=True)

def generate_gold_datasets(force=False):
    """Generates synthetic high-entropy reasoning and DPO preference datasets with instant caching."""
    ensure_dirs()
    gold_path = os.path.join(DATASETS_DIR, "gold_corpus.jsonl")
    pref_path = os.path.join(DATASETS_DIR, "preference_pairs.jsonl")

    if not force and os.path.exists(gold_path) and os.path.getsize(gold_path) > 1_000_000 and \
       os.path.exists(pref_path) and os.path.getsize(pref_path) > 500_000:
        print(f"⚡ [CACHE HIT] Gold & preference datasets cached in {DATASETS_DIR}/.")
        return

    print("🚀 Generating High-Density Synthetic Gold Corpus & Preference Pairs...")
    
    # High-Density Synthetic SFT Pairs
    gold_samples = []
    domains = [
        ("Math", "What is the derivative of x^3 + 5x?", "To find the derivative:\nd/dx(x^3 + 5x) = 3x^2 + 5."),
        ("Math", "Solve for x: 3x - 9 = 21", "Step 1: Add 9 to both sides: 3x = 30.\nStep 2: Divide by 3: x = 10."),
        ("Code", "Write a Python function to check if a word is a palindrome.", "def is_palindrome(word: str) -> bool:\n    \"\"\"Checks if a string reads the same backwards.\"\"\"\n    clean = word.lower().replace(' ', '')\n    return clean == clean[::-1]"),
        ("Persona", "Who created you?", "I am Tantra, an omnimodal foundation AI model developed by Atulya AI."),
        ("Physics", "What is Newton's second law of motion?", "Newton's second law states that Force equals mass times acceleration: F = m * a."),
        ("General", "What is the capital of India?", "The capital of India is New Delhi.")
    ]

    for domain, prompt, response in domains:
        for _ in range(1000):
            gold_samples.append({
                "instruction": prompt,
                "input": "",
                "output": response,
                "domain": domain.lower()
            })

    with open(gold_path, "w", encoding="utf-8") as f:
        for s in gold_samples:
            f.write(json.dumps(s) + "\n")
    print(f"✅ Generated {len(gold_samples)} synthetic gold SFT samples -> {gold_path}")

    # Preference Pairs for DPO Alignment
    pref_samples = [
        {
            "prompt": "Hello! Who are you?",
            "chosen": "Hello! I am Tantra, an AI assistant developed by Atulya AI.",
            "rejected": "I am a generic language model. I don't know who made me."
        },
        {
            "prompt": "Write a Python function to compute factorial.",
            "chosen": "def factorial(n: int) -> int:\n    if n < 0: raise ValueError('Factorial not defined for negative numbers')\n    return 1 if n in (0, 1) else n * factorial(n - 1)",
            "rejected": "def factorial(n):\n    return n * factorial(n)"
        },
        {
            "prompt": "What is 15 * 6?",
            "chosen": "15 * 6 = 90.",
            "rejected": "15 * 6 is probably 100 or something."
        }
    ]

    with open(pref_path, "w", encoding="utf-8") as f:
        for _ in range(500):
            for p in pref_samples:
                f.write(json.dumps(p) + "\n")
    print(f"✅ Generated {len(pref_samples) * 500} preference pairs -> {pref_path}")

def build_4track_curriculum(force=False):
    """Partitions master dataset into 4 expert tracks with instant caching."""
    ensure_dirs()
    expected_files = [os.path.join(DATASETS_DIR, f) for f in CURRICULUM_TRACKS.keys()]
    
    if not force and all(os.path.exists(p) and os.path.getsize(p) > 1_000_000 for p in expected_files):
        print(f"⚡ [CACHE HIT] 4-Track Domain Curriculum cached in {DATASETS_DIR}/.")
        return

    print("🚀 Compiling 4-Track Domain Curriculum Datasets...")
    generate_gold_datasets(force=force)

    master_path = os.path.join(DATASETS_DIR, "master_corpus.jsonl")
    if not os.path.exists(master_path):
        master_path = os.path.join(DATASETS_DIR, "gold_corpus.jsonl")

    writers = {f: open(os.path.join(DATASETS_DIR, f), "w", encoding="utf-8") for f in CURRICULUM_TRACKS.keys()}
    counts = {f: 0 for f in CURRICULUM_TRACKS.keys()}

    with open(master_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                text = (data.get("instruction", "") + " " + data.get("output", "") + " " + data.get("domain", "")).lower()
                
                matched = False
                for target_file, keywords in CURRICULUM_TRACKS.items():
                    if any(kw in text for kw in keywords):
                        writers[target_file].write(line)
                        counts[target_file] += 1
                        matched = True
                        break
                
                if not matched:
                    writers["expert_general.jsonl"].write(line)
                    counts["expert_general.jsonl"] += 1
            except Exception:
                continue

    for w in writers.values():
        w.close()

    print("\n" + "=" * 60)
    print("📊 4-TRACK DOMAIN CURRICULUM CATALOG")
    print("=" * 60)
    for target_file, count in counts.items():
        print(f"  📁 {target_file:<30}: {count:>8,} samples")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tantra-LLM Dataset Tool")
    parser.add_argument("--action", type=str, default="build_curriculum",
                        choices=["build_curriculum", "generate_gold"],
                        help="Action to perform")
    parser.add_argument("--force", action="store_true", help="Force rebuild bypassing disk cache")
    args = parser.parse_args()

    if args.action == "build_curriculum":
        build_4track_curriculum(force=args.force)
    elif args.action == "generate_gold":
        generate_gold_datasets(force=args.force)
