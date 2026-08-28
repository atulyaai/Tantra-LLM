#!/usr/bin/env python3
"""
tools/build_curriculum_datasets.py — Splits & curates the 417K sample corpus into focused domain experts:
  1. Datasets/expert_conversation.jsonl (Greetings, Persona, Turn-Taking, Polite Assistant)
  2. Datasets/expert_code.jsonl (Python, SQL, JavaScript, Algorithms)
  3. Datasets/expert_math_science.jsonl (GSM8K Math, Linear Equations, Newton's Laws, Science)
  4. Datasets/expert_general.jsonl (World Knowledge, MMLU, Multi-subject Reasoning)
"""
import os
import glob
import json
import random

def build_curriculum():
    import sys
    force = "--force" in sys.argv
    expected_files = [
        "Datasets/expert_conversation.jsonl",
        "Datasets/expert_code.jsonl",
        "Datasets/expert_math_science.jsonl",
        "Datasets/expert_general.jsonl"
    ]
    if not force and all(os.path.exists(p) and os.path.getsize(p) > 1024 * 1024 for p in expected_files):
        print("[CURRICULUM CACHE] Verified existing expert curriculum tracks in Datasets/. Skipping re-creation for instant startup.")
        return

    os.makedirs("Datasets", exist_ok=True)
    
    conv_items = []
    code_items = []
    math_sci_items = []
    general_items = []
    
    # 1. Base Greetings & Persona
    greetings = [
        ("hi", "Hello! How can I help you today?"),
        ("hello", "Hi there! I am Tantra, your AI assistant created by Atulya AI. What would you like to work on?"),
        ("hey", "Hey! Feel free to ask me anything about coding, math, science, or general questions."),
        ("how are you", "I'm doing well, thank you for asking! How can I assist you with your projects today?"),
        ("who are you", "I am **Tantra**, an intelligent AI assistant developed and trained by **Atulya AI** using the NeuroCore architecture."),
        ("what can you do", "I can help you write code (Python, SQL, JS), solve step-by-step math problems, explain science and physics concepts, and engage in natural conversations."),
        ("thank you", "You're very welcome! Let me know if you need help with anything else."),
    ]
    for _ in range(50):
        for u, a in greetings:
            conv_items.append({
                "system": "You are Tantra, a helpful, precise, and polite AI assistant created by Atulya AI.",
                "user": u,
                "assistant": a
            })
            
    # 2. Ingest all JSONL files from Datasets/
    jsonl_files = glob.glob("Datasets/**/*.jsonl", recursive=True)
    for path in jsonl_files:
        if "preference" in path or "expert_" in path:
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    
                    user = data.get("user") or data.get("instruction") or data.get("prompt")
                    asst = data.get("assistant") or data.get("response") or data.get("output")
                    if not user or not asst:
                        continue
                    
                    item = {
                        "system": data.get("system", "You are Tantra, a helpful, precise, and polite AI assistant created by Atulya AI."),
                        "user": str(user).strip(),
                        "assistant": str(asst).strip()
                    }
                    
                    user_l = user.lower()
                    if any(k in user_l for k in ["def ", "python", "code", "function", "sql", "select ", "javascript", "import ", "class "]):
                        code_items.append(item)
                    elif any(k in user_l for k in ["solve", "equation", "math", "+", "-", "*", "/", "newton", "force", "physics", "chemistry", "biology"]):
                        math_sci_items.append(item)
                    elif any(k in user_l for k in ["hi", "hello", "who are you", "how are you", "help", "thank", "name"]):
                        conv_items.append(item)
                    else:
                        general_items.append(item)
        except Exception:
            continue
            
    # Shuffle and save each curriculum track
    tracks = [
        ("Datasets/expert_conversation.jsonl", conv_items),
        ("Datasets/expert_code.jsonl", code_items),
        ("Datasets/expert_math_science.jsonl", math_sci_items),
        ("Datasets/expert_general.jsonl", general_items)
    ]
    
    random.seed(42)
    print("=" * 80)
    print("[CURRICULUM] TANTRA DOMAIN EXPERT CURRICULUM DATASETS:")
    print("=" * 80)
    for out_path, items in tracks:
        random.shuffle(items)
        with open(out_path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"  [TRACK] {os.path.basename(out_path):35s}: {len(items):,} high-quality samples")
    print("=" * 80)

if __name__ == "__main__":
    build_curriculum()
