"""
tools/build_staged_curriculum.py — Builds a chronological 4-phase staged curriculum
designed for ultra-fast CPU learning with low token count and high information density:
Stage 1: Conversation & Persona -> Stage 2: General Knowledge -> Stage 3: Mathematics -> Stage 4: Coding.
"""
import os
import json
import random

SYSTEM_PROMPT = "You are Tantra, a helpful, concise AI assistant built by Atulya AI."

# ── Phase 1: Conversation, Persona & Bilingual Greetings ──────────────────────
CONVERSATION_SAMPLES = [
    ("Hello! Who are you?",
     "Hello! I am **Tantra**, a fast, privacy-focused on-device AI assistant developed by **Atulya AI**."),
    
    ("How are you today?",
     "I am functioning optimally! How can I assist you with your work or learning today?"),

    ("नमस्ते! आप कौन हैं?",
     "नमस्ते! मैं **तन्त्र** (Tantra) हूँ — अतुल्य एआई (Atulya AI) द्वारा निर्मित एक तेज़ और सुरक्षित एआई सहायक।"),

    ("Can you help me learn programming and math?",
     "Yes, absolutely! I can explain programming concepts, solve math equations step-by-step, and write clean code."),

    ("What languages do you speak?",
     "I primarily speak English and Hindi (हिन्दी), and can understand multiple programming languages including Python, C++, and JavaScript."),

    ("धन्यवाद!",
     "आपका स्वागत है! यदि आपको किसी अन्य सहायता की आवश्यकता हो, तो अवश्य बताएं।"),

    ("What makes you different from cloud AI?",
     "I run 100% locally on your device with complete data privacy, zero API costs, and fast CPU inference.")
]

# ── Phase 2: General Knowledge & Scientific Axioms ─────────────────────────────
GENERAL_SAMPLES = [
    ("Why is the sky blue?",
     "The sky is blue due to **Rayleigh scattering**: Earth's atmosphere scatters shorter blue wavelengths of sunlight much more than longer red wavelengths."),

    ("What is photosynthesis?",
     "Photosynthesis is the process where plants convert sunlight, water, and CO2 into glucose and oxygen:\n$$6\\text{CO}_2 + 6\\text{H}_2\\text{O} + \\text{light} \\rightarrow \\text{C}_6\\text{H}_{12}\\text{O}_6 + 6\\text{O}_2$$"),

    ("State Newton's Second Law of Motion.",
     "Newton's Second Law states that force equals mass times acceleration: **$F = ma$**."),

    ("What are the three states of matter?",
     "The three primary classical states are **Solid** (fixed shape/volume), **Liquid** (fixed volume, adaptable shape), and **Gas** (expands to fill container)."),

    ("पृथ्वी की सूर्य से दूरी कितनी है?",
     "पृथ्वी की सूर्य से औसत दूरी लगभग **14.96 करोड़ किलोमीटर** (1 Astronomical Unit / AU) है।"),

    ("What is the speed of light in a vacuum?",
     "The speed of light $c$ is exactly **$299,792,458\\text{ m/s}$** (approximately $3 \\times 10^8\\text{ m/s}$).")
]

# ── Phase 3: Mathematical Reasoning & Proofs ──────────────────────────────────
MATH_SAMPLES = [
    ("What is 15% of 480?",
     "$$\\text{Value} = 0.15 \\times 480 = 72$$\n**Answer:** 72"),

    ("Find the derivative of f(x) = 5x^3 - 4x + 9.",
     "Apply the power rule $\\frac{d}{dx}[x^n] = n x^{n-1}$:\n$$f'(x) = 15x^2 - 4$$"),

    ("Solve for x: 3x + 15 = 45.",
     "1. Subtract 15: $3x = 30$\n2. Divide by 3: $x = 10$\n**Answer:** $x = 10$"),

    ("What is the area of a circle with radius r = 7 cm?",
     "$$\\text{Area} = \\pi r^2 = \\frac{22}{7} \\times 7^2 = 154\\text{ cm}^2$$"),

    ("Calculate 384 multiplied by 25.",
     "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"384 * 25\"}}\n</tool_call>\n<tool_result>\n9600\n</tool_result>\n\n384 multiplied by 25 is **9,600**."),

    ("What is the Pythagorean theorem?",
     "In a right triangle with legs $a, b$ and hypotenuse $c$:\n$$a^2 + b^2 = c^2$$")
]

# ── Phase 4: Coding & Algorithms (Crisp & Verified) ───────────────────────────
CODING_SAMPLES = [
    ("Write a Python function to reverse a string.",
     "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```"),

    ("How do I find the largest element in a Python list?",
     "```python\ndef find_max(nums: list[int]) -> int:\n    return max(nums) if nums else None\n```"),

    ("Write a function to check if a number is even.",
     "```python\ndef is_even(n: int) -> bool:\n    return n % 2 == 0\n```"),

    ("Implement linear search in Python.",
     "```python\ndef linear_search(arr: list, target) -> int:\n    for i, val in enumerate(arr):\n        if val == target:\n            return i\n    return -1\n```"),

    ("How to count word frequency in Python?",
     "```python\nfrom collections import Counter\n\ndef word_count(text: str) -> dict:\n    return dict(Counter(text.split()))\n```"),

    ("Run Python to compute factorial of 6.",
     "<tool_call>\n{\"name\": \"python_executor\", \"arguments\": {\"code\": \"import math\\nprint(math.factorial(6))\"}}\n</tool_call>\n<tool_result>\n720\n</tool_result>\n\nThe factorial of 6 is **720**.")
]

def build_staged_dataset():
    print("=" * 65)
    print("     BUILDING CHRONOLOGICAL 4-STAGE STAGED CURRICULUM")
    print("=" * 65)

    stages = [
        ("Phase 1: Conversation & Persona", CONVERSATION_SAMPLES, 2000),
        ("Phase 2: General Knowledge & Science", GENERAL_SAMPLES, 2000),
        ("Phase 3: Mathematics & Calculus", MATH_SAMPLES, 2000),
        ("Phase 4: Coding & Algorithms", CODING_SAMPLES, 2000),
    ]

    all_staged = []
    random.seed(42)

    for stage_name, samples, count in stages:
        stage_pool = []
        while len(stage_pool) < count:
            for u, a in samples:
                stage_pool.append({
                    "system": SYSTEM_PROMPT,
                    "user": u,
                    "assistant": a
                })
        random.shuffle(stage_pool)
        stage_pool = stage_pool[:count]
        all_staged.extend(stage_pool)
        print(f"[OK] {stage_name:38}: {len(stage_pool):,} high-density samples")

    out_paths = [
        "Datasets/master_train.jsonl",
        "Datasets/master_curriculum/master_train.jsonl"
    ]

    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fp:
            for s in all_staged:
                fp.write(json.dumps(s, ensure_ascii=False) + "\n")

    total_mb = os.path.getsize(out_paths[0]) / (1024 * 1024)

    manifest = {
        "dataset_name": "Tantra 4-Stage Chronological Staged Curriculum",
        "total_samples": len(all_staged),
        "size_mb": round(total_mb, 2),
        "stages": [
            {"stage": 1, "name": "Conversation & Persona", "samples": 2000},
            {"stage": 2, "name": "General Knowledge & Science", "samples": 2000},
            {"stage": 3, "name": "Mathematics & Calculus", "samples": 2000},
            {"stage": 4, "name": "Coding & Algorithms", "samples": 2000}
        ]
    }

    with open("Datasets/manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    print("=" * 65)
    print(f"STAGED CURRICULUM READY: {len(all_staged):,} Total Samples ({total_mb:.2f} MB)")
    print(f"AVERAGE TOKENS PER ITEM : ~60-120 tokens (Ultra-fast CPU training)")
    print("=" * 65)

if __name__ == "__main__":
    build_staged_dataset()
