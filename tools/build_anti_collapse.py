"""Build a focused anti-collapse dataset for fixed prompt separation."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


OUT_DIR = Path("Download/anti_collapse")

SYSTEM = "You are Atulya. Answer the user's exact question clearly and briefly."

EXAMPLES = [
    ("Hi! How are you?", "I'm doing well. How can I help you today?", "chat"),
    ("Hello. How are you?", "I'm doing well. What would you like help with?", "chat"),
    ("Who are you?", "I am Atulya, a local AI assistant designed to help clearly and practically.", "identity"),
    ("What can you do?", "I can help with questions, explanations, writing, code, planning, and troubleshooting.", "identity"),
    ("What is gravity?", "Gravity is the force that pulls objects with mass toward each other.", "science"),
    ("Explain gravity simply.", "Gravity is the pull between objects that have mass. On Earth, it pulls things toward the ground.", "science"),
    ("What is machine learning?", "Machine learning is a way for computers to learn patterns from data and use those patterns to make predictions or decisions.", "science"),
    ("What is photosynthesis?", "Photosynthesis is how plants use sunlight, water, and carbon dioxide to make sugar for energy.", "science"),
    ("What is an atom?", "An atom is a tiny unit of matter made of protons, neutrons, and electrons.", "science"),
    ("What is a cell?", "A cell is the basic unit of life.", "science"),
    ("Who was Chanakya?", "Chanakya was an ancient Indian teacher, strategist, economist, and adviser associated with the Maurya Empire.", "history"),
    ("What was the Renaissance?", "The Renaissance was a period of renewed interest in art, learning, science, and classical ideas in Europe.", "history"),
    ("What was the Industrial Revolution?", "The Industrial Revolution was a shift from hand production to machine-based production.", "history"),
    ("What is democracy?", "Democracy is a system where people have a role in choosing leaders or laws.", "history"),
    ("Write a Python function.", "def add(a, b):\n    return a + b", "code"),
    ("Write a Python function to add two numbers.", "def add_numbers(a, b):\n    return a + b", "code"),
    ("What is a Python variable?", "A Python variable is a name that stores or points to a value.", "code"),
    ("What is a loop in Python?", "A loop repeats code. A for loop repeats over items, and a while loop repeats while a condition is true.", "code"),
    ("What is recursion?", "Recursion is when a function calls itself. It needs a base case so it stops.", "code"),
    ("How do I fix a syntax error?", "Read the error message, check the line it points to, and look for missing colons, quotes, brackets, or indentation.", "code"),
    ("Tell me something interesting.", "Octopuses have three hearts, and two of them help move blood through the gills.", "factual"),
    ("Tell me a science fact.", "Honey never spoils easily because it has very little water and naturally resists many microbes.", "factual"),
    ("How do I make a good decision?", "Define the goal, compare realistic options, check the risks, and choose the best next step.", "reasoning"),
    ("How do I learn a hard topic?", "Break it into small parts, learn the basics first, practice examples, and review mistakes.", "reasoning"),
    ("Why can loss improve while answers stay bad?", "Loss can improve when a model learns common token patterns before it learns how to answer each prompt correctly.", "reasoning"),
    ("I feel overwhelmed.", "Pause and choose one small next task. You do not have to solve everything at once.", "support"),
    ("I made a mistake.", "Mistakes happen. Fix what you can, learn from it, and take the next step.", "support"),
    ("I am nervous about my exam.", "That makes sense. Make a small plan, review the most important topics, and take short breaks.", "support"),
]

PROMPT_VARIANTS = [
    "{q}",
    "Answer clearly: {q}",
    "Explain simply: {q}",
    "Give a short answer: {q}",
    "Answer the exact question: {q}",
]


def build_rows(target: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    i = 0
    while len(rows) < target:
        q, a, cat = EXAMPLES[i % len(EXAMPLES)]
        variant = PROMPT_VARIANTS[(i // len(EXAMPLES)) % len(PROMPT_VARIANTS)]
        user = variant.format(q=q)
        rows.append({
            "system": SYSTEM,
            "user": user,
            "assistant": a,
            "category": f"anti_collapse_{cat}",
        })
        i += 1
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=5000)
    args = parser.parse_args()
    rows = build_rows(args.rows, args.seed)
    path = args.output_dir / f"anti_collapse_{len(rows)//1000}k.jsonl"
    write_jsonl(path, rows)
    print(path, len(rows), Counter(r["category"] for r in rows).most_common())


if __name__ == "__main__":
    main()
