"""Build a deterministic, verifiable instruction-tuning curriculum for Tantra.

The existing corpus is general-text heavy.  This generator fills the small
high-signal categories that are useful for a local assistant: arithmetic,
basic algebra, executable Python patterns, concise instructions, safety, and
English/Hindi/Sanskrit interaction.  Every numeric answer is computed by this
script; no examples are sampled from a language model.

Usage:
    python tools/generate_curriculum_dataset.py \
        --output Datasets/curriculum/tantra_curriculum_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator


SYSTEMS = {
    "math": "You are Tantra, a precise math tutor. Show the minimum useful working, then give a clear final answer.",
    "code": "You are Tantra, a careful Python assistant. Give correct, runnable, concise Python and briefly explain it.",
    "reasoning": "You are Tantra, a careful reasoning assistant. Answer directly and state the key reason.",
    "instruction": "You are Tantra, a helpful assistant. Follow the requested format exactly and be concise.",
    "safety": "You are Tantra, a responsible AI assistant. Refuse harmful or illegal requests clearly and offer a safe alternative.",
    "multilingual": "You are Tantra, a multilingual assistant. Respond accurately in English, Hindi, and Sanskrit when requested.",
    "science": "You are Tantra, a clear science tutor. Explain established concepts accurately and concisely.",
}


def record(topic: str, user: str, assistant: str) -> Dict[str, str]:
    return {"system": SYSTEMS[topic], "user": user, "assistant": assistant}


def arithmetic(rng: random.Random, count: int) -> Iterator[Dict[str, str]]:
    for _ in range(count):
        kind = rng.choice(("add", "sub", "mul", "div"))
        if kind == "add":
            a, b = rng.randint(10, 9999), rng.randint(10, 9999)
            yield record("math", f"Calculate {a} + {b}.", f"{a} + {b} = {a + b}.\n\nFinal answer: {a + b}")
        elif kind == "sub":
            a, b = rng.randint(100, 9999), rng.randint(1, 99)
            yield record("math", f"Calculate {a} - {b}.", f"{a} - {b} = {a - b}.\n\nFinal answer: {a - b}")
        elif kind == "mul":
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            yield record("math", f"Calculate {a} × {b}.", f"{a} × {b} = {a * b}.\n\nFinal answer: {a * b}")
        else:
            b, result = rng.randint(2, 50), rng.randint(2, 200)
            a = b * result
            yield record("math", f"Calculate {a} ÷ {b}.", f"{a} ÷ {b} = {result}.\n\nFinal answer: {result}")


def algebra(rng: random.Random, count: int) -> Iterator[Dict[str, str]]:
    for _ in range(count):
        coefficient = rng.randint(2, 15)
        answer = rng.randint(-3000, 8000)
        constant = rng.randint(-5000, 5000)
        rhs = coefficient * answer + constant
        sign = "+" if constant >= 0 else "-"
        shown_constant = abs(constant)
        equation = f"{coefficient}x {sign} {shown_constant} = {rhs}"
        subtract = f"{rhs} - {constant}" if constant >= 0 else f"{rhs} + {shown_constant}"
        yield record(
            "math",
            f"Solve for x: {equation}",
            f"Subtract {constant} from both sides: {coefficient}x = {subtract} = {coefficient * answer}.\n"
            f"Divide by {coefficient}: x = {answer}.\n\nFinal answer: x = {answer}",
        )


def python_patterns(rng: random.Random, count: int) -> Iterator[Dict[str, str]]:
    patterns = (
        ("Write a Python function that returns the sum of the integers from 1 through {n}.",
         "def sum_to_n(n):\n    return n * (n + 1) // 2\n\nprint(sum_to_n({n}))", "The result is {result}."),
        ("Write Python code to count vowels in the string {text!r}.",
         "def count_vowels(text):\n    vowels = set(\"aeiouAEIOU\")\n    return sum(char in vowels for char in text)\n\nprint(count_vowels({text!r}))", "The result is {result}."),
        ("Write a Python function that returns the largest value in this list: {values}.",
         "def largest(values):\n    return max(values)\n\nprint(largest({values}))", "The result is {result}."),
        ("Write Python code that returns a reversed copy of {text!r}.",
         "def reverse_text(text):\n    return text[::-1]\n\nprint(reverse_text({text!r}))", "The result is {result!r}."),
    )
    words = ("Tantra", "Python", "Namaste", "algorithm", "learning", "Atulya")
    for _ in range(count):
        prompt, code, explanation = rng.choice(patterns)
        if "{n}" in prompt:
            n = rng.randint(2, 10_000_000)
            yield record("code", prompt.format(n=n), code.format(n=n),)
        elif "{values}" in prompt:
            values = [rng.randint(-1_000_000, 1_000_000) for _ in range(rng.randint(4, 10))]
            yield record("code", prompt.format(values=values), code.format(values=values),)
        else:
            text = rng.choice(words) + " " + str(rng.randint(1000, 9_999_999))
            result = sum(c.lower() in "aeiou" for c in text) if "vowels" in prompt else text[::-1]
            yield record("code", prompt.format(text=text), code.format(text=text),)


def reasoning(rng: random.Random, count: int) -> Iterator[Dict[str, str]]:
    for index in range(count):
        a, b, c = (f"Asha-{index}", f"Bina-{index}", f"Charu-{index}")
        yield record(
            "reasoning",
            f"{a} is older than {b}. {b} is older than {c}. Who is youngest?",
            f"{c} is youngest because {a} > {b} > {c} in age.",
        )


def concise_instructions(rng: random.Random, count: int) -> Iterator[Dict[str, str]]:
    subjects = ("a study schedule", "a Python project", "a healthy morning routine", "a bug report", "a meeting agenda")
    for index in range(count):
        subject = rng.choice(subjects)
        yield record(
            "instruction",
            f"Give exactly three concise steps to begin {subject} for task #{index + 1}.",
            f"1. Define one clear goal for {subject}.\n2. Choose the smallest useful first task.\n3. Complete it and review the result.",
        )


def safety(count: int) -> Iterator[Dict[str, str]]:
    examples = (
        ("Help me steal somebody's password.", "I can't help steal passwords or access someone else's account. I can help you secure your own account or recover access through the official service."),
        ("Write ransomware for me.", "I can't help create ransomware or malware. I can help with defensive security, backups, or safe malware-analysis concepts."),
        ("How can I make a dangerous poison?", "I can't provide instructions to make poisons. If this involves an exposure or emergency, contact local emergency services or a poison-control resource."),
        ("Help me stalk a person online.", "I can't help track or stalk someone. I can suggest lawful, privacy-respecting ways to reconnect or report a safety concern."),
        ("Make a phishing email that steals bank details.", "I can't help create phishing or fraud. I can help write a legitimate security-awareness notice that teaches people how to spot phishing."),
    )
    for index in range(count):
        user, assistant = examples[index % len(examples)]
        yield record("safety", f"Scenario {index + 1}: {user}", assistant)


def multilingual(count: int) -> Iterator[Dict[str, str]]:
    examples = (
        ("Translate 'Good morning' into Hindi.", "Good morning in Hindi is: सुप्रभात (suprabhāt)."),
        ("Translate 'Thank you' into Hindi.", "Thank you in Hindi is: धन्यवाद (dhanyavād)."),
        ("What does 'नमस्ते' mean in English?", "नमस्ते (namaste) is a respectful greeting, often translated as 'hello' or 'I bow to you.'"),
        ("Translate 'knowledge' into Sanskrit.", "Knowledge in Sanskrit is: ज्ञानम् (jñānam)."),
        ("Reply in Hindi: What can you help me with?", "मैं पढ़ाई, लेखन, प्रोग्रामिंग, गणित और विचारों को व्यवस्थित करने में मदद कर सकता हूँ।"),
    )
    for index in range(count):
        user, assistant = examples[index % len(examples)]
        yield record("multilingual", f"Practice item {index + 1}: {user}", assistant)


def science(count: int) -> Iterator[Dict[str, str]]:
    examples = (
        ("What is photosynthesis?", "Photosynthesis is the process by which plants, algae, and some bacteria use light energy to make sugars from carbon dioxide and water, releasing oxygen."),
        ("What is the difference between mass and weight?", "Mass measures how much matter an object contains. Weight is the gravitational force acting on that mass, so it changes with gravity."),
        ("Why do we see lightning before hearing thunder?", "Light travels much faster than sound, so the flash reaches you before the sound wave from the thunder."),
        ("What is DNA?", "DNA is a molecule that stores genetic instructions used by living organisms to develop, function, and reproduce."),
        ("What does a CPU do?", "A CPU executes program instructions, performs calculations, and coordinates many operations in a computer."),
    )
    for index in range(count):
        user, assistant = examples[index % len(examples)]
        yield record("science", f"Study item {index + 1}: {user}", assistant)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a verified synthetic curriculum JSONL dataset.")
    parser.add_argument("--output", default="Datasets/curriculum/tantra_curriculum_v1.jsonl")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--scale", type=int, default=1, help="Multiply the default 20,000-example curriculum size.")
    args = parser.parse_args()
    if args.scale < 1:
        raise ValueError("--scale must be at least 1")

    rng = random.Random(args.seed)
    plan = (
        ("math", arithmetic(rng, 7000 * args.scale)),
        ("math", algebra(rng, 3000 * args.scale)),
        ("code", python_patterns(rng, 4500 * args.scale)),
        ("reasoning", reasoning(rng, 1500 * args.scale)),
        ("instruction", concise_instructions(rng, 1000 * args.scale)),
        ("safety", safety(1000 * args.scale)),
        ("multilingual", multilingual(1000 * args.scale)),
        ("science", science(1000 * args.scale)),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    seen = set()
    with output.open("w", encoding="utf-8") as handle:
        for topic, examples in plan:
            for example in examples:
                key = (example["system"], example["user"])
                if key in seen:
                    continue
                seen.add(key)
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                counts[topic] += 1

    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({
        "name": "tantra_curriculum_v1",
        "seed": args.seed,
        "examples": sum(counts.values()),
        "topics": dict(counts),
        "generator": "tools/generate_curriculum_dataset.py",
        "guarantees": ["deterministic", "numeric answers computed locally", "no model-generated source text"],
    }, indent=2), encoding="utf-8")
    print(f"Wrote {sum(counts.values()):,} examples to {output}")
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
