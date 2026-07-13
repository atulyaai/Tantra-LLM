"""Generate local teacher QA datasets for staged NP-DNA training.

This script creates deterministic, category-labeled JSONL files with short,
medium, and long answers. It is intentionally conservative: examples are
simple, clean, and balanced for early response learning.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path


OUT_DIR = Path("Download/teacher")
SOURCE_BALANCED = Path("Download/qa/seed_chat_balanced_100k.jsonl")
SYSTEMS = [
    "You are Atulya. Answer clearly and briefly.",
    "You are Atulya, a helpful AI assistant. Be accurate and direct.",
    "You are Atulya. Give a useful answer with simple wording.",
]

FACTS = [
    ("What is gravity?", "Gravity is the force that pulls objects with mass toward each other.", "It gives objects weight on Earth and keeps planets moving around the Sun."),
    ("What is photosynthesis?", "Photosynthesis is how plants use sunlight to make food.", "Plants take in carbon dioxide and water, then produce sugar and oxygen."),
    ("What is evaporation?", "Evaporation is when liquid changes into vapor.", "It happens faster with heat, wind, and a larger surface area."),
    ("What is condensation?", "Condensation is when vapor cools and becomes liquid.", "Clouds, dew, and water drops on a cold glass are common examples."),
    ("What is friction?", "Friction is a force that resists motion between surfaces.", "It helps us walk, brake, and grip objects."),
    ("What is inertia?", "Inertia is an object's tendency to keep its current motion.", "A still object tends to stay still, and a moving object tends to keep moving."),
    ("What is energy?", "Energy is the ability to do work or cause change.", "It can appear as motion, heat, light, electricity, or stored chemical energy."),
    ("What is a molecule?", "A molecule is a group of atoms bonded together.", "Water is a molecule made from two hydrogen atoms and one oxygen atom."),
    ("What is an atom?", "An atom is the basic unit of ordinary matter.", "Atoms contain protons, neutrons, and electrons."),
    ("What is a cell?", "A cell is the basic unit of life.", "Living things are made of one or more cells."),
    ("What is DNA?", "DNA stores genetic instructions.", "It helps cells build proteins and pass traits from parents to offspring."),
    ("What is an ecosystem?", "An ecosystem is a community of organisms and their environment.", "Plants, animals, microbes, water, soil, and climate all interact in it."),
    ("What is climate?", "Climate is the long-term pattern of weather in a region.", "Weather changes daily, while climate is measured over many years."),
    ("What is a democracy?", "A democracy is a system where people have a voice in government.", "Citizens usually vote for leaders or laws."),
    ("What is economics?", "Economics studies how people use limited resources.", "It looks at production, trade, money, choices, and incentives."),
    ("What is inflation?", "Inflation is a general rise in prices over time.", "When inflation is high, the same money buys less than before."),
    ("What is a database?", "A database is an organized collection of data.", "It helps store, search, update, and manage information efficiently."),
    ("What is an algorithm?", "An algorithm is a step-by-step method for solving a problem.", "Recipes, search procedures, and sorting methods are examples."),
    ("What is machine learning?", "Machine learning lets computers learn patterns from data.", "A model is trained on examples and then used to make predictions or generate output."),
    ("What is artificial intelligence?", "Artificial intelligence is software that performs tasks associated with human intelligence.", "It can include language, vision, planning, search, and decision-making."),
    ("What is a neural network?", "A neural network is a model made of connected layers of numbers.", "It learns by adjusting weights so its outputs better match examples."),
    ("What is overfitting?", "Overfitting happens when a model memorizes training data too closely.", "It performs well on seen examples but poorly on new ones."),
    ("What is underfitting?", "Underfitting happens when a model is too simple or undertrained.", "It fails to capture important patterns in the data."),
    ("What is a tokenizer?", "A tokenizer converts text into tokens that a model can process.", "Tokens may be words, word pieces, characters, or byte-like units."),
    ("What is a variable in programming?", "A variable is a named place to store a value.", "Programs use variables to remember and update information."),
    ("What is a function in programming?", "A function is a reusable block of code.", "It can take inputs, perform work, and return a result."),
    ("What is recursion?", "Recursion is when a function calls itself.", "It needs a base case so it eventually stops."),
    ("What is a loop?", "A loop repeats a block of code.", "Loops are useful for processing lists, counting, or retrying work."),
    ("What is an API?", "An API is a way for software systems to communicate.", "It defines what requests can be made and what responses are returned."),
    ("What is encryption?", "Encryption turns readable data into protected unreadable data.", "Only someone with the right key can turn it back into readable form."),
    ("What is a hypothesis?", "A hypothesis is a testable explanation or prediction.", "Scientists use experiments or observations to check whether it is supported."),
    ("What is evidence?", "Evidence is information that supports or challenges a claim.", "Good evidence is relevant, reliable, and specific."),
    ("What is critical thinking?", "Critical thinking means judging ideas carefully using reason and evidence.", "It includes checking assumptions, comparing explanations, and avoiding quick conclusions."),
    ("What is empathy?", "Empathy is understanding or sharing another person's feelings.", "It helps people communicate kindly and respond with care."),
    ("What is resilience?", "Resilience is the ability to recover from difficulty.", "It grows through support, practice, patience, and problem-solving."),
    ("What is communication?", "Communication is sharing information or meaning.", "It can happen through words, tone, body language, images, or writing."),
    ("What is leadership?", "Leadership is guiding people toward a goal.", "Good leaders communicate clearly, take responsibility, and help others work well."),
    ("What is project management?", "Project management is planning and organizing work to reach a goal.", "It includes scope, tasks, timelines, resources, risks, and communication."),
    ("What is a budget?", "A budget is a plan for income and spending.", "It helps people control costs and prepare for future needs."),
    ("What is interest?", "Interest is the cost of borrowing money or the reward for lending it.", "It is often shown as a percentage over time."),
]

REASONING = [
    ("Why should I verify important answers?", "Because any model can make mistakes. Verification protects you from acting on incomplete or incorrect information."),
    ("How can I learn a hard topic?", "Break it into smaller parts, practice one part at a time, and test yourself often. Short daily practice usually works better than rare long sessions."),
    ("How do I debug a problem?", "Reproduce it, isolate the smallest failing case, inspect inputs and outputs, then test one change at a time."),
    ("Why does clean data matter for training?", "The model learns patterns from whatever it sees. Clean data teaches useful patterns, while noisy data teaches noise."),
    ("How can I explain a complex idea simply?", "Start with the main point, use familiar examples, avoid extra terms, and add detail only after the core idea is clear."),
    ("Why is a smaller model sometimes better at first?", "A smaller model trains faster and can reveal data or architecture problems sooner. Once the pipeline works, scaling becomes safer."),
    ("How do I make better decisions?", "Define the goal, list options, compare tradeoffs, check evidence, and choose the option with the best expected outcome."),
    ("Why does repetition happen in generated text?", "The model may overvalue frequent tokens or fail to learn when to stop. Better data balance and repetition penalties can reduce it."),
    ("How do I improve a dataset?", "Remove bad rows, balance common patterns, add missing examples, and keep answers accurate and varied."),
    ("Why use validation loss?", "Validation loss checks performance on held-out data. It helps detect whether training progress is real or just memorization."),
]

CONVERSATION = [
    ("Hi! How are you?", "I'm doing well. How can I help you today?"),
    ("Hello.", "Hello! What would you like help with?"),
    ("Who are you?", "I am Atulya, a helpful AI assistant."),
    ("What can you do?", "I can answer questions, explain concepts, help with code, and organize ideas."),
    ("Thanks for your help.", "You're welcome. I'm glad I could help."),
    ("Can you explain that again?", "Sure. I can restate it more simply."),
    ("I am confused.", "No problem. Tell me which part is unclear, and we can work through it step by step."),
    ("Can you give an example?", "Yes. A concrete example can make the idea easier to understand."),
    ("Please be brief.", "Understood. I will keep the answer short."),
    ("Can you go deeper?", "Yes. I can add more detail and explain the reasoning behind it."),
]

CODE = [
    ("Write a Python function to add two numbers.", "def add(a, b):\n    return a + b"),
    ("Write a Python function to multiply two numbers.", "def multiply(a, b):\n    return a * b"),
    ("Check if a number is even in Python.", "def is_even(n):\n    return n % 2 == 0"),
    ("Reverse a string in Python.", "def reverse_text(s):\n    return s[::-1]"),
    ("Find the largest number in a list.", "def largest(nums):\n    return max(nums)"),
    ("Filter even numbers from a list.", "def even_numbers(nums):\n    return [n for n in nums if n % 2 == 0]"),
    ("Count words in a string.", "def count_words(text):\n    return len(text.split())"),
    ("Sort a list in Python.", "def sort_items(items):\n    return sorted(items)"),
    ("Check if a string is a palindrome.", "def is_palindrome(s):\n    cleaned = ''.join(ch.lower() for ch in s if ch.isalnum())\n    return cleaned == cleaned[::-1]"),
    ("Create a dictionary from two lists.", "def make_dict(keys, values):\n    return dict(zip(keys, values))"),
]

OPENERS = [
    "{answer}",
    "In short, {lower}",
    "Simply put, {lower}",
    "A useful way to say it is: {answer}",
    "The simple answer is: {lower}",
    "It means {lower}",
    "You can think of it as this: {answer}",
    "Practically, {lower}",
    "For most purposes, {lower}",
    "One clear answer is: {answer}",
]

QUESTION_PREFIXES = [
    "",
    "Answer briefly: ",
    "Explain simply: ",
    "Can you answer this? ",
    "Give a useful answer: ",
    "In plain English, ",
    "Quick question: ",
]


def lower_first(text: str) -> str:
    text = text.strip()
    return text[:1].lower() + text[1:] if text else text


def style_answer(answer: str, detail: str, length: str, idx: int) -> str:
    base = answer.strip()
    if length == "short":
        text = base
    elif length == "medium":
        text = f"{base} {detail.strip()}"
    else:
        text = (
            f"{base} {detail.strip()} "
            "A good way to understand it is to connect the definition to a familiar example. "
            "Start with the main idea, then add details only when they help the answer become clearer."
        )
    template = OPENERS[idx % len(OPENERS)]
    return template.format(answer=text, lower=lower_first(text))


def make_row(user: str, assistant: str, category: str, rng: random.Random, prefix_idx: int) -> dict[str, str]:
    prefix = QUESTION_PREFIXES[prefix_idx % len(QUESTION_PREFIXES)]
    prompt = user.strip()
    if prefix:
        prompt = prefix + lower_first(prompt)
    return {
        "system": rng.choice(SYSTEMS),
        "user": prompt,
        "assistant": assistant.strip(),
        "category": category,
    }


def first_token(text: str) -> str:
    match = re.search(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]", text.strip())
    return match.group(0) if match else ""


def entropy(rows: list[dict[str, str]]) -> float:
    counts = Counter(first_token(r["assistant"]) for r in rows)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values()) if total else 0.0


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    out = []
    for row in rows:
        key = (
            re.sub(r"\s+", " ", row["user"].lower()).strip(),
            re.sub(r"\s+", " ", row["assistant"].lower()).strip(),
        )
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def expand(rows: list[tuple[str, str, str]], category: str, length: str, target: int, rng: random.Random) -> list[dict[str, str]]:
    out = []
    i = 0
    while len(out) < target:
        user, answer, detail = rows[i % len(rows)]
        answer_text = style_answer(answer, detail, length, i)
        out.append(make_row(user, answer_text, category, rng, i))
        i += 1
        if i > target * 20:
            break
    return dedupe(out)[:target]


def expand_simple(rows: list[tuple[str, str]], category: str, length: str, target: int, rng: random.Random) -> list[dict[str, str]]:
    triples = []
    for user, answer in rows:
        detail = "This answer focuses on the practical point and avoids unnecessary detail."
        triples.append((user, answer, detail))
    return expand(triples, category, length, target, rng)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_source_row(row: dict[str, str]) -> bool:
    user = row.get("user", "").strip()
    assistant = row.get("assistant", "").strip()
    joined = f"{user} {assistant}".lower()
    bad = (
        "thought:",
        "action:",
        "available apis:",
        "relevant apis:",
        "api_name",
        "tool_name",
        "paraphrase answer:",
        "i'm not sure i can answer",
        "what is it?",
        "what is it,",
    )
    if any(x in joined for x in bad):
        return False
    if len(user) < 4 or len(assistant) < 8:
        return False
    if len(user) > 220 or len(assistant) > 420:
        return False
    return True


def load_balanced_source(path: Path = SOURCE_BALANCED) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if clean_source_row(row):
                rows.append(row)
    return rows


SOURCE_OPENERS = [
    "{answer}",
    "In short, {lower}",
    "Simply put, {lower}",
    "The key idea is that {lower}",
    "It means {lower}",
    "A practical way to say it is: {answer}",
    "You can understand it as follows: {answer}",
    "For a simple answer, {lower}",
    "One useful answer is: {answer}",
    "Briefly, {lower}",
]


def source_variant(row: dict[str, str], length: str, category: str, i: int) -> dict[str, str]:
    user = row["user"].strip()
    answer = row["assistant"].strip()
    if length == "short":
        answer_text = answer.split("\n")[0]
        if len(answer_text) > 180:
            answer_text = answer_text[:177].rstrip() + "..."
    elif length == "medium":
        answer_text = answer
        if len(answer_text) < 120:
            answer_text = answer_text.rstrip(".") + ". This gives the main point without adding unnecessary detail."
    else:
        answer_text = (
            answer.rstrip(".")
            + ". The important part is to connect the answer to the user's question, keep the wording precise, "
            + "and avoid adding claims that are not needed. A clear response should define the idea, give one useful detail, "
            + "and stop before it becomes noisy."
        )
        if len(answer_text) > 760:
            answer_text = answer_text[:757].rstrip() + "..."
    template = SOURCE_OPENERS[i % len(SOURCE_OPENERS)]
    answer_text = template.format(answer=answer_text, lower=lower_first(answer_text))
    prefixes = [
        "",
        "Answer clearly: ",
        "Give a practical answer: ",
        "Explain in plain English: ",
        "Help me understand: ",
        "Can you answer this? ",
    ]
    prefix = prefixes[(i // len(SOURCE_OPENERS)) % len(prefixes)]
    if prefix:
        user = prefix + lower_first(user)
    return {
        "system": row.get("system") or SYSTEMS[i % len(SYSTEMS)],
        "user": user,
        "assistant": answer_text,
        "category": category,
    }


def fill_from_source(
    rows: list[dict[str, str]],
    source: list[dict[str, str]],
    target: int,
    length: str,
    category: str,
    rng: random.Random,
) -> list[dict[str, str]]:
    out = list(rows)
    seen = {
        (
            re.sub(r"\s+", " ", r["user"].lower()).strip(),
            re.sub(r"\s+", " ", r["assistant"].lower()).strip(),
        )
        for r in out
    }
    if not source:
        return out[:target]
    shuffled = list(source)
    rng.shuffle(shuffled)
    i = 0
    while len(out) < target and i < target * 20:
        base = shuffled[i % len(shuffled)]
        row = source_variant(base, length, category, i)
        key = (
            re.sub(r"\s+", " ", row["user"].lower()).strip(),
            re.sub(r"\s+", " ", row["assistant"].lower()).strip(),
        )
        if key not in seen:
            seen.add(key)
            out.append(row)
        i += 1
    return out[:target]


def report(name: str, rows: list[dict[str, str]]) -> None:
    cats = Counter(r["category"] for r in rows)
    firsts = Counter(first_token(r["assistant"]) for r in rows)
    print(f"{name}: rows={len(rows)} entropy={entropy(rows):.2f}")
    print("  categories:", dict(cats))
    print("  first_tokens:", firsts.most_common(12))


def build(seed: int) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(seed)
    source = load_balanced_source()
    fact_triples = FACTS
    reasoning_triples = [(u, a, "The key is to use a clear process instead of guessing.") for u, a in REASONING]
    conversation_triples = [(u, a, "A helpful reply should be friendly, direct, and easy to continue.") for u, a in CONVERSATION]
    code_triples = [(u, a, "This code is intentionally small so it is easy to inspect and modify.") for u, a in CODE]

    short = []
    short += expand(fact_triples, "factual_short", "short", 12_000, rng)
    short += expand(reasoning_triples, "reasoning_short", "short", 4_000, rng)
    short += expand(conversation_triples, "conversation_short", "short", 4_000, rng)
    short += expand(code_triples, "code_short", "short", 5_000, rng)

    medium = []
    medium += expand(fact_triples, "factual_medium", "medium", 10_000, rng)
    medium += expand(reasoning_triples, "reasoning_medium", "medium", 7_000, rng)
    medium += expand(conversation_triples, "conversation_medium", "medium", 4_000, rng)
    medium += expand(code_triples, "code_medium", "medium", 4_000, rng)

    long = []
    long += expand(fact_triples, "factual_long", "long", 5_000, rng)
    long += expand(reasoning_triples, "reasoning_long", "long", 4_000, rng)
    long += expand(conversation_triples, "conversation_long", "long", 2_000, rng)
    long += expand(code_triples, "code_long", "long", 4_000, rng)

    paraphrase = []
    all_bases = fact_triples + reasoning_triples + conversation_triples + code_triples
    for length, count in [("short", 5_000), ("medium", 5_000), ("long", 5_000)]:
        paraphrase += expand(all_bases, f"paraphrase_{length}", length, count, rng)

    # Dedupe and shuffle within each file.
    datasets = {
        "teacher_short_25k.jsonl": fill_from_source(dedupe(short), source, 25_000, "short", "teacher_short", rng),
        "teacher_medium_25k.jsonl": fill_from_source(dedupe(medium), source, 25_000, "medium", "teacher_medium", rng),
        "teacher_long_15k.jsonl": fill_from_source(dedupe(long), source, 15_000, "long", "teacher_long", rng),
        "teacher_paraphrase_15k.jsonl": fill_from_source(dedupe(paraphrase), source, 15_000, "medium", "teacher_paraphrase", rng),
    }
    mixed = []
    for rows in datasets.values():
        mixed.extend(rows)
    rng.shuffle(mixed)
    datasets["teacher_mixed_80k.jsonl"] = dedupe(mixed)[:80_000]
    for rows in datasets.values():
        rng.shuffle(rows)
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = build(args.seed)
    for name, rows in datasets.items():
        path = args.output_dir / name
        write_jsonl(path, rows)
        report(name, rows)


if __name__ == "__main__":
    main()
