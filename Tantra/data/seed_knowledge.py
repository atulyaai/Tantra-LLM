"""
Seed data for the first knowledge experts (English):

  gk_geography   capitals, continents, big rivers/mountains/oceans
  gk_science     basic science facts (planets, body, physics, chemistry)
  gk_math        arithmetic, generated and always correct

Each fact has several question phrasings and short, correct answers.
These experts' data is NEVER shown to the base model, so the expert has to
add the knowledge itself (that is the point of the expert design).

    python -m Tantra.data.seed_knowledge --out Datasets/curated
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

CAPITALS = {
    "India": "New Delhi", "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid",
    "the United Kingdom": "London", "the United States": "Washington, D.C.", "Canada": "Ottawa",
    "Mexico": "Mexico City", "Brazil": "Brasilia", "Argentina": "Buenos Aires", "Japan": "Tokyo",
    "China": "Beijing", "Russia": "Moscow", "Australia": "Canberra", "Egypt": "Cairo", "Kenya": "Nairobi",
    "Nigeria": "Abuja", "South Africa": "Pretoria", "Nepal": "Kathmandu", "Bangladesh": "Dhaka",
    "Pakistan": "Islamabad", "Sri Lanka": "Colombo", "Thailand": "Bangkok", "Indonesia": "Jakarta",
    "South Korea": "Seoul", "Turkey": "Ankara", "Greece": "Athens", "Portugal": "Lisbon",
    "the Netherlands": "Amsterdam", "Sweden": "Stockholm", "Norway": "Oslo", "Poland": "Warsaw",
    "Ireland": "Dublin", "Saudi Arabia": "Riyadh", "Iran": "Tehran", "Afghanistan": "Kabul",
    "New Zealand": "Wellington", "Switzerland": "Bern", "Austria": "Vienna", "Vietnam": "Hanoi",
    "Malaysia": "Kuala Lumpur", "the Philippines": "Manila", "Peru": "Lima", "Chile": "Santiago",
}
CONTINENTS = {
    "India": "Asia", "France": "Europe", "Brazil": "South America", "Egypt": "Africa", "Canada": "North America",
    "Australia": "Oceania", "Japan": "Asia", "Kenya": "Africa", "Germany": "Europe", "Argentina": "South America",
}
GEO_FACTS: List[Tuple[List[str], str]] = [
    (["What is the largest ocean?", "Which ocean is the biggest?"], "The Pacific Ocean is the largest ocean."),
    (["What is the highest mountain in the world?", "Which is the tallest mountain?"],
     "Mount Everest is the highest mountain in the world, at about 8,849 metres."),
    (["What is the longest river in the world?", "Which river is the longest?"],
     "The Nile is usually considered the longest river in the world."),
    (["How many continents are there?", "How many continents does Earth have?"], "There are seven continents."),
    (["What is the largest continent?", "Which continent is the biggest?"], "Asia is the largest continent."),
    (["What is the largest country by area?", "Which country is the biggest?"], "Russia is the largest country by area."),
    (["What is the largest desert?", "Which is the biggest hot desert?"],
     "The Sahara is the largest hot desert. Antarctica is the largest desert overall."),
    (["Which river flows through Delhi?"], "The Yamuna river flows through Delhi."),
    (["What is the longest river in India?"], "The Ganga is the longest river in India."),
    (["What is the capital of Australia?"], "Canberra is the capital of Australia."),
    (["Which country is famous for kangaroos?"], "Australia is famous for kangaroos."),
    (["What is the smallest country in the world?", "Which country is the smallest?"],
     "Vatican City is the smallest country in the world."),
    (["Which is the most populous country?", "Which country has the most people?"],
     "India is the most populous country in the world."),
    (["What is the capital of Canada?"], "Ottawa is the capital of Canada."),
    (["Which country is shaped like a boot?", "What country looks like a boot?"], "Italy is shaped like a boot."),
    (["What is the highest waterfall in the world?", "Which waterfall is the tallest?"],
     "Angel Falls in Venezuela is the highest waterfall in the world."),
    (["Which is the deepest ocean trench?", "What is the deepest part of the ocean?"],
     "The Mariana Trench is the deepest part of the ocean."),
    (["What is the largest island in the world?", "Which island is the biggest?"], "Greenland is the largest island in the world."),
    (["What is the capital of Egypt?"], "Cairo is the capital of Egypt."),
    (["Which continent is Egypt in?", "What continent is Egypt part of?"], "Egypt is in Africa."),
    (["What is the capital of Brazil?"], "Brasilia is the capital of Brazil."),
    (["Which country is home to the Amazon rainforest?", "Where is the Amazon rainforest?"],
     "Brazil is home to most of the Amazon rainforest."),
    (["What is the capital of Japan?"], "Tokyo is the capital of Japan."),
    (["Which country spans two continents?", "What country is in both Europe and Asia?"],
     "Russia spans both Europe and Asia."),
    (["What is the capital of South Korea?"], "Seoul is the capital of South Korea."),
    (["Which country is known for the Great Wall?", "Where is the Great Wall?"], "The Great Wall is in China."),
    (["What is the capital of Mexico?"], "Mexico City is the capital of Mexico."),
    (["Which city straddles two continents?", "What city is in both Europe and Asia?"],
     "Istanbul straddles Europe and Asia."),
    (["What is the capital of Turkey?"], "Ankara is the capital of Turkey."),
    (["Which country is famous for ancient Persia?", "Where was the Persian Empire?"], "Iran is the heart of ancient Persia."),
    (["What is the capital of Afghanistan?"], "Kabul is the capital of Afghanistan."),
    (["Which country is known for the Himalayan mountains?", "Where are the Himalayas?"],
     "Nepal is home to the Himalayas, including Mount Everest."),
    (["What is the capital of Nepal?"], "Kathmandu is the capital of Nepal."),
    (["Which country is home to Mount Everest?", "Where is Mount Everest located?"],
     "Mount Everest is on the border of Nepal and China."),
    (["What is the capital of Sri Lanka?"], "Sri Jayawardenepura Kotte is the capital of Sri Lanka."),
    (["Which country is famous for tea?", "Where does much of the world's tea come from?"],
     "Sri Lanka is famous for Ceylon tea."),
    (["What is the capital of Bangladesh?"], "Dhaka is the capital of Bangladesh."),
    (["Which country has the Sundarbans mangrove forest?", "Where is the Sundarbans?"],
     "Bangladesh is home to the Sundarbans."),
    (["What is the capital of Indonesia?"], "Jakarta is the capital of Indonesia."),
    (["Which country is made up of over 17,000 islands?", "What country has the most islands?"],
     "Indonesia is made up of over 17,000 islands."),
    (["What is the capital of Malaysia?"], "Kuala Lumpur is the capital of Malaysia."),
    (["Which country is famous for the Petronas Towers?", "Where are the Petronas Towers?"],
     "The Petronas Towers are in Malaysia."),
    (["What is the capital of the Philippines?"], "Manila is the capital of the Philippines."),
    (["Which country is made up of over 7,000 islands?", "What country has 7,000 islands?"],
     "The Philippines is made up of over 7,000 islands."),
    (["What is the capital of Vietnam?"], "Hanoi is the capital of Vietnam."),
    (["Which country is famous for pho?", "Where did pho originate?"], "Vietnam is famous for pho."),
    (["What is the capital of Peru?"], "Lima is the capital of Peru."),
    (["Which country is the longest north to south?", "What country is very long and thin?"],
     "Chile is the longest north-to-south country in the world."),
    (["What is the capital of Argentina?"], "Buenos Aires is the capital of Argentina."),
]
SCIENCE_FACTS: List[Tuple[List[str], str]] = [
    (["How many planets are in the solar system?", "How many planets are there?"],
     "There are eight planets in the solar system."),
    (["What is the largest planet?", "Which planet is the biggest?"], "Jupiter is the largest planet in the solar system."),
    (["What is the closest planet to the sun?", "Which planet is nearest the sun?"], "Mercury is the closest planet to the sun."),
    (["Which planet is called the red planet?", "What is the red planet?"], "Mars is called the red planet."),
    (["What planet do we live on?"], "We live on Earth, the third planet from the sun."),
    (["What is the sun?", "Is the sun a star?"], "The sun is a star, a huge ball of hot gas at the centre of our solar system."),
    (["What is water made of?", "What is the chemical formula of water?"],
     "Water is made of hydrogen and oxygen. Its formula is H2O."),
    (["At what temperature does water boil?", "What is the boiling point of water?"],
     "Water boils at 100 degrees Celsius at sea level."),
    (["At what temperature does water freeze?", "What is the freezing point of water?"],
     "Water freezes at 0 degrees Celsius."),
    (["What gas do plants take in?", "What do plants need from the air?"],
     "Plants take in carbon dioxide and release oxygen during photosynthesis."),
    (["What gas do humans breathe in?", "What do we need to breathe?"], "Humans need oxygen to breathe."),
    (["How many bones are in the human body?", "How many bones does an adult have?"],
     "An adult human has 206 bones."),
    (["What is the largest organ of the human body?"], "The skin is the largest organ of the human body."),
    (["What pumps blood in the body?", "What does the heart do?"], "The heart pumps blood around the body."),
    (["What is gravity?"], "Gravity is the force that pulls objects toward each other, like things falling to the ground."),
    (["What is the speed of light?", "How fast is light?"], "Light travels about 300,000 kilometres per second."),
    (["What is photosynthesis?"], "Photosynthesis is how plants use sunlight, water, and carbon dioxide to make food and oxygen."),
    (["What are the three states of matter?"], "The three common states of matter are solid, liquid, and gas."),
    (["What is the chemical symbol for gold?"], "The chemical symbol for gold is Au."),
    (["What is the chemical symbol for iron?"], "The chemical symbol for iron is Fe."),
    (["What is the moon?", "What is Earth's moon?"], "The moon is Earth's natural satellite. It orbits Earth about once a month."),
    (["Why is the sky blue?"], "The sky looks blue because air scatters blue sunlight more than other colours."),
]

CAP_Q = ["What is the capital of {c}?", "capital of {c}?", "What's the capital city of {c}?",
         "Which city is the capital of {c}?", "Tell me the capital of {c}."]
CAP_A = ["The capital of {c} is {k}.", "{k} is the capital of {c}."]
CONT_Q = ["Which continent is {c} in?", "{c} is in which continent?", "On which continent is {c}?"]
CONT_A = ["{c} is in {k}.", "{c} is located in {k}."]


def _cap(c: str) -> str:
    return c[:1].upper() + c[1:]


def _noise(s: str, rng: random.Random) -> str:
    if rng.random() < 0.3:
        s = s.lower()
    if rng.random() < 0.3:
        s = s.rstrip("?.")
    return s


def _row(q: str, a: str, expert: str) -> Dict:
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "expert": expert}


def gen_geography(n: int, rng: random.Random) -> List[Dict]:
    pool = []
    for c, k in CAPITALS.items():
        for q in CAP_Q:
            for a in CAP_A:
                pool.append((q.format(c=c), _cap(a.format(c=c, k=k))))
    for c, k in CONTINENTS.items():
        for q in CONT_Q:
            for a in CONT_A:
                pool.append((_cap(q.format(c=c)), _cap(a.format(c=c, k=k))))
    for qs, a in GEO_FACTS:
        pool += [(q, a) for q in qs] * 6
    return [_row(_noise(q, rng), a, "gk_geography") for q, a in rng.sample(pool, min(n, len(pool)))]


def gen_science(n: int, rng: random.Random) -> List[Dict]:
    pool = []
    for qs, a in SCIENCE_FACTS:
        pool += [(q, a) for q in qs]
    out = []
    while len(out) < n:
        q, a = rng.choice(pool)
        out.append(_row(_noise(q, rng), a, "gk_science"))
    return out


def gen_math(n: int, rng: random.Random) -> List[Dict]:
    out, seen = [], set()
    ops = [("+", lambda a, b: a + b, ["plus", "+"]), ("-", lambda a, b: a - b, ["minus", "-"]),
           ("*", lambda a, b: a * b, ["times", "*", "multiplied by", "x"])]
    while len(out) < n:
        sym, f, words = rng.choice(ops)
        a = rng.randint(0, 20 if sym == "*" else 100)
        b = rng.randint(0, 20 if sym == "*" else 100)
        if sym == "-" and b > a:
            a, b = b, a
        w = rng.choice(words)
        q = rng.choice(["What is {a} {w} {b}?", "{a} {w} {b} = ?", "Calculate {a} {w} {b}.", "what's {a} {w} {b}"]
                       ).format(a=a, w=w, b=b)
        if q in seen:
            continue
        seen.add(q)
        out.append(_row(q, f"{a} {sym.replace('*', 'x')} {b} = {f(a, b)}.", "gk_math"))
    return out


GENERATORS = {"gk_geography": gen_geography, "gk_science": gen_science, "gk_math": gen_math}


def write_all(out_dir: str | Path, n: int = 1500, seed: int = 0) -> Dict[str, int]:
    out_dir = Path(out_dir) / "knowledge"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, gen in GENERATORS.items():
        rows = gen(n, random.Random(f"{seed}-{name}"))
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[name] = len(rows)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Datasets/curated")
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    print(write_all(a.out, a.n))
