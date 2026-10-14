"""
General-text corpus for BASE training.

The base should learn plain English here, while specialist conversation data
is reserved for the experts. If the base trained on the greeting data too,
the greeting expert would have nothing left to learn (loss 0.143 -> 0.141).

    python -m Tantra.data.seed_general --out Datasets/general --n 4000

Rows are {"text": ...} plain text (not chat), encoded with assistant_only=False
during base training.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

# topic -> sentence templates. Plain informative English, no chat formatting,
# no expert-specific answers (no "I'm Tantra", no refusal scripts, no jokes).
TOPICS: Dict[str, List[str]] = {
    "nature": [
        "A river starts as a small stream high in the mountains and grows wider as more streams join it.",
        "Trees communicate through their roots and through chemicals released into the air.",
        "Deserts are dry because they receive very little rain, sometimes less than a centimetre a year.",
        "Coral reefs support about a quarter of all ocean species despite covering less than one percent of the sea floor.",
        "Birds migrate south in autumn to find food and warmer weather, then return in spring.",
        "Bees travel from flower to flower collecting nectar, and in doing so they pollinate the plants.",
        "The deepest part of the ocean reaches almost eleven kilometres below the surface.",
        "Winter dormancy helps animals survive months when food is scarce.",
    ],
    "science": [
        "Gravity pulls objects toward each other, and it keeps the planets in orbit around the sun.",
        "Water expands when it freezes, which is why ice floats on the surface of a lake.",
        "Cells are the basic units of life, and every living thing is built from them.",
        "Light travels faster than anything else in the universe, covering three hundred thousand kilometres each second.",
        "Chemical reactions happen when atoms break old bonds and form new ones.",
        "The human brain contains billions of neurons that pass signals to one another.",
        "Energy cannot be created or destroyed; it only changes from one form to another.",
        "Plants convert sunlight, water and carbon dioxide into sugar and oxygen through photosynthesis.",
    ],
    "history": [
        "The first writing systems appeared in Mesopotamia over five thousand years ago.",
        "The printing press made books affordable and spread ideas much faster than before.",
        "Ancient traders crossed the Sahara carrying salt, gold and cloth between kingdoms.",
        "Cities grew rapidly during the industrial revolution as people moved from farms to factories.",
        "The first photographs were made in the early nineteenth century and took a long time to exposure.",
        "Navigators once used the stars to find their way across open oceans.",
        "Many modern languages descend from older tongues that changed slowly over centuries.",
    ],
    "everyday": [
        "Bread rises because yeast produces gas bubbles inside the dough.",
        "A calendar keeps track of days, weeks and months so people can plan ahead.",
        "Kettles boil water quickly using a heating element at the base.",
        "Most buses run on a fixed timetable so passengers know when to wait.",
        "A key cuts a pattern that matches the pins inside a lock.",
        "Umbrellas keep rain off because their fabric is stretched over a waterproof frame.",
        "Bicycles move forward when the rider pushes the pedals, which turn the wheels.",
        "Refrigerators keep food fresh by removing heat from the inside compartment.",
    ],
    "technology": [
        "A computer processor executes instructions one after another at very high speed.",
        "Data travels across the internet as small packets that are reassembled at the destination.",
        "Batteries store chemical energy and release it as electricity when a circuit is connected.",
        "Antennas send and receive radio waves that carry sound, video and data.",
        "Software is a set of instructions that tells hardware what to do.",
        "Satellites orbit the Earth and relay signals between distant places.",
        "A touchscreen detects where a finger touches by measuring changes in an electrical field.",
        "Databases store information in structured tables so it can be searched quickly.",
    ],
    "people": [
        "Children learn language by listening to the adults around them and trying the words themselves.",
        "Teams work best when everyone understands the goal and communicates clearly.",
        "A good night's sleep helps people remember what they learned during the day.",
        "Farmers plant seeds in spring and harvest the crops in autumn.",
        "Teachers explain new ideas with examples so students can connect them to what they already know.",
        "Doctors ask about symptoms before deciding on a treatment.",
        "Neighbours often help each other during storms or power cuts.",
        "Artists spend years practising their craft before their work looks effortless.",
    ],
    "space": [
        "The moon goes through phases as different parts of it catch sunlight.",
        "Mars has seasons like Earth because its axis is tilted at a similar angle.",
        "Astronauts float inside a space station because they are constantly falling around the Earth.",
        "The sun is a star made mostly of hydrogen that releases energy by fusing atoms together.",
        "Comets are made of ice and dust and grow bright tails when they near the sun.",
        "Light from the sun takes about eight minutes to reach the Earth.",
        "Telescopes gather more light than the human eye, so they can see much fainter objects.",
    ],
    "food": [
        "Rice is a staple food for more than half of the world's population.",
        "Cheese is made by adding bacteria and enzymes to milk, then letting it age.",
        "Spices were once so valuable that explorers sailed across oceans to find their source.",
        "A balanced meal includes proteins, carbohydrates, fats, vitamins and minerals.",
        "Soup tastes richer after simmering for a long time because flavors blend together.",
        "Fruit ripens faster when it is kept near bananas, which release a ripening gas.",
        "Baking requires precise measurements because small changes affect how bread rises.",
    ],
    "places": [
        "A lighthouse stands on a rocky coast and warns ships away from danger.",
        "Markets in old cities are narrow lanes full of stalls selling spices and cloth.",
        "Bridges carry roads and railways across rivers that would otherwise block travel.",
        "Museums keep objects from the past so people can learn how others lived.",
        "Mountains influence local weather by blocking clouds and forcing air upward.",
        "Harbours provide sheltered water where boats can dock safely.",
        "Parks give city residents a place to walk, rest and play away from traffic.",
    ],
}

FILLERS = [
    "", "", "",  # most rows are a single sentence
    " It is a small detail that most people never notice.",
    " This matters more than it might seem at first.",
    " Scientists are still studying why this happens.",
    " The effect can be seen clearly with a little patience.",
    " People have known this for a very long time.",
]


def generate(n: int, seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    topics = list(TOPICS)
    all_sents = [(t, s) for t in TOPICS for s in TOPICS[t]]
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        # 1-4 sentences, no repeats within a row -> combinatorial variety
        k = rng.choice([1, 1, 2, 2, 3, 4])
        picks = rng.sample(all_sents, k)
        text = " ".join(s for _, s in picks) + rng.choice(FILLERS)
        if text in seen:
            continue
        seen.add(text)
        row = {"text": text}
        if k == 1:
            row["topic"] = picks[0][0]
        out.append(row)
    return out


def write_all(out_dir: str | Path, n: int = 4000, seed: int = 0) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = generate(n, seed)
    with open(out_dir / "general.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Datasets/general")
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    print(f"wrote {write_all(a.out, a.n)} rows to {a.out}")
