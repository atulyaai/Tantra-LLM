"""
Stage-2 expert data: tasks + reasoning + multilingual
=====================================================
Adds five experts on top of the stage-1 model (base untouched, only new LoRA
experts + a router row):

  tasks/summarization.jsonl    short passage -> one-line summary
  translation/translation.jsonl  English <-> Hindi sentence pairs
  reasoning/logic.jsonl        riddles and logic questions with real answers
  tasks/creative_writing.jsonl short poems / lines on a theme
  multilingual/lang_hi.jsonl   Hindi conversation

    python -m Tantra.data.seed_stage2 --out Datasets/curated --n 400

Same rules as the stage-1 seeders: hand-written intents x phrasings x surface
noise, every row a real user/assistant pair, no template filler.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

# ── summarization ─────────────────────────────────────────────────────────────
# topic -> facts; a passage picks 3 facts, the summary keeps the key one.
FACTS: Dict[str, List[str]] = {
    "space": [
        "The moon goes through phases as different parts catch sunlight.",
        "Mars has seasons because its axis is tilted like Earth's.",
        "Astronauts float in orbit because they are constantly falling around Earth.",
        "The sun releases energy by fusing hydrogen atoms together.",
        "Light from the sun takes about eight minutes to reach us.",
        "Comets grow bright tails of gas and dust when they near the sun.",
    ],
    "nature": [
        "Bees pollinate plants as they travel between flowers for nectar.",
        "Coral reefs support a quarter of ocean species.",
        "Deserts get less than a centimetre of rain some years.",
        "Birds migrate south in autumn and return in spring.",
        "Trees share water and signals through their root networks.",
        "The deepest ocean trenches reach almost eleven kilometres down.",
    ],
    "body": [
        "The brain passes signals between billions of neurons.",
        "Sleep helps the brain store what was learned during the day.",
        "The heart pumps blood through vessels to every organ.",
        "Cells are the basic building blocks of every living thing.",
        "Exercise strengthens the heart and improves circulation.",
        "The immune system remembers germs it has fought before.",
    ],
    "tech": [
        "Processors execute billions of instructions every second.",
        "Data crosses the internet in small reassembled packets.",
        "Batteries store chemical energy and release it as electricity.",
        "Software is instructions that tell hardware what to do.",
        "Satellites relay signals between distant places on Earth.",
        "Touchscreens sense finger position through changes in an electric field.",
    ],
    "history": [
        "The printing press made books cheap and spread ideas fast.",
        "The first writing appeared in Mesopotamia over five thousand years ago.",
        "Ancient traders crossed the Sahara with salt and gold.",
        "Cities grew quickly when factories opened during industrialisation.",
        "Navigators once steered across oceans by watching the stars.",
        "The first photographs took a long time to expose.",
    ],
}

SUM_LEAD = ["In short,", "Basically,", "To sum up,", "Simply put,", "", ""]
SUM_TAIL = ["", "", " That is the main point.", " That is what matters most."]


def _summarization(rng: random.Random, n: int) -> List[Dict]:
    out, seen = [], set()
    topics = list(FACTS)
    while len(out) < n and len(seen) < n * 20:
        t = rng.choice(topics)
        facts = rng.sample(FACTS[t], 3)
        passage = " ".join(facts)
        key = facts[0].split(".")[0].lower()
        summary = rng.choice(SUM_LEAD) + f"the passage explains {key}." + rng.choice(SUM_TAIL)
        summary = " ".join(summary.split())
        if summary in seen:
            continue
        seen.add(summary)
        out.append({"messages": [
            {"role": "user", "content": f"summarise this in one line:\n\n{passage}"},
            {"role": "assistant", "content": summary},
        ]})
    return out


# ── translation (hand-written pairs, en <-> hi) ───────────────────────────────
PAIRS = [
    ("hello", "namaste"),
    ("good morning", "suprabhat"),
    ("good night", "shubh ratri"),
    ("thank you", "dhanyavaad"),
    ("please help me", "kripya meri madad karein"),
    ("how are you", "aap kaise hain"),
    ("i am fine", "main theek hoon"),
    ("what is your name", "aapka naam kya hai"),
    ("my name is rahul", "mera naam rahul hai"),
    ("where do you live", "aap kahan rehte hain"),
    ("i live in delhi", "main delhi mein rehta hoon"),
    ("what time is it", "ab kitne baje hain"),
    ("i am hungry", " mujhe bhookh lagi hai"),
    ("i am thirsty", "mujhe pyaas lagi hai"),
    ("the weather is nice", "mausam accha hai"),
    ("it is raining", "baarish ho rahi hai"),
    ("i go to school", "main school jaata hoon"),
    ("i like to read", "mujhe padhna pasand hai"),
    ("call a doctor", "doctor ko bulao"),
    ("i do not understand", "mujhe samajh nahin aaya"),
    ("please speak slowly", "kripya dheere boliye"),
    ("open the door", "darwaza kholo"),
    ("close the window", "khidki band karo"),
    ("the food is delicious", "khana bahut swadisht hai"),
    ("i love my family", "main apne pariwar se pyaar karta hoon"),
    ("we are friends", "hum dost hain"),
    ("happy birthday", "janamdin mubarak ho"),
    ("good luck", "shubhkaamnayein"),
    ("see you tomorrow", "kal milte hain"),
    ("i am going home", "main ghar ja raha hoon"),
    ("the book is on the table", "kitab mez par hai"),
    ("children are playing", "bachche khel rahe hain"),
    ("the sun is bright", "suraj tez hai"),
    ("water is precious", "paani anmol hai"),
    ("study hard", "kathin mehnat karo"),
    ("never give up", "kabhi haar mat maano"),
    ("help the poor", "garibon ki madad karo"),
    ("truth always wins", "sach hamesha jeetata hai"),
    ("time is money", "samay paisa hai"),
    ("knowledge is power", "gyaan shakti hai"),
]

EN_HI_LEAD = ["", "", "", "please ", "can you ", "i want to say "]
HI_EN_LEAD = ["", "", "", "what does this mean in english: ", "translate: "]


def _translation(rng: random.Random, n: int) -> List[Dict]:
    out = []
    while len(out) < n:
        en, hi = rng.choice(PAIRS)
        if rng.random() < 0.5:  # en -> hi
            q = rng.choice(EN_HI_LEAD) + en + (" to hindi" if rng.random() < 0.5 else "")
            a = f'"{en}" ka hindi mein matlab hai "{hi.strip()}".'
        else:  # hi -> en
            q = rng.choice(HI_EN_LEAD) + hi.strip()
            a = f'"{hi.strip()}" ka english mein matlab hai "{en}".'
        out.append({"messages": [
            {"role": "user", "content": q.strip()},
            {"role": "assistant", "content": a},
        ]})
        if len(out) >= n:
            break
        out.append({"messages": [
            {"role": "user", "content": f"how do i say '{en}' in hindi?"},
            {"role": "assistant", "content": f"You can say \"{hi.strip()}\" in hindi."},
        ]})
    return out[:n]


# ── logic / riddles ───────────────────────────────────────────────────────────
RIDDLES = [
    ("what has keys but cannot open locks", "a piano"),
    ("what has hands and a face but cannot smile", "a clock"),
    ("what gets wetter the more it dries", "a towel"),
    ("what goes up but never comes down", "your age"),
    ("what has a neck but no head", "a bottle"),
    ("what can you catch but not throw", "a cold"),
    ("what belongs to you but others use it more than you", "your name"),
    ("what has many teeth but cannot bite", "a comb"),
    ("i speak without a mouth and hear without ears. what am i", "an echo"),
    ("the more you take, the more you leave behind. what are they", "footsteps"),
    ("what can fill a room but takes up no space", "light"),
    ("what is seen in the middle of march and april that cannot be seen at the start or end of either month", "the letter r"),
    ("what five-letter word becomes shorter when you add two letters to it", "short"),
    ("what has one eye but cannot see", "a needle"),
    ("what is it that once you say you will never take it back", "your word"),
]

LOGIC_Q = [
    ("if all roses are flowers and some flowers fade fast, do roses fade fast",
     "not necessarily. the statement only says some flowers fade fast - roses may or may not be among them."),
    ("a farmer has 15 cows. all but 8 die. how many are left",
     "8 are left. 'all but 8' means everything except 8."),
    ("if you have three apples and take away two, how many do you have",
     "you have 2, because you took 2."),
    ("which is heavier: a kilogram of feathers or a kilogram of steel",
     "neither - both weigh exactly one kilogram."),
    ("if a clock shows 3:00, what angle do the hour and minute hands make",
     "90 degrees. a quarter of a full circle."),
    ("there are 4 lights and 2 switches. how do you tell which switch controls which light",
     "turn one on for a while, then off, turn another on, then feel the bulbs: on now, recently warm, and cold."),
    ("if today is monday, what day is it in 100 days",
     "100 mod 7 is 2, so it is wednesday."),
    ("a bat and a ball cost 1.10 total. the bat costs 1 more than the ball. what does the ball cost",
     "5 cents. if the ball is 0.05 the bat is 1.05, and together that is 1.10."),
]


def _logic(rng: random.Random, n: int) -> List[Dict]:
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 20:
        tries += 1
        if rng.random() < 0.55:
            q, a = rng.choice(RIDDLES)
            q = rng.choice(["what am i: ", "riddle: ", "guess: ", ""]) + q + "?"
        else:
            q, a = rng.choice(LOGIC_Q)
            q = rng.choice(["think carefully: ", "", "solve this: "]) + q + "?"
        if q in seen and tries < n * 10:  # repeats allowed only after many tries
            continue
        seen.add(q)
        ans = a[0].upper() + a[1:] if not a[0].isupper() else a
        ans = ans if ans.endswith(".") else ans + "."
        out.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": ans},
        ]})
    return out


# ── creative writing ──────────────────────────────────────────────────────────
CW_THEMES = [
    ("the sea", ["the sea hums a patient song,", "waves fold and unfold the shore,", "salt hangs where the gulls argue."]),
    ("night", ["night pulls its blanket over the town,", "one lamp keeps watch for the rest of us,", "stars arrive without asking."]),
    ("rain", ["rain writes the same word on every roof,", "puddles practise holding the sky,", "the street smells newly made."]),
    ("mountains", ["mountains keep their own weather,", "snow writes in white on dark stone,", "the trail forgets you slowly."]),
    ("home", ["home is the door that knows your hand,", "a kettle announces the evening,", "chairs remember who sits where."]),
    ("time", ["time is a river that forgets its banks,", "clocks argue, the sun settles it,", "yesterday is a room we can only look into."]),
    ("dreams", ["dreams build cities and leave no maps,", "sleep rehearses what waking fears,", "morning dissolves the best ideas."]),
    ("fire", ["fire practises being a flower,", "sparks argue upward with the dark,", "ashes keep the warm arithmetic."]),
]

CW_PROMPT = ["write a short poetic line about ", "give me three lines about ", "a few words on ", ""]


def _creative(rng: random.Random, n: int) -> List[Dict]:
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 20:
        tries += 1
        theme, lines = rng.choice(CW_THEMES)
        prompt = rng.choice(CW_PROMPT)
        k = 3 if "three" in prompt else rng.choice([1, 2])
        body = "\n".join(rng.sample(lines, k))
        q = prompt + theme
        if q in seen and tries < n * 10:
            continue
        seen.add(q)
        out.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": body},
        ]})
    return out


# ── Hindi conversation ────────────────────────────────────────────────────────
HI_CONV = [
    ("namaste", "namaste! aap kaise hain? main aapki kya madad kar sakta hoon?"),
    ("aapka naam kya hai", "mera naam tantra hai. main ek chhota sa AI model hoon."),
    ("aap kahan se ho", "main atulya ai dwara banaya gaya hoon."),
    ("shukriya", "aapka swagat hai! aur bataiye."),
    ("kya aap hindi bol sakte hain", "haan, main hindi aur english dono mein baat kar sakta hoon."),
    ("mujhe thodi madad chahiye", "bilkul bataiye, main poori koshish karunga."),
    ("aaj mausam kaisa hai", "mujhe bahar ka mausam nahi dikh sakta, lekin aap bataiye toh main baat kar sakta hoon."),
    ("mera din accha hai", "bahut accha! aaj kya khaas hai?"),
    ("mujhe ek chutkula sunao", "ek chhota sa chutkula: doctor ne patient ko kaha 'aapko aaram chahiye', patient bola 'lekin doctor sahab, main toh aaram karke hi toh thaka hua hoon!'"),
    ("good night", "shubh ratri! acchi neend lein."),
    ("subah bakhair", "subah bakhair! aaj ka din shubh rahe."),
    ("main thoda udas hoon", "samajh sakta hoon. kabhi kabhi aisa hota hai. baat karne se mann halka hota hai."),
    ("kya aap insaan ho", "nahi, main ek computer program hoon, lekin main madad ke liye yahan hoon."),
    ("aapko kya karna pasand hai", "mujhe sawaal jawab karna aur aapki madad karna pasand hai."),
    ("chalo phir milte hain", "zaroor! jab bhi chahein baat karte hain. alvida!"),
]

HI_VARIANTS = ["", "", " ", "batao", "please"]


def _hindi(rng: random.Random, n: int) -> List[Dict]:
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 20:
        tries += 1
        q, a = rng.choice(HI_CONV)
        q = (q + " " + rng.choice(HI_VARIANTS)).strip()
        if q in seen and tries < n * 10:
            continue
        seen.add(q)
        out.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]})
    return out


BUILDERS = {
    ("tasks", "summarization"): _summarization,
    ("translation", "translation"): _translation,
    ("reasoning", "logic"): _logic,
    ("tasks", "creative_writing"): _creative,
    ("multilingual", "lang_hi"): _hindi,
}


def write_all(out_dir: str | Path, n: int = 400, seed: int = 0) -> Dict[str, int]:
    out_dir = Path(out_dir)
    counts: Dict[str, int] = {}
    for (domain, name), build in BUILDERS.items():
        d = out_dir / domain
        d.mkdir(parents=True, exist_ok=True)
        rows = build(random.Random(seed), n)
        with open(d / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[f"{domain}/{name}"] = len(rows)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Datasets/curated")
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    for k, v in write_all(a.out, a.n).items():
        print(f"{k}: {v} rows")
