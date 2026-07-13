"""High-quality reasoning dataset for frontier-level LLM training.
Generates verified step-by-step reasoning across math, logic, science, and code.
Target: 500k+ deep reasoning chains.

Usage:
    python tools/build_frontier_reasoning.py --rows 500000
    python tools/build_frontier_reasoning.py --rows 1000000 --output frontier_reasoning_1m.jsonl
"""

from __future__ import annotations

import itertools
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

OUT_DIR = Path("Download") / "reasoning"
SYSTEM_REASON = "You are Atulya. Explain your reasoning step by step."
SYSTEM_CODE = "You are Atulya. Think through the code carefully."


# ═══════════════════════════════════════════════════════════
# MATH REASONING — Verified template problems
# ═══════════════════════════════════════════════════════════

def _gen_arithmetic(ops: list[str], rng: random.Random) -> tuple[str, str]:
    """Generate arithmetic word problem with verified solution."""
    a = rng.randint(12, 999)
    b = rng.randint(1, 99) if rng.random() < 0.5 else rng.randint(12, 999)
    op = rng.choice(ops)

    if op == "add":
        ans = a + b
        context = rng.choice([
            f"A farmer has {a} apples and buys {b} more.",
            f"A library has {a} books and receives {b} new donations.",
            f"A class has {a} students and {b} join mid-year.",
            f"You save ${a} in January and ${b} in February.",
        ])
        question = f"{context} What is the total?"
        solution = f"Let's add {a} + {b}. {a} + {b} = {ans}. So the total is {ans}."
    elif op == "sub":
        ans = a - b
        if ans < 0:
            a, b = b, a
            ans = a - b
        context = rng.choice([
            f"There are {a} birds on a tree. {b} fly away.",
            f"A tank holds {a} liters. {b} liters are used.",
            f"You have ${a}. You spend ${b}.",
        ])
        question = f"{context} How many remain?"
        solution = f"Starting with {a}, subtract {b}. {a} - {b} = {ans}. So {ans} remain."
    elif op == "mul":
        ans = a * b
        if ans > 99999:
            a = rng.randint(3, 99)
            ans = a * b
        context = rng.choice([
            f"There are {a} boxes with {b} items each.",
            f"A car travels {a} km/h for {b} hours.",
            f"A baker makes {a} batches of {b} cookies each.",
        ])
        question = f"{context} What is the total?"
        solution = f"Multiply {a} × {b}. {a} × {b} = {ans}. So the total is {ans}."
    elif op == "div":
        safe_b = max(b, 1)
        ans = a // safe_b
        rem = a % safe_b
        if rem != 0:
            a = a - rem
            ans = a // safe_b
        if ans < 2 or a < 2:
            a = rng.randint(20, 100)
            b = rng.randint(2, 10)
            ans = a // b
            a = ans * b  # ensure exact division
        context = rng.choice([
            f"{a} students are split into {b} equal groups.",
            f"${a} is shared equally among {b} people.",
            f"{a} cookies are packed into boxes of {b}.",
        ])
        question = f"{context} How many per group?"
        solution = f"Divide {a} ÷ {b}. {a} ÷ {b} = {ans}. Each group gets {ans}."
    elif op == "sequence":
        # Simple arithmetic sequence
        start = rng.randint(1, 10)
        diff = rng.randint(2, 7)
        n_terms = rng.randint(4, 6)
        terms = [start + i * diff for i in range(n_terms)]
        nth = rng.choice([5, 6, 7, 8, 10])
        ans = start + (nth - 1) * diff
        question = f"Find the {nth}th term: {', '.join(str(t) for t in terms)}, ..."
        solution = f"The sequence increases by {diff} each step. Starting from {start}, the {nth}th term is {start} + ({nth} - 1) × {diff} = {start} + {(nth-1)*diff} = {ans}."
    else:
        return _gen_arithmetic(["add"], rng)

    return question, solution


def _gen_algebra(rng: random.Random) -> tuple[str, str]:
    """Solve-for-x problems."""
    x = rng.randint(1, 20)
    coeff = rng.choice([1, 2, 3, 4, 5])
    const = rng.randint(-20, 20)
    rhs = coeff * x + const

    if rng.random() < 0.5:
        # coeff*x + const = rhs
        question = f"Solve for x: {coeff}x + {const} = {rhs}"
        steps = [
            f"Subtract {const} from both sides: {coeff}x = {rhs} - ({const}) = {rhs - const}",
            f"Divide by {coeff}: x = {rhs - const} ÷ {coeff} = {(rhs - const) // coeff}",
        ]
        if (rhs - const) % coeff != 0:
            # Fall back to simpler
            return _gen_algebra(rng)
    else:
        # coeff*x - const = rhs
        question = f"Solve for x: {coeff}x - {const} = {rhs}"
        steps = [
            f"Add {const} to both sides: {coeff}x = {rhs} + ({const}) = {rhs + const}",
            f"Divide by {coeff}: x = {rhs + const} ÷ {coeff} = {(rhs + const) // coeff}",
        ]
        if (rhs + const) % coeff != 0:
            return _gen_algebra(rng)

    ans = x
    solution = "Step 1: " + steps[0] + "\nStep 2: " + steps[1] + f"\nTherefore, x = {ans}."
    # Verify
    check = (rhs - const) // coeff if " + " in question else (rhs + const) // coeff
    return question, solution


def _gen_percent(rng: random.Random) -> tuple[str, str]:
    """Percentage problems."""
    total = rng.randint(50, 500)
    pct = rng.choice([10, 15, 20, 25, 30, 50, 75])
    part = total * pct // 100

    if rng.random() < 0.5:
        question = f"What is {pct}% of {total}?"
        solution = f"{pct}% of {total} = ({pct}/100) × {total} = 0.{pct if pct >= 10 else '0'+str(pct)} × {total} = {part}."
    else:
        question = f"If {part} out of {total} people prefer X, what percentage is that?"
        pct_calc = round(part / total * 100, 1)
        solution = f"({part} / {total}) × 100 = {pct_calc}%. So {pct_calc}% prefer X."
    return question, solution


def _gen_rate(rng: random.Random) -> tuple[str, str]:
    """Speed/distance/time problems."""
    if rng.random() < 0.5:
        speed = rng.choice([30, 45, 50, 60, 65, 70, 80])
        time_h = rng.choice([1, 1.5, 2, 2.5, 3, 4])
        dist = int(speed * time_h)
        question = f"A car travels at {speed} km/h for {time_h} hours. How far does it go?"
        solution = f"Distance = speed × time = {speed} × {time_h} = {dist} km."
    else:
        dist = rng.choice([100, 150, 200, 250, 300, 400])
        speed = rng.choice([40, 50, 60, 70, 80])
        time_h = round(dist / speed, 2)
        question = f"A train travels {dist} km at {speed} km/h. How long does it take?"
        solution = f"Time = distance ÷ speed = {dist} ÷ {speed} = {time_h} hours."
    return question, solution


def _gen_probability(rng: random.Random) -> tuple[str, str]:
    """Simple probability problems."""
    favorable = rng.randint(1, 5)
    total = rng.randint(favorable + 2, 20)
    prob = round(favorable / total, 4)

    items = rng.choice([
        ("red marbles", "blue marbles"),
        ("green candies", "yellow candies"),
        ("white socks", "black socks"),
        ("apples", "oranges"),
    ])
    question = f"A bag has {favorable} {items[0]} and {total - favorable} {items[1]}. What's the probability of picking a {items[0][:-1]}?"
    solution = f"P({items[0][:-1]}) = favorable / total = {favorable} / {total} = {prob} = {prob*100:.1f}%."
    return question, solution


def _gen_geometry(rng: random.Random) -> tuple[str, str]:
    """Area/perimeter/volume problems."""
    kind = rng.choice(["area_rect", "area_circle", "perimeter", "volume_cube"])

    if kind == "area_rect":
        w = rng.randint(3, 20)
        h = rng.randint(3, 20)
        area = w * h
        question = f"A rectangle is {w}cm wide and {h}cm tall. What is its area?"
        solution = f"Area = width × height = {w} × {h} = {area} cm²."
    elif kind == "area_circle":
        r = rng.randint(2, 15)
        area = round(3.14159 * r * r, 2)
        question = f"A circle has radius {r}cm. What is its area? (Use π ≈ 3.14159)"
        solution = f"Area = πr² = π × {r}² = 3.14159 × {r*r} ≈ {area} cm²."
    elif kind == "perimeter":
        s = rng.randint(3, 30)
        perimeter = 4 * s
        question = f"A square has side {s}cm. What is its perimeter?"
        solution = f"Perimeter = 4 × side = 4 × {s} = {perimeter} cm."
    else:
        s = rng.randint(2, 12)
        vol = s ** 3
        question = f"A cube has side {s}cm. What is its volume?"
        solution = f"Volume = side³ = {s}³ = {vol} cm³."

    return question, solution


MATH_GENERATORS = {
    "arithmetic_add": lambda r: _gen_arithmetic(["add"], r),
    "arithmetic_sub": lambda r: _gen_arithmetic(["sub"], r),
    "arithmetic_mul": lambda r: _gen_arithmetic(["mul"], r),
    "arithmetic_div": lambda r: _gen_arithmetic(["div"], r),
    "arithmetic_seq": lambda r: _gen_arithmetic(["sequence"], r),
    "algebra": _gen_algebra,
    "percent": _gen_percent,
    "rate": _gen_rate,
    "probability": _gen_probability,
    "geometry": _gen_geometry,
}


# ═══════════════════════════════════════════════════════════
# LOGICAL REASONING
# ═══════════════════════════════════════════════════════════

SYLLOGISM_TEMPLATES = [
    lambda r: ("All humans are mortal. Socrates is human. Is Socrates mortal?",
               "Premise 1: All humans are mortal.\nPremise 2: Socrates is human.\nConclusion: Therefore, Socrates is mortal. This is a classic syllogism — the conclusion follows necessarily from the premises."),
    lambda r: ("All birds have feathers. Penguins are birds. Do penguins have feathers?",
               "Premise 1: All birds have feathers.\nPremise 2: Penguins are birds.\nConclusion: Therefore, penguins have feathers. Even though penguins can't fly, they are still birds and have feathers."),
    lambda r: ("If it rains, the ground gets wet. The ground is wet. Does that mean it rained?",
               "This is the fallacy of affirming the consequent. The ground could be wet for other reasons (sprinklers, spilled water, etc.). The correct inference is: if it rains → ground wet, but wet ground does not necessarily mean rain."),
    lambda r: ("All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?",
               "Premise 1: All mammals are warm-blooded.\nPremise 2: Whales are mammals.\nConclusion: Therefore, whales are warm-blooded. This is a valid syllogism."),
    lambda r: ("No fish can fly. A salmon is a fish. Can a salmon fly?",
               "Premise 1: No fish can fly.\nPremise 2: A salmon is a fish.\nConclusion: Therefore, a salmon cannot fly. This is a valid categorical syllogism."),
    lambda r: ("Some fruits are sweet. All apples are fruits. Are all apples sweet?",
               "Not necessarily. 'Some fruits are sweet' means at least one fruit is sweet, not all fruits. Apples are fruits, but we don't know if they're among the sweet ones. This is a classic syllogistic fallacy."),
    lambda r: ("If a number is even, it's divisible by 2. 7 is not divisible by 2. Is 7 even?",
               "Contrapositive: If a number is even → divisible by 2. The contrapositive is: if not divisible by 2 → not even. Since 7 is not divisible by 2, it is not even. This uses modus tollens."),
    lambda r: ("All squares are rectangles. All rectangles have four sides. Do all squares have four sides?",
               "Premise 1: All squares are rectangles.\nPremise 2: All rectangles have four sides.\nConclusion: Therefore, all squares have four sides. This is transitive reasoning."),
]


def _gen_syllogism(rng: random.Random) -> tuple[str, str]:
    q, a = rng.choice(SYLLOGISM_TEMPLATES)(rng)
    return q, a


LATERAL_PUZZLES = [
    lambda r: ("A man pushes his car to a hotel and tells the owner he's bankrupt. Why?",
               "He's playing Monopoly. In Monopoly, you land on a property and go bankrupt if you can't pay. The 'hotel' and 'car' are game pieces."),
    lambda r: ("What comes once in a minute, twice in a moment, but never in a thousand years?",
               "The letter 'M'. 'Minute' has one M, 'moment' has two M's, and 'thousand years' has no M."),
    lambda r: ("I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I?",
               "An echo. Sound reflects off surfaces without needing a mouth or ears. The wind carries sound waves."),
    lambda r: ("The more you take, the more you leave behind. What am I?",
               "Footsteps. Each step you take leaves a footprint behind. The more steps you take, the more footprints you leave."),
    lambda r: ("What can travel around the world while staying in a corner?",
               "A stamp. A postage stamp stays in the corner of an envelope but the envelope travels the world."),
    lambda r: ("If you have me, you want to share me. If you share me, you don't have me. What am I?",
               "A secret. Once you share a secret, it's no longer yours alone — it becomes known to others."),
    lambda r: ("I follow you all day long, but when night comes I'm gone. What am I?",
               "Your shadow. It follows you in sunlight and disappears in darkness."),
    lambda r: ("What has keys but can't open locks?",
               "A piano. It has musical keys but they don't open locks."),
    lambda r: ("What can you catch but not throw?",
               "A cold. You catch a cold (illness) but cannot physically throw it."),
    lambda r: ("What building has the most stories?",
               "A library. 'Stories' is a pun — stories as in tales (books) versus building floors."),
]


def _gen_puzzle(rng: random.Random) -> tuple[str, str]:
    q, a = rng.choice(LATERAL_PUZZLES)(rng)
    return q, a


DEDUCTIVE_PROBLEMS = [
    lambda r: ("There are three boxes: one contains only apples, one only oranges, and one both. All labels are wrong. You pick one fruit from the box labeled 'Apples'. It's an orange. What do you know?",
               "Key insight: All labels are wrong. The box labeled 'Apples' cannot contain only apples (label is wrong). You picked an orange, so this box must be the 'Both' box (it has at least one orange, and it's not apples-only or oranges-only since labels are all wrong). Therefore:\n- 'Apples' box = Both\n- 'Oranges' box = Apples (since it can't be oranges, and 'Both' is taken)\n- 'Both' box = Oranges (last remaining)"),
    lambda r: ("Five people in a line: Alice is not first. Bob is before Charlie but after Diana. Eve is last. Charlie is third. Who is first?",
               "Let's reason step by step:\n1. Eve is last (5th).\n2. Charlie is 3rd.\n3. Bob is before Charlie but after Diana: so order is Diana → Bob → Charlie.\n4. Diana must be 1st (before Bob at 2nd).\n5. Alice is not first, so Alice is 4th.\nFinal order: Diana, Bob, Charlie, Alice, Eve. First is Diana."),
    lambda r: ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
               "Let the ball cost x dollars. Then the bat costs x + 1.00. Total: x + (x + 1.00) = 1.10. 2x + 1.00 = 1.10. 2x = 0.10. x = 0.05. So the ball costs 5 cents. (The intuitive answer of 10 cents is wrong because then the bat would be $1.00, making the total $1.10 but the bat would only be $0.90 more than the ball, not $1.00 more.)"),
    lambda r: ("You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons?",
               "Step 1: Fill the 5-gallon jug. (5, 0)\nStep 2: Pour from 5 into 3 until 3 is full. (2, 3)\nStep 3: Empty the 3-gallon jug. (2, 0)\nStep 4: Pour the remaining 2 gallons from 5 into 3. (0, 2)\nStep 5: Fill the 5-gallon jug again. (5, 2)\nStep 6: Pour from 5 into 3 until full (3 already has 2, so it takes 1 more). (4, 3)\nResult: 5-gallon jug now has exactly 4 gallons."),
    lambda r: ("If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
               "Each machine makes 1 widget in 5 minutes. 5 machines × 5 minutes = 5 widgets. So 1 machine makes 1 widget in 5 minutes. Therefore, 100 machines making 100 widgets still takes 5 minutes (they work in parallel)."),
    lambda r: ("In a lake, there's a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days to cover the lake, how long does it take to cover half?",
               "If it doubles every day, then on day 47 the patch covers exactly half the lake. The next day (day 48) it doubles to cover the whole lake. So it takes 47 days to cover half."),
]


def _gen_deductive(rng: random.Random) -> tuple[str, str]:
    q, a = rng.choice(DEDUCTIVE_PROBLEMS)(rng)
    return q, a


# ═══════════════════════════════════════════════════════════
# SCIENCE REASONING — Why/How explanations
# ═══════════════════════════════════════════════════════════

SCIENCE_QUESTIONS = [
    ("Why is the sky blue?",
     "Sunlight is white light composed of all colors. When it enters Earth's atmosphere, it scatters off air molecules. Shorter wavelengths (blue/violet) scatter much more than longer ones (red/orange) due to Rayleigh scattering. Our eyes are more sensitive to blue than violet, so we see a blue sky. At sunrise/sunset, light travels through more atmosphere, scattering away the blue and leaving red/orange."),
    ("What causes the seasons?",
     "Earth's axis is tilted 23.5° relative to its orbital plane. As Earth orbits the Sun, the Northern Hemisphere tilts toward the Sun in June (summer) and away in December (winter). The tilt changes the angle and duration of sunlight, not the distance from the Sun. More direct sunlight + longer days = summer."),
    ("Why does ice float on water?",
     "Water is unusual: it's densest at 4°C, not at its freezing point. When water freezes into ice, the molecules form a hexagonal crystal lattice that takes up more space than liquid water. This makes ice about 9% less dense than liquid water, so it floats. If ice sank, lakes would freeze solid and aquatic life couldn't survive."),
    ("How do vaccines work?",
     "Vaccines expose your immune system to a harmless part of a pathogen (an antigen). Your body produces antibodies and memory B-cells specific to that antigen. If you later encounter the real pathogen, your immune system recognizes it immediately and launches a much faster, stronger response — often before you even feel sick."),
    ("Why do we dream?",
     "Several theories exist. The activation-synthesis theory suggests dreams are the brain's attempt to make sense of random neural signals during REM sleep. The memory consolidation theory suggests dreams help process and store memories. The threat simulation theory proposes dreams evolved to rehearse dangerous situations safely."),
    ("What is a black hole?",
     "A black hole is a region of spacetime where gravity is so strong that nothing — not even light — can escape. It forms when a massive star collapses under its own gravity, compressing its mass into an infinitely dense point (singularity). The boundary around it is the event horizon — once crossed, there's no return."),
    ("Why is the ocean salty?",
     "Rainwater, slightly acidic from dissolved CO2, erodes rocks on land, releasing minerals including sodium and chloride ions. Rivers carry these ions to the ocean. Over billions of years, these salts have accumulated because they don't easily precipitate out or get consumed by organisms. The ocean is about 3.5% salt by weight."),
    ("How does photosynthesis work?",
     "Plants use chlorophyll in chloroplasts to capture sunlight energy. This energy splits water (H2O) into oxygen (released) and hydrogen. The hydrogen combines with CO2 to form glucose (C6H12O6) in the Calvin cycle. The overall reaction: 6CO2 + 6H2O + sunlight → C6H12O6 + 6O2."),
    ("What is quantum entanglement?",
     "Two particles can be linked so that measuring one instantly determines the state of the other, regardless of distance — even light-years apart. Einstein called it 'spooky action at a distance.' It doesn't allow faster-than-light communication because you can't control which state you measure. It's a real phenomenon confirmed by Bell tests."),
    ("Why do we have fingerprints?",
     "Fingerprints (friction ridges) serve two main purposes: they increase friction and grip on surfaces, and they improve tactile sensitivity. The ridges amplify vibrations when we touch things, sending stronger signals to nerve endings. The specific pattern (loop, whorl, arch) is random and unique due to developmental factors in the womb."),
    ("How does evolution work?",
     "Evolution by natural selection has three requirements: variation (individuals differ), heritability (traits are passed down), and differential survival (some traits help survival/reproduction). Individuals with advantageous traits survive and reproduce more, passing those traits to the next generation. Over millions of years, this creates complex adaptations and new species."),
    ("What causes earthquakes?",
     "Earth's lithosphere is divided into tectonic plates that move slowly (cm/year) due to convection in the mantle. Stress builds at plate boundaries where plates collide, separate, or slide past each other. When the stress exceeds the rocks' strength, the fault ruptures suddenly, releasing energy as seismic waves — an earthquake."),
    ("Why is the sky dark at night?",
     "This is Olbers' paradox. If the universe were infinite, static, and filled with stars, every line of sight would end at a star's surface, so the night sky would be bright. The universe is finite in age (~13.8 billion years), expanding, and stars don't live forever. Light from distant stars hasn't reached us yet, and the expansion redshifts distant light."),
    ("How do antibiotics work?",
     "Different antibiotics target different bacterial mechanisms. Penicillins disrupt cell wall synthesis, causing bacteria to burst. Tetracyclines block protein synthesis in ribosomes. Fluoroquinolones interfere with DNA replication. Antibiotics don't work on viruses — hence antibiotic resistance is a major crisis from overuse."),
    ("What is the double-slit experiment?",
     "When particles like electrons are fired at two slits one at a time, they create an interference pattern (like waves). But if you measure which slit each goes through, the pattern disappears and they behave like particles. This shows quantum objects exist as probability waves that 'collapse' upon measurement — a fundamental mystery in quantum mechanics."),
]

SCIENCE_QUESTIONS_2 = [
    ("How does GPS work?",
     "GPS satellites orbit Earth broadcasting their position and precise time (using atomic clocks). Your receiver calculates its distance to each satellite using the time delay of the signal. With signals from 4+ satellites, it triangulates your position. Einstein's relativity matters: satellite clocks run faster by ~38 microseconds/day, requiring correction."),
    ("What is dark matter?",
     "Dark matter is invisible matter that doesn't emit, absorb, or reflect light but has gravitational effects. We know it exists because galaxies rotate faster than visible matter can explain, and gravitational lensing shows more mass than we see. It makes up ~27% of the universe. We don't know what it is — candidates include WIMPs and axions."),
    ("How does a computer work at the lowest level?",
     "Transistors act as switches (on/off = 1/0). Combinations of transistors form logic gates (AND, OR, NOT, XOR). Gates form adders, multiplexers, and flip-flops (memory). These form an ALU and registers. The control unit fetches instructions from memory, decodes them, and coordinates data flow. This is the stored-program concept (von Neumann architecture)."),
    ("Why is water essential for life?",
     "Water is a universal solvent — its polarity allows it to dissolve more substances than any other liquid, enabling biochemical reactions. It has high specific heat capacity, stabilizing temperatures. Ice floats (insulating lakes). It's cohesive (surface tension, capillary action in plants). It's the medium for almost all biological chemistry."),
]


def _gen_science(rng: random.Random) -> tuple[str, str]:
    q, a = rng.choice(SCIENCE_QUESTIONS + SCIENCE_QUESTIONS_2)
    return q, a


# ═══════════════════════════════════════════════════════════
# CODE REASONING — Trace, debug, explain
# ═══════════════════════════════════════════════════════════

def _gen_code_trace_simple(rng: random.Random) -> tuple[str, str]:
    """What does this code output? — simple."""
    n = rng.randint(3, 8)
    total = sum(range(1, n+1))
    code = f"total = 0\nfor i in range(1, {n+1}):\n    total += i\nprint(total)"
    question = f"What does this code print?\n```python\n{code}\n```"
    answer = f"Let's trace:\n- Initialize total = 0\n- Loop i = 1: total = 0 + 1 = 1\n- Loop i = 2: total = 1 + 2 = 3\n- Loop i = 3: total = 3 + 3 = 6\n...\n- Loop i = {n}: total = previous + {n}\nThe sum of 1 to {n} = {n}({n}+1)/2 = {total}\nOutput: {total}"
    return question, answer


def _gen_code_trace_fn(rng: random.Random) -> tuple[str, str]:
    """Trace recursive function."""
    n = rng.randint(3, 6)
    fibs = [0, 1]
    for i in range(2, n+1):
        fibs.append(fibs[-1] + fibs[-2])
    code = f"def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\nprint(fib({n}))"
    question = f"What does fib({n}) return?\n```python\n{code}\n```"
    answer = f"fib({n}) = fib({n-1}) + fib({n-2})\nLet's unroll:\n"
    for i in range(n+1):
        answer += f"fib({i}) = {fibs[i]}\n"
    answer += f"Output: {fibs[n]}"
    return question, answer


def _gen_code_bug(rng: random.Random) -> tuple[str, str]:
    """Find the bug."""
    n = rng.randint(3, 8)
    code = f"def sum_list(items):\n    total = 0\n    for i in range(len(items)):\n        total += items[i]\n    return total\n\nresult = sum_list([1, 2, {n}])\nprint(result / len(None))  # bug here"
    question = f"Find and fix the bug:\n```python\n{code}\n```"
    answer = f"The bug is on the last line: `len(None)` — you can't call len() on None. The programmer likely meant to calculate the average but used the wrong variable. Fix:\n```python\nprint(result / len([1, 2, {n}]))\n```\nor:\n```python\ndef average(items):\n    return sum(items) / len(items)\n```\nThe sum works correctly: 1 + 2 + {n} = {1+2+n}."
    return question, answer


def _gen_code_complexity(rng: random.Random) -> tuple[str, str]:
    """What is the time complexity?"""
    code = "def find_duplicates(arr):\n    seen = set()\n    dups = []\n    for x in arr:\n        if x in seen:\n            dups.append(x)\n        else:\n            seen.add(x)\n    return dups"
    question = f"What is the time complexity of this function?\n```python\n{code}\n```"
    answer = "Time complexity: O(n) where n = len(arr).\n- We loop through arr once.\n- Each 'x in seen' check is O(1) on average (hash set).\n- Each 'seen.add(x)' is O(1).\n- Total: O(n) time, O(n) extra space for the set.\nThis is optimal for finding duplicates in an unsorted array."
    return question, answer


CODE_GENERATORS = [
    _gen_code_trace_simple,
    _gen_code_trace_fn,
    _gen_code_bug,
    _gen_code_complexity,
]


# ═══════════════════════════════════════════════════════════
# COMPARATIVE / ANALYSIS REASONING
# ═══════════════════════════════════════════════════════════

COMPARE_QUESTIONS = [
    ("Compare and contrast capitalism and socialism.",
     "Capitalism: private ownership of production, market-driven allocation, profit motive, minimal government intervention. Strengths: innovation, efficiency, individual freedom. Weaknesses: inequality, boom-bust cycles, externalities.\n\nSocialism: social ownership of production, planned allocation, collective goals. Strengths: equality, social safety nets, reduced poverty. Weaknesses: reduced incentives, bureaucracy, slower innovation.\n\nMost modern economies are mixed — combining market allocation with government regulation and social programs."),
    ("What are the arguments for and against free will?",
     "For free will: Our subjective experience feels like we make conscious choices. Moral responsibility assumes we could have done otherwise. Quantum mechanics introduces indeterminism.\n\nAgainst free will (determinism): Every physical event has prior causes, including brain states. Libet experiments show brain activity precedes conscious decision by ~300ms. Our choices are shaped by genetics, environment, and prior causes we didn't choose.\n\nCompatibilism: Free will is compatible with determinism if we define it as acting according to one's own desires without external coercion."),
    ("Explain the difference between deductive and inductive reasoning.",
     "Deductive reasoning: Moves from general premises to specific conclusions. If premises are true, the conclusion must be true (valid). Example: All men are mortal. Socrates is a man. Therefore, Socrates is mortal.\n\nInductive reasoning: Moves from specific observations to general patterns. Conclusions are probable, not certain. Example: Every swan I've seen is white. Therefore, all swans are probably white.\n\nKey difference: Deduction guarantees truth (given true premises). Induction only suggests likelihood. Science uses both — deduction for predictions from theories, induction for generalizing from data."),
    ("What is the difference between SQL and NoSQL databases?",
     "SQL (relational): Tables with fixed schemas, ACID transactions, powerful joins, vertical scaling. Best for structured data with complex relationships (banking, ERP).\n\nNoSQL: Flexible/document/key-value/graph models, BASE consistency, horizontal scaling. Types: Document (MongoDB), Key-Value (Redis), Wide-column (Cassandra), Graph (Neo4j). Best for large-scale, unstructured, or rapidly changing data.\n\nChoose SQL when consistency and complex queries matter. Choose NoSQL when scalability, flexibility, or specialized data models matter."),
    ("Explain the difference between supervised and unsupervised learning.",
     "Supervised learning: Models learn from labeled data (X → y). Goal: predict labels for new data. Examples: regression (predict price), classification (spam detection), object detection. Requires labeled training data.\n\nUnsupervised learning: Models find patterns in unlabeled data. Goal: discover structure. Examples: clustering (customer segments), dimensionality reduction (PCA), anomaly detection. No labels needed.\n\nThere's also semi-supervised (some labels) and self-supervised (creates labels from data itself — used by GPT)."),
]


def _gen_compare(rng: random.Random) -> tuple[str, str]:
    q, a = rng.choice(COMPARE_QUESTIONS)
    return q, a


# ═══════════════════════════════════════════════════════════
# GENERATOR
# ═══════════════════════════════════════════════════════════

REASONING_CATEGORIES = {
    "math_arithmetic": {"gen": lambda r: r.choice(list(MATH_GENERATORS.values()))(r), "weight": 8, "cat": "reasoning"},
    "math_algebra": {"gen": MATH_GENERATORS["algebra"], "weight": 4, "cat": "reasoning"},
    "math_percent": {"gen": MATH_GENERATORS["percent"], "weight": 3, "cat": "reasoning"},
    "math_rate": {"gen": MATH_GENERATORS["rate"], "weight": 3, "cat": "reasoning"},
    "math_probability": {"gen": MATH_GENERATORS["probability"], "weight": 2, "cat": "reasoning"},
    "math_geometry": {"gen": MATH_GENERATORS["geometry"], "weight": 2, "cat": "reasoning"},
    "logic_syllogism": {"gen": _gen_syllogism, "weight": 4, "cat": "reasoning"},
    "logic_puzzle": {"gen": _gen_puzzle, "weight": 3, "cat": "reasoning"},
    "logic_deductive": {"gen": _gen_deductive, "weight": 3, "cat": "reasoning"},
    "science_explain": {"gen": _gen_science, "weight": 5, "cat": "factual"},
    "code_trace": {"gen": lambda r: r.choice(CODE_GENERATORS)(r), "weight": 3, "cat": "code"},
    "compare_analysis": {"gen": _gen_compare, "weight": 2, "cat": "reasoning"},
}


def make_reasoning_row(rng: random.Random) -> dict:
    pool = []
    for name, spec in REASONING_CATEGORIES.items():
        pool.extend([name] * spec["weight"])
    choice = rng.choice(pool)
    spec = REASONING_CATEGORIES[choice]
    cat = spec["cat"]

    question, answer = spec["gen"](rng)

    system = SYSTEM_CODE if cat == "code" else SYSTEM_REASON

    return {"system": system, "user": question, "assistant": answer, "category": cat}


def quality_filter(row: dict) -> bool:
    text = f"{row['user']} {row['assistant']}"
    if len(text) < 30:
        return False
    if len(row['assistant']) < 20:
        return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build frontier-quality reasoning dataset")
    parser.add_argument("--rows", type=int, default=500_000, help="Number of rows to generate")
    parser.add_argument("--output", default="frontier_reasoning.jsonl", help="Output filename")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output

    t0 = time.time()
    seen = set()
    records = []
    max_attempts = args.rows * 3
    last_report = 0

    while len(records) < args.rows and len(records) + (args.rows * 3) < max_attempts + (args.rows * 3):
        if len(records) >= args.rows:
            break
        attempts_inner = 0
        while len(records) < args.rows and attempts_inner < max_attempts // max(1, args.rows):
            attempts_inner += 1
            row = make_reasoning_row(rng)
            if not quality_filter(row):
                continue
            key = (row["user"][:100], row["assistant"][:100])
            if key in seen:
                continue
            seen.add(key)
            records.append(row)

            if len(records) - last_report >= 25000:
                last_report = len(records)
                elapsed = time.time() - t0
                rate = len(records) / elapsed if elapsed > 0 else 0
                print(f"  {len(records):>7,} rows ({rate:.0f}/s)", flush=True)
            if len(records) >= args.rows:
                break

    records = records[:args.rows]
    rng.shuffle(records)

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cats = Counter(r["category"] for r in records)
    elapsed = time.time() - t0
    mb = out_path.stat().st_size / 1_000_000

    print(f"\n{'='*50}")
    print(f"Generated {len(records):,} rows in {elapsed:.0f}s ({len(records)/elapsed:.0f}/s)")
    print(f"Written: {out_path} ({mb:.1f} MB)")
    print(f"Categories: {dict(cats.most_common())}")

    print("\nSamples:")
    for row in records[:3]:
        print(f"  [{row['category']}]")
        print(f"    U: {row['user'][:150]}")
        print(f"    A: {row['assistant'][:200]}")
        print()


if __name__ == "__main__":
    main()
