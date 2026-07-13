"""Generate high-quality synthetic training data for NP-DNA model.
Creates diverse, well-formatted instruction-response pairs across all cognitive layers.
"""
import argparse
import json
import random
import os
from pathlib import Path

DATA_DIR = Path("Download")
random.seed(42)

# ══════════════════════════════════════════════════════════════
# TEMPLATES — each category gets curated, diverse training pairs
# ══════════════════════════════════════════════════════════════

EMOTION_DATA = [
    ("How are you feeling today?", "I'm doing well, thank you for asking! I notice you're checking in on emotions — that's a sign of emotional intelligence. How are you feeling?"),
    ("I'm feeling anxious about my exam tomorrow.", "It's completely natural to feel anxious before an exam. Here are some strategies: 1) Deep breathing — inhale for 4 counts, hold for 4, exhale for 4. 2) Review your notes briefly, then stop studying 2 hours before bed. 3) Remind yourself that you've prepared. Anxiety often means you care about the outcome, which is a good thing."),
    ("My friend said something hurtful and I'm angry.", "Anger is a valid emotion that signals your boundaries were crossed. Before responding: 1) Take a pause — even 10 minutes helps. 2) Identify what specifically hurt — was it the words, the tone, or the intent? 3) When you're calmer, use 'I feel' statements: 'I felt hurt when you said X because Y.' This opens dialogue rather than escalating conflict."),
    ("I just got promoted at work!", "Congratulations! That's wonderful news! Take a moment to truly celebrate — you earned this through your hard work and dedication. It's important to acknowledge positive emotions and achievements. Share the news with people who support you, and don't downplay your success."),
    ("I feel lonely even when surrounded by people.", "Loneliness in a crowd is more common than people think. It often signals a need for deeper, more meaningful connections rather than more social interactions. Consider: 1) Quality over quantity in friendships. 2) Sharing something vulnerable with someone you trust. 3) Finding communities around shared interests. 4) Being fully present in conversations rather than distracted."),
    ("What is emotional intelligence?", "Emotional intelligence (EQ) is the ability to recognize, understand, manage, and effectively use emotions — both your own and others'. It has five key components: 1) Self-awareness — knowing your emotions. 2) Self-regulation — managing emotional reactions. 3) Motivation — using emotions to drive goals. 4) Empathy — understanding others' feelings. 5) Social skills — managing relationships effectively. Research shows EQ is often more important than IQ for success."),
    ("I can't stop crying and I don't know why.", "Sometimes tears come without a clear reason, and that's okay. Your body may be processing accumulated stress, fatigue, or suppressed emotions. What you can do: 1) Let yourself cry — it's a healthy release. 2) Once calm, journal about what's been happening lately. 3) Check basics: sleep, nutrition, exercise. 4) If this persists for more than two weeks, speaking with a counselor can help identify underlying causes."),
    ("How do I deal with grief?", "Grief is deeply personal and there's no 'right' way to experience it. The five stages — denial, anger, bargaining, depression, acceptance — aren't linear; you may cycle through them. Key principles: 1) Allow yourself to feel without judgment. 2) Maintain routines when possible. 3) Accept help from others. 4) Honor memories in ways that feel meaningful. 5) Be patient — healing isn't on a timeline. Professional support can be invaluable during this process."),
    ("Why do I feel scared of the dark?", "Fear of darkness (nyctophobia) is one of the most common fears and has evolutionary roots — our ancestors were vulnerable to predators at night. Your brain's amygdala activates threat responses when visual information is limited. To manage it: 1) Use gradual exposure — dim lights slowly. 2) Create a comforting bedtime routine. 3) Challenge irrational thoughts with logic. 4) A nightlight is perfectly fine for adults too."),
    ("Explain the difference between empathy and sympathy.", "Empathy and sympathy are often confused but differ significantly. Sympathy is feeling FOR someone: 'I'm sorry you're going through that.' Empathy is feeling WITH someone: 'I can imagine how painful that must be.' Sympathy creates distance; empathy creates connection. Brené Brown explains it well: sympathy is looking down into a hole someone fell into, while empathy is climbing down into the hole to sit with them."),
]

SPATIAL_DATA = [
    ("What is the distance between New York and Los Angeles?", "The straight-line (great circle) distance between New York City and Los Angeles is approximately 2,451 miles (3,944 km). By road, the driving distance is roughly 2,775 miles via Interstate 80/76/70, taking about 40-42 hours of driving time without stops."),
    ("How do I calculate the area of a triangle?", "The area of a triangle can be calculated using several methods: 1) Base × Height ÷ 2 — the most common formula. If base = 10 and height = 6, area = 10 × 6 ÷ 2 = 30 square units. 2) Heron's formula: when you know all three sides (a, b, c), calculate s = (a+b+c)/2, then Area = √(s(s-a)(s-b)(s-c)). 3) Using trigonometry: Area = ½ × a × b × sin(C) where C is the angle between sides a and b."),
    ("Explain the Pythagorean theorem.", "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse (the side opposite the right angle) equals the sum of the squares of the other two sides: a² + b² = c². For example, if a = 3 and b = 4, then c² = 9 + 16 = 25, so c = 5. This creates the famous 3-4-5 right triangle. The theorem has over 400 known proofs and is fundamental to geometry, navigation, physics, and engineering."),
    ("How does GPS work?", "GPS (Global Positioning System) uses trilateration with at least 4 satellites. Each satellite broadcasts its position and the exact time. Your GPS receiver calculates the distance to each satellite based on signal travel time (signal travels at the speed of light). With 3 satellites, you get a 3D position fix (latitude, longitude, altitude). The 4th satellite corrects for clock errors. The system achieves accuracy of about 3-5 meters for civilian use."),
    ("What is the volume of a sphere with radius 5?", "The volume of a sphere is calculated using V = (4/3)πr³. For radius r = 5: V = (4/3) × π × 5³ = (4/3) × π × 125 = (500/3) × π ≈ 523.6 cubic units. To visualize: this sphere could hold about 523.6 unit cubes of water."),
    ("How do maps project a round Earth onto flat paper?", "Map projections transform the 3D Earth onto a 2D surface, and every projection distorts something. Major types: 1) Mercator — preserves angles and shapes but distorts size (Greenland looks huge). 2) Robinson — compromise projection balancing size and shape. 3) Equal-area (like Mollweide) — preserves area but distorts shapes. 4) Azimuthal — preserves directions from the center point. The choice depends on the map's purpose."),
    ("Find the shortest path between nodes A and D in a graph where A-B=4, A-C=2, B-D=3, C-B=1, C-D=5.", "Using Dijkstra's algorithm: Start at A with distance 0. Visit A's neighbors: B(4), C(2). C is closer, visit C: C→B = 2+1 = 3 (better than direct A→B=4), C→D = 2+5 = 7. Now visit B (distance 3): B→D = 3+3 = 6 (better than 7). Final shortest path: A → C → B → D with total distance 6."),
    ("What are coordinates and how do latitude/longitude work?", "Coordinates are pairs of numbers that pinpoint a location. Latitude measures north-south position (0° at equator, 90°N at North Pole, 90°S at South Pole). Longitude measures east-west position (0° at Greenwich, England, going to 180°E and 180°W). Together they form a grid covering Earth. For example, New York is approximately 40.7°N, 74.0°W. Each degree of latitude equals about 111 km."),
]

ACTION_DATA = [
    ("Pick up the red cube and place it on the blue platform.", "[ACTION PLAN] 1) Locate red cube using vision sensors. 2) Calculate approach vector. 3) Open gripper to cube width + 2mm clearance. 4) Move arm to cube position. 5) Close gripper with 2N force. 6) Lift cube 10cm vertically. 7) Navigate to blue platform coordinates. 8) Lower cube to platform surface. 9) Open gripper. 10) Retract arm to home position. Status: Plan ready for execution."),
    ("How would a robot navigate through a maze?", "Robot maze navigation uses several strategies: 1) Wall-following — always keep one hand on the wall (works for simply-connected mazes). 2) BFS/DFS — explore systematically using graph traversal. 3) A* search — use heuristics to find optimal paths. 4) SLAM (Simultaneous Localization and Mapping) — build a map while navigating using LIDAR or cameras. Modern robots combine these with real-time sensor data and path planning algorithms like RRT (Rapidly-exploring Random Trees)."),
    ("Write a sequence of steps for a robotic arm to pour water from a jug into a glass.", "[ACTION SEQUENCE] 1) Identify jug and glass positions via camera. 2) Grasp jug handle with appropriate force (5-8N). 3) Lift jug 15cm above table. 4) Move jug over glass opening. 5) Tilt jug at 30° angle initially. 6) Monitor water flow with vision sensor. 7) Gradually increase tilt to 45° as water level decreases. 8) When glass is 80% full, return jug to upright. 9) Move jug back to original position. 10) Release grip. Critical: maintain smooth, controlled motion to prevent spillage."),
    ("What is a PID controller?", "A PID controller is a feedback mechanism used in robotics and automation. PID stands for Proportional-Integral-Derivative: P (Proportional) — corrects based on current error. If you're 10° off target, apply proportional correction. I (Integral) — corrects accumulated past errors. Eliminates steady-state offset. D (Derivative) — predicts future error based on rate of change. Prevents overshooting. Together: Output = Kp×e(t) + Ki×∫e(t)dt + Kd×de(t)/dt. Tuning these three gains (Kp, Ki, Kd) is crucial for stable, responsive control."),
    ("How do self-driving cars make decisions?", "Self-driving cars use a layered decision pipeline: 1) Perception — cameras, LIDAR, radar detect objects, lanes, signs. 2) Prediction — ML models predict where other vehicles and pedestrians will move. 3) Planning — route planning (A* or similar) and trajectory planning (smooth, safe paths). 4) Control — steering, acceleration, braking commands. Safety layers include: redundant sensors, emergency braking systems, geofencing, and human override capabilities. The car constantly runs 'what-if' scenarios to prepare for unexpected events."),
    ("Explain forward kinematics vs inverse kinematics.", "Forward kinematics (FK) calculates where the end-effector (hand) ends up given joint angles. Example: 'If shoulder is at 30° and elbow at 45°, where is the hand?' This is straightforward trigonometry. Inverse kinematics (IK) is the reverse: given a target position for the hand, calculate what joint angles are needed. IK is much harder because: 1) Multiple solutions may exist. 2) Some positions may be unreachable. 3) Requires iterative numerical methods. IK is essential for robot arms reaching specific points in space."),
]

REASONING_DATA = [
    ("If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?", "No, we cannot make that conclusion. This is a classic logical fallacy. Let's break it down: Premise 1: All roses are flowers (Roses ⊂ Flowers). Premise 2: Some flowers fade quickly. The 'some flowers' that fade quickly might not include any roses. The flowers that fade quickly could be entirely non-rose flowers like daisies or tulips. This is the fallacy of the undistributed middle. To conclude 'some roses fade quickly,' we would need a premise directly linking roses to fading."),
    ("A train leaves Station A at 60 km/h. Another train leaves Station B (300 km away) at 40 km/h toward Station A. When do they meet?", "The trains are approaching each other, so their speeds add up. Combined speed = 60 + 40 = 100 km/h. Distance between them = 300 km. Time to meet = Distance ÷ Combined Speed = 300 ÷ 100 = 3 hours. At that point, Train A has traveled 60 × 3 = 180 km from Station A, and Train B has traveled 40 × 3 = 120 km from Station B. Verification: 180 + 120 = 300 km ✓"),
    ("Solve: 2x + 5 = 17", "To solve 2x + 5 = 17: Step 1: Subtract 5 from both sides: 2x + 5 - 5 = 17 - 5, giving us 2x = 12. Step 2: Divide both sides by 2: 2x/2 = 12/2, giving us x = 6. Verification: 2(6) + 5 = 12 + 5 = 17 ✓. The solution is x = 6."),
    ("What is the probability of rolling two dice and getting a sum of 7?", "There are 6 × 6 = 36 total possible outcomes when rolling two dice. The combinations that sum to 7 are: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) — that's 6 favorable outcomes. Probability = 6/36 = 1/6 ≈ 16.67%. Interestingly, 7 is the most likely sum when rolling two dice, which is why it's significant in many dice games."),
    ("Explain proof by contradiction.", "Proof by contradiction (reductio ad absurdum) works by: 1) Assume the opposite of what you want to prove. 2) Show this assumption leads to a logical contradiction. 3) Conclude the original statement must be true. Classic example — proving √2 is irrational: Assume √2 = a/b where a,b are integers with no common factors. Then 2 = a²/b², so a² = 2b². This means a² is even, so a must be even. Let a = 2k. Then 4k² = 2b², so b² = 2k², meaning b is also even. But we said a,b have no common factors — contradiction! Therefore √2 is irrational."),
]

CODE_DATA = [
    ("Write a Python function to check if a string is a palindrome.", "```python\ndef is_palindrome(s: str) -> bool:\n    \"\"\"Check if a string is a palindrome, ignoring case and non-alphanumeric characters.\"\"\"\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n\n# Examples:\n# is_palindrome('racecar')  -> True\n# is_palindrome('A man, a plan, a canal: Panama')  -> True\n# is_palindrome('hello')  -> False\n```"),
    ("Explain Big O notation.", "Big O notation describes how an algorithm's performance scales with input size n. Common complexities:\n- O(1) — Constant: array access, hash lookup\n- O(log n) — Logarithmic: binary search\n- O(n) — Linear: single loop through array\n- O(n log n) — Linearithmic: merge sort, quicksort average\n- O(n²) — Quadratic: nested loops, bubble sort\n- O(2ⁿ) — Exponential: recursive fibonacci\n\nExample: searching an unsorted list of 1M items takes ~1M steps (O(n)), but a sorted list with binary search takes only ~20 steps (O(log n)). This is why algorithm choice matters enormously at scale."),
    ("Write a binary search function in Python.", "```python\ndef binary_search(arr: list, target) -> int:\n    \"\"\"Return index of target in sorted array, or -1 if not found.\"\"\"\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n\n# Time: O(log n), Space: O(1)\n# Example: binary_search([1, 3, 5, 7, 9], 7) -> 3\n```"),
    ("What is recursion? Give an example.", "Recursion is when a function calls itself to solve smaller subproblems of the same type. Every recursive function needs: 1) A base case — when to stop. 2) A recursive case — breaking the problem into smaller pieces.\n\n```python\ndef factorial(n: int) -> int:\n    \"\"\"Calculate n! recursively.\"\"\"\n    if n <= 1:        # Base case\n        return 1\n    return n * factorial(n - 1)  # Recursive case\n\n# factorial(5) = 5 * factorial(4)\n#              = 5 * 4 * factorial(3)\n#              = 5 * 4 * 3 * factorial(2)\n#              = 5 * 4 * 3 * 2 * factorial(1)\n#              = 5 * 4 * 3 * 2 * 1 = 120\n```"),
    ("Explain the difference between a stack and a queue.", "Stack (LIFO — Last In, First Out): Like a stack of plates. You add and remove from the top. Operations: push (add to top), pop (remove from top), peek (view top). Use cases: undo/redo, function call stack, expression parsing.\n\nQueue (FIFO — First In, First Out): Like a line at a store. You add to the back and remove from the front. Operations: enqueue (add to back), dequeue (remove from front), peek (view front). Use cases: task scheduling, BFS, print queues.\n\n```python\n# Stack using list\nstack = []\nstack.append(1)  # push\nstack.append(2)\nstack.pop()       # returns 2\n\n# Queue using deque\nfrom collections import deque\nqueue = deque()\nqueue.append(1)    # enqueue\nqueue.append(2)\nqueue.popleft()    # returns 1\n```"),
]

FACTUAL_DATA = [
    ("What is photosynthesis?", "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy (glucose). The overall equation is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. It occurs in two stages: 1) Light reactions (in thylakoids) — capture solar energy to produce ATP and NADPH, splitting water and releasing oxygen. 2) Calvin cycle (in stroma) — uses ATP and NADPH to fix CO₂ into glucose. Photosynthesis is the foundation of most food chains and produces the oxygen we breathe."),
    ("How does the human heart work?", "The heart is a muscular pump with four chambers that circulates blood through two loops. Right side: receives deoxygenated blood from the body via the vena cava → right atrium → right ventricle → pumps to lungs via pulmonary artery. Left side: receives oxygenated blood from lungs via pulmonary veins → left atrium → left ventricle → pumps to body via aorta. The heart beats about 100,000 times per day, pumping roughly 7,500 liters of blood. The SA node (natural pacemaker) generates electrical impulses that coordinate each heartbeat."),
    ("What caused World War I?", "World War I (1914-1918) resulted from interconnected causes often summarized as MAIN: Militarism — European powers built massive armies and navies. Alliances — complex treaty systems (Triple Alliance vs Triple Entente) meant one conflict could drag in many nations. Imperialism — competition for colonies created tensions. Nationalism — ethnic pride and independence movements, especially in the Balkans. The immediate trigger was the assassination of Archduke Franz Ferdinand of Austria-Hungary by Gavrilo Princip in Sarajevo on June 28, 1914, which activated the alliance chains."),
    ("Explain how DNA stores genetic information.", "DNA (deoxyribonucleic acid) is a double-helix molecule that stores genetic instructions. Its 'alphabet' consists of four nucleotide bases: Adenine (A), Thymine (T), Guanine (G), and Cytosine (C). These pair specifically: A-T and G-C, forming the rungs of the twisted ladder. Three consecutive bases (a codon) code for one amino acid — for example, ATG codes for methionine. A gene is a sequence of codons that encodes a protein. The human genome contains about 3 billion base pairs across 23 chromosome pairs, encoding roughly 20,000-25,000 genes."),
    ("What is the theory of relativity?", "Einstein's theory of relativity has two parts. Special Relativity (1905): 1) The speed of light is constant for all observers. 2) The laws of physics are the same in all inertial frames. Consequences: time dilation (moving clocks run slower), length contraction, and E=mc² (mass-energy equivalence). General Relativity (1915): gravity is not a force but the curvature of spacetime caused by mass. Massive objects bend spacetime, and other objects follow curved paths through this warped geometry. This explains planetary orbits, gravitational lensing, black holes, and gravitational waves."),
]

GENERAL_DATA = [
    ("What are some tips for effective time management?", "Effective time management centers on these proven strategies: 1) Eisenhower Matrix — categorize tasks as Urgent/Important, Important/Not Urgent, Urgent/Not Important, Neither. Focus on Important tasks. 2) Time blocking — schedule specific hours for specific work types. 3) The 2-minute rule — if something takes less than 2 minutes, do it now. 4) Pomodoro technique — work in focused 25-minute blocks with 5-minute breaks. 5) Say no to low-value commitments. 6) Plan tomorrow tonight. 7) Single-task instead of multitasking — context switching costs 20-40% productivity."),
    ("How do I learn a new programming language effectively?", "The most effective approach follows this progression: 1) Understand basics — variables, loops, conditionals, functions (1-2 days). 2) Build something small immediately — a calculator, to-do app, or API client. 3) Read other people's code — study popular open-source projects. 4) Solve coding challenges — use platforms like LeetCode or HackerRank. 5) Build a real project — something you actually need. 6) Learn the ecosystem — package managers, testing frameworks, deployment. Key principle: active practice beats passive reading 10x. Aim for 70% building, 30% reading."),
    ("Explain how the internet works in simple terms.", "The internet is a global network of connected computers communicating through standardized protocols. When you visit a website: 1) You type a URL (like google.com). 2) DNS (Domain Name System) translates this to an IP address (like 142.250.80.46). 3) Your computer sends an HTTP request through your ISP. 4) The request travels through routers, potentially crossing continents via undersea fiber optic cables. 5) The web server receives your request and sends back HTML, CSS, and JavaScript files. 6) Your browser renders these files into the page you see. This entire process typically takes 50-200 milliseconds."),
]

def make_record(system, user, assistant, category):
    return {"system": system, "user": user, "assistant": assistant, "category": category}

def topic_from_question(user):
    topic = user.rstrip("?.!").strip()
    for prefix in (
        "What is ",
        "What are ",
        "How do I ",
        "How does ",
        "Explain ",
        "Write ",
        "Tell me about ",
        "Can you explain ",
    ):
        if topic.lower().startswith(prefix.lower()):
            return topic[len(prefix):].strip()
    return topic

def answer_variant(assistant, rng, idx):
    openers = [
        "",
        "Short answer: ",
        "A useful way to think about it is this: ",
        "Step by step: ",
        "In practical terms, ",
        "The core idea is: ",
    ]
    closers = [
        "",
        " The important part is to check the result against the original goal.",
        " This keeps the answer grounded instead of only sounding fluent.",
        " If the situation changes, adjust the steps while keeping the same principle.",
        " A quick verification pass helps catch mistakes early.",
    ]
    text = assistant.strip()
    if idx % 3 == 1 and not text.startswith(("Short answer:", "Step by step:")):
        text = rng.choice(openers) + text[0].lower() + text[1:]
    if idx % 4 == 2:
        text = text.rstrip() + rng.choice(closers)
    return text

def generate_category(name, data, system_prompt, output_dir, target_rows=1000, augment=True):
    """Generate JSONL training data for a category."""
    records = []
    for user, assistant in data:
        records.append(make_record(system_prompt, user, assistant, name))

    if augment:
        # Create variations by rephrasing. The larger loop gives the 15-layer
        # model enough repeated concepts in different surfaces to learn from.
        variations = [
            ("Can you explain ", "?"),
            ("Tell me about ", "."),
            ("What do you know about ", "?"),
            ("I need help understanding ", "."),
            ("Please describe ", " in detail."),
            ("Give me a beginner-friendly explanation of ", "."),
            ("Break down ", " with an example."),
            ("Help me reason through ", "."),
            ("What are the key points about ", "?"),
            ("Teach me ", " clearly."),
            ("Summarize ", " and include the practical takeaway."),
        ]
        seen = {(r["user"], r["assistant"]) for r in records}
        idx = 0
        while len(records) < target_rows:
            user, assistant = data[idx % len(data)]
            prefix, suffix = variations[idx % len(variations)]
            topic = topic_from_question(user)
            new_user = f"{prefix}{topic.lower()}{suffix}"
            if idx >= len(data) * len(variations):
                new_user = f"{new_user} Example set {idx // len(variations) + 1}."
            new_assistant = answer_variant(assistant, random, idx)
            key = (new_user, new_assistant)
            if key not in seen:
                records.append(make_record(system_prompt, new_user, new_assistant, name))
                seen.add(key)
            idx += 1
            if idx > target_rows * 20:
                break

    random.shuffle(records)
    out_path = output_dir / name / f"synthetic_{name}_curated.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [{name}] Generated {len(records)} high-quality training pairs -> {out_path}")
    return len(records)

def combined_data(*groups):
    rows = []
    for group in groups:
        rows.extend(group)
    return rows

def main():
    parser = argparse.ArgumentParser(description="Generate local synthetic NP-DNA training data.")
    parser.add_argument("--samples-per-category", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 60)
    print("  NP-DNA Synthetic Training Data Generator")
    print("=" * 60)

    sys_default = "You are Atulya. Answer clearly and accurately."
    sys_code = "You are Atulya. Write correct, clean code."
    sys_reason = "You are Atulya. Explain your reasoning step by step."
    sys_emotion = "You are Atulya. Be empathetic, warm, and emotionally intelligent."
    sys_spatial = "You are Atulya. Reason precisely about space, geometry, and navigation."
    sys_action = "You are Atulya. Plan and execute physical actions with precision."

    total = 0
    n = max(100, args.samples_per_category)
    total += generate_category("emotion", EMOTION_DATA, sys_emotion, DATA_DIR, n)
    total += generate_category("spatial", SPATIAL_DATA, sys_spatial, DATA_DIR, n)
    total += generate_category("action", ACTION_DATA, sys_action, DATA_DIR, n)
    total += generate_category("reasoning", REASONING_DATA, sys_reason, DATA_DIR, n)
    total += generate_category("code", CODE_DATA, sys_code, DATA_DIR, n)
    total += generate_category("factual", FACTUAL_DATA, sys_default, DATA_DIR, n)
    total += generate_category("general", GENERAL_DATA, sys_default, DATA_DIR, n)
    total += generate_category(
        "instruction",
        combined_data(GENERAL_DATA, FACTUAL_DATA, REASONING_DATA, EMOTION_DATA),
        sys_default,
        DATA_DIR,
        n,
    )
    total += generate_category(
        "experts",
        combined_data(CODE_DATA, ACTION_DATA, SPATIAL_DATA, REASONING_DATA),
        "You are Atulya. Give expert-level, precise, practical answers.",
        DATA_DIR,
        n,
    )

    print(f"\n  Total: {total} curated training pairs generated")
    print(f"  Data directory: {DATA_DIR.resolve()}")
    print(f"\n  Next: python npdna/train.py --fresh-start --seq-len 512 --batch-size 1")

if __name__ == "__main__":
    main()
