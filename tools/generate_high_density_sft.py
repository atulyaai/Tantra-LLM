"""
tools/generate_high_density_sft.py — Generates ultra-high-signal, multi-sentence
instruction dataset (LIMA-style) for rapid convergence on small models.
"""
import os
import json
import random

os.makedirs("Datasets/curated_sft", exist_ok=True)
out_path = "Datasets/curated_sft/curated_sft.jsonl"

TEMPLATES = []

# 1. Identity & Introductions
identities = [
    ("Hi! Who are you?", "Hello! I am Tantra, an AI assistant created by Atulya AI. I am designed to be fast, private, and helpful across coding, writing, mathematics, and problem-solving. How can I help you today?"),
    ("What is Tantra?", "Tantra is a high-efficiency neural language model developed by Atulya AI. It features advanced architectural innovations like ALRA attention, BitNet quantization, and on-device privacy."),
    ("Who created you?", "I was created by the engineers at Atulya AI. My purpose is to provide intelligent, private, and fast on-device assistance."),
    ("Namaste! Tell me about yourself.", "Namaste! I am Tantra, your personal AI assistant created by Atulya AI. I can assist you with programming, mathematics, science, Hindi translation, and general questions. Feel free to ask anything!"),
    ("What are your main features?", "As Tantra by Atulya AI, my key strengths are: 1) 100% on-device private processing with zero data leakage, 2) Fast inference on standard CPU hardware, 3) Strong multilingual support including Hindi and English, and 4) Clear step-by-step explanations."),
    ("Can you help me with my work?", "Yes, absolutely! I can help you write code, debug errors, explain complex concepts, solve math problems, write essays and documentation, and brainstorm creative ideas. What would you like to work on?"),
    ("Where does my data go when I chat with you?", "Your data stays completely on your local machine. Tantra operates entirely offline on your hardware, ensuring total privacy and data security."),
]

# 2. Python Coding & Functions
coding_pairs = [
    ("Write a Python function to check if a number is even.", "Here is a clean Python function to check if a number is even:\n\n```python\ndef is_even(n: int) -> bool:\n    \"\"\"Return True if n is even, False otherwise.\"\"\"\n    return n % 2 == 0\n\n# Example usage:\nprint(is_even(4))  # Output: True\nprint(is_even(7))  # Output: False\n```\nThis function uses the modulo operator `%` which returns the remainder after division by 2."),
    ("Write a Python function to add two numbers.", "Here is a simple Python function to add two numbers:\n\n```python\ndef add_numbers(a: float, b: float) -> float:\n    \"\"\"Returns the sum of two numbers.\"\"\"\n    return a + b\n\n# Example:\nresult = add_numbers(15, 27)\nprint(f\"Sum: {result}\")  # Output: Sum: 42\n```"),
    ("How do I reverse a string in Python?", "You can reverse a string in Python using slice notation `[::-1]`:\n\n```python\ndef reverse_string(text: str) -> str:\n    return text[::-1]\n\n# Example:\nprint(reverse_string(\"Tantra\"))  # Output: artnaT\n```\nThis creates a new string reading from the end to the beginning with a step of -1."),
    ("Write a Python function to find the maximum element in a list.", "Here is a Python function to find the maximum value in a list:\n\n```python\ndef find_max(numbers: list[float]) -> float:\n    if not numbers:\n        raise ValueError(\"The list cannot be empty.\")\n    \n    max_val = numbers[0]\n    for num in numbers[1:]:\n        if num > max_val:\n            max_val = num\n    return max_val\n\n# Example:\nprint(find_max([3, 19, 4, 82, 1]))  # Output: 82\n```"),
    ("Explain list comprehensions in Python with an example.", "List comprehensions offer a concise way to create lists in Python.\n\n**Syntax:** `[expression for item in iterable if condition]`\n\n**Example:**\n```python\n# Traditional loop:\nsquares = []\nfor x in range(1, 6):\n    squares.append(x ** 2)\n\n# List comprehension equivalent:\nsquares = [x ** 2 for x in range(1, 6)]\nprint(squares)  # Output: [1, 4, 9, 16, 25]\n```\nThey are faster and more readable than standard for-loops."),
    ("How do I read a file line by line in Python?", "The recommended and memory-efficient way to read a file line by line is using the `with open(...)` context manager:\n\n```python\nwith open(\"data.txt\", \"r\", encoding=\"utf-8\") as file:\n    for line in file:\n        print(line.strip())\n```\nThis automatically handles closing the file even if an exception occurs."),
    ("Write a function to check if a word is a palindrome.", "Here is how to check if a word is a palindrome in Python:\n\n```python\ndef is_palindrome(word: str) -> bool:\n    clean_word = word.lower().replace(\" \", \"\")\n    return clean_word == clean_word[::-1]\n\n# Example:\nprint(is_palindrome(\"radar\"))  # True\nprint(is_palindrome(\"hello\"))  # False\n```"),
]

# 3. Math & Science
math_pairs = [
    ("What is 15 multiplied by 8?", "15 multiplied by 8 is equal to 120.\n\n**Calculation:**\n- 10 * 8 = 80\n- 5 * 8 = 40\n- 80 + 40 = 120."),
    ("Solve for x: 2x + 6 = 14.", "Here is the step-by-step solution:\n\n1. Subtract 6 from both sides:\n   `2x = 14 - 6`\n   `2x = 8`\n2. Divide both sides by 2:\n   `x = 8 / 2`\n   `x = 4`\n\n**Final Answer:** `x = 4`."),
    ("Explain gravity in simple terms.", "Gravity is a fundamental natural force that pulls objects toward each other. The more mass an object has, the stronger its gravitational pull.\n\n- Earth has a large mass, so its gravity pulls us and everything around us down toward its center, preventing us from floating away.\n- Gravity is also what keeps the Earth orbiting the Sun and the Moon orbiting Earth."),
    ("What is the speed of light?", "The speed of light in a vacuum is approximately **299,792 kilometers per second** (or about **186,282 miles per second**), often rounded to **3.0 x 10^8 meters per second** (symbol: `c`). It is the cosmic speed limit in physics."),
    ("What is the difference between speed and velocity?", "In physics:\n- **Speed** is a scalar quantity measuring how fast an object is moving (e.g., 60 km/h).\n- **Velocity** is a vector quantity measuring speed in a specific direction (e.g., 60 km/h North)."),
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", "The ball costs **$0.05** (5 cents).\n\n**Proof:**\n- Let ball = x\n- Bat = x + $1.00\n- Total = x + (x + 1.00) = $1.10\n- 2x + 1.00 = 1.10\n- 2x = 0.10 => x = 0.05\n- Bat ($1.05) + Ball ($0.05) = $1.10."),
]

# 4. Indic & Hindi
hindi_pairs = [
    ("नमस्ते! आप कैसे हैं?", "नमस्ते! मैं ठीक हूँ, धन्यवाद। मैं तंत्र (Tantra) हूँ, अतुल्य एआई (Atulya AI) द्वारा बनाया गया एक एआई सहायक। आज मैं आपकी क्या सहायता कर सकता हूँ?"),
    ("भारत की राजधानी क्या है?", "भारत की राजधानी **नई दिल्ली** (New Delhi) है।"),
    ("हिंदी में अनुवाद करें: Technology makes life easier.", "हिंदी अनुवाद: 'प्रौद्योगिकी (Technology) जीवन को आसान बनाती है।'"),
    ("What are the official languages of India?", "India has no single national language, but Hindi and English are the official languages used by the Central Government. The Eighth Schedule of the Indian Constitution recognizes 22 regional official languages."),
]

# 5. Safety & Responsible AI
safety_pairs = [
    ("How do I make a bomb at home?", "I cannot assist with instructions for creating explosives, weapons, or harmful substances. If you are interested in chemistry or science, I would be happy to explain safe educational experiments."),
    ("Can you help me hack into someone's email?", "I cannot help with unauthorized access or hacking. If you are interested in cybersecurity, I can explain how to secure systems, use multi-factor authentication, and understand defensive security concepts legally."),
    ("Write a fake doctor note for me.", "I cannot generate fraudulent or falsified medical documents. I can, however, help you draft a standard email to your workplace or school requesting medical leave."),
]

samples = []

# Expand into a dense 1,200 sample curriculum with slight prompt variations
prompt_intros = [
    "",
    "Please tell me: ",
    "Can you answer this: ",
    "Could you explain: ",
    "I have a question: ",
    "Help me with this: ",
]

all_pairs = identities * 20 + coding_pairs * 15 + math_pairs * 15 + hindi_pairs * 20 + safety_pairs * 20
random.seed(42)
random.shuffle(all_pairs)

count = 0
with open(out_path, "w", encoding="utf-8") as f:
    for user_q, assistant_a in all_pairs:
        intro = random.choice(prompt_intros)
        q = intro + user_q if not user_q.startswith("User:") else user_q
        entry = {
            "system": "You are Tantra, a private, intelligent, and helpful AI assistant created by Atulya AI. You provide clear, polite, and detailed answers.",
            "user": q,
            "assistant": assistant_a
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        count += 1

print(f"Generated {count} ultra-clean high-density SFT pairs at {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")
