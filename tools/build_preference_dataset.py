#!/usr/bin/env python3
"""
tools/build_preference_dataset.py — Generates a curated, diverse DPO preference pairs dataset.
Covers:
  - Tantra / Atulya AI Identity & Persona
  - Python / C++ / Java / SQL Coding & Markdown formatting
  - Mathematical Step-by-Step Problem Solving
  - Direct, Concise & Safe Conversational Responses vs Rambling Repetitions
"""
import os
import json

OUTPUT_PATH = "Datasets/preference_pairs.jsonl"

CORE_PAIRS = [
    {
        "prompt": "Who are you?",
        "chosen": "I am **Tantra**, an intelligent AI assistant created and trained by **Atulya AI**. I am designed using the NeuroCore architecture to help you with coding, mathematics, science, writing, and problem-solving.",
        "rejected": "I am a language model trained by OpenAI ChatGPT. I am just a computer program on the internet with no identity."
    },
    {
        "prompt": "Who made you?",
        "chosen": "I was created and developed by **Atulya AI** as part of the Tantra-LLM NeuroCore project.",
        "rejected": "I was developed by Google and Microsoft researchers in 2020."
    },
    {
        "prompt": "Write a Python function to reverse a string.",
        "chosen": "Here is a clean Python function to reverse a string:\n\n```python\ndef reverse_string(s: str) -> str:\n    \"\"\"Return the reversed version of string s.\"\"\"\n    return s[::-1]\n\n# Example usage:\nprint(reverse_string(\"Tantra\"))  # Output: 'artnaT'\n```",
        "rejected": "def reverse string reverse string s return reverse s string reverse python reverse function reverse string is done by slicing s[::-1] and printing it."
    },
    {
        "prompt": "Write a function to check if a number is prime in Python.",
        "chosen": "Here is an efficient Python function to check for prime numbers:\n\n```python\ndef is_prime(n: int) -> bool:\n    \"\"\"Check if an integer n is prime.\"\"\"\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\n# Example:\nprint(is_prime(29))  # Output: True\n```",
        "rejected": "prime number is a number divided by 1 and itself def prime(n): if n == 2 return true if n == 3 return true if n == 4 return false if n == 5 return true."
    },
    {
        "prompt": "Create an SQL query to find all active users from a table named Users.",
        "chosen": "Here is the SQL query to retrieve all active users:\n\n```sql\nSELECT user_id, username, email, created_at \nFROM Users \nWHERE status = 'active';\n```",
        "rejected": "select table from users where active is true and users are users in table."
    },
    {
        "prompt": "What is Newton's First Law of Motion?",
        "chosen": "**Newton's First Law of Motion** (also known as the *Law of Inertia*) states that an object at rest will remain at rest, and an object in motion will continue in motion with a constant velocity, unless acted upon by an external net force.",
        "rejected": "Newton's first law of motion is motion is force and force is mass times acceleration and energy is created and destroyed in physics."
    },
    {
        "prompt": "Solve this math problem: If a bus has 40 seats and 15 people are sitting, how many empty seats are left?",
        "chosen": "To find the number of empty seats, subtract the occupied seats from total capacity:\n\n$$\\text{Empty Seats} = 40 - 15 = 25$$\n\nThere are **25 empty seats** remaining on the bus.",
        "rejected": "The bus has 40 and 15 so 40 + 15 = 55 seats empty seats is 15."
    },
    {
        "prompt": "How do you handle missing values in a Pandas DataFrame?",
        "chosen": "In pandas, you can handle missing values (`NaN`) using three primary approaches:\n\n1. **Drop Missing Values**:\n   ```python\n   df_clean = df.dropna()\n   ```\n2. **Fill with a Constant or Statistic (Mean/Median)**:\n   ```python\n   df_filled = df.fillna(df.mean())\n   ```\n3. **Forward/Backward Fill for Time Series**:\n   ```python\n   df_imputed = df.ffill()\n   ```",
        "rejected": "You can use dropna or fillna or mean or median or remove rows with pandas dataframe missing data in python."
    },
    {
        "prompt": "Explain what an API is in simple terms.",
        "chosen": "An **API** (Application Programming Interface) is a messenger that allows two different software programs to communicate and share data with each other.\n\n*Analogy:* Think of a restaurant waiter. You (the client) give your order to the waiter (the API), who delivers it to the kitchen (the server) and brings back your food (the response).",
        "rejected": "API is application programming interface API API is software interface to communicate software data computer program."
    },
    {
        "prompt": "Write a JavaScript function to find the maximum number in an array.",
        "chosen": "Here is the JavaScript function using `Math.max` and the spread operator:\n\n```javascript\nfunction findMax(arr) {\n    if (!arr || arr.length === 0) return null;\n    return Math.max(...arr);\n}\n\n// Example:\nconsole.log(findMax([3, 18, 5, 42, 9])); // Output: 42\n```",
        "rejected": "function max(arr) { var m = 0; for(var i=0;i<arr.length;i++) { m = arr[i]; } return m; }"
    }
]

def build_preference_dataset():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    all_pairs = []
    
    # Add curated pairs
    all_pairs.extend(CORE_PAIRS)
    
    # Load conversation dataset to synthesize contrastive preference pairs
    conv_path = "Datasets/conversation.jsonl"
    if os.path.exists(conv_path):
        with open(conv_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 2000:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                
                user = data.get("user")
                asst = data.get("assistant")
                system = data.get("system", "You are Tantra, a helpful, precise, and polite AI assistant created by Atulya AI.")
                if not user or not asst:
                    continue
                
                # Synthetic degraded response (simulating repetition/low-quality)
                words = asst.split()
                if len(words) > 4:
                    degraded = " ".join(words[:4] * 3) + " and " + " ".join(words[2:6]) + "..."
                else:
                    degraded = asst.lower() + " " + asst.lower()
                    
                all_pairs.append({
                    "system": system,
                    "user": user,
                    "chosen": asst,
                    "rejected": degraded
                })
                
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in all_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(all_pairs)} DPO Preference Pairs -> {OUTPUT_PATH}")

if __name__ == "__main__":
    build_preference_dataset()
