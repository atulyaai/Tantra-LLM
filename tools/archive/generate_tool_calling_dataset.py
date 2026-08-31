"""
tools/generate_tool_calling_dataset.py — Generates a high-density SFT dataset
for Tool / Function Calling (<tool_call> and <tool_result>).
"""
import os
import json
import random

os.makedirs("Datasets/tool_calling", exist_ok=True)
out_path = "Datasets/tool_calling/tool_calling.jsonl"

SYSTEM_PROMPT = """You are Tantra, a helpful AI assistant equipped with function calling capabilities.
When a user asks a question that requires exact calculation, executing code, or reading files, output a tool call using this exact XML format:
<tool_call>
{"name": "<function_name>", "arguments": {"<key>": "<value>"}}
</tool_call>
Available tools:
- calculator(expression: str) -> str: Evaluates a mathematical expression accurately.
- python_executor(code: str) -> str: Executes safe Python code and returns standard output.
- file_reader(filepath: str) -> str: Reads and returns the text content of a local file."""

SAMPLES = [
    # 1. Calculator tools
    (
        "Calculate 9482 multiplied by 387.",
        "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"9482 * 387\"}}\n</tool_call>\n<tool_result>\n3669534\n</tool_result>\n\n9482 multiplied by 387 is **3,669,534**."
    ),
    (
        "What is the square root of 104976?",
        "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"104976 ** 0.5\"}}\n</tool_call>\n<tool_result>\n324.0\n</tool_result>\n\nThe square root of 104,976 is **324**."
    ),
    (
        "What is (45 * 89) + (1200 / 25)?",
        "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"(45 * 89) + (1200 / 25)\"}}\n</tool_call>\n<tool_result>\n4053.0\n</tool_result>\n\nThe value is **4,053**."
    ),
    # 2. Python executor tools
    (
        "Write and run Python code to count how many prime numbers are between 1 and 100.",
        "<tool_call>\n{\"name\": \"python_executor\", \"arguments\": {\"code\": \"primes = [x for x in range(2, 101) if all(x % d != 0 for d in range(2, int(x**0.5) + 1))]\nprint(len(primes))\"}}\n</tool_call>\n<tool_result>\n25\n</tool_result>\n\nThere are **25 prime numbers** between 1 and 100."
    ),
    (
        "Use Python to calculate the MD5 hash of the string Tantra-2026.",
        "<tool_call>\n{\"name\": \"python_executor\", \"arguments\": {\"code\": \"import hashlib\nprint(hashlib.md5(b'Tantra-2026').hexdigest())\"}}\n</tool_call>\n<tool_result>\ne6d1209b533d36b4ff04a8e2cb5a83a0\n</tool_result>\n\nThe MD5 hash of `Tantra-2026` is `e6d1209b533d36b4ff04a8e2cb5a83a0`."
    ),
    # 3. File reader tools
    (
        "Read the contents of config.yaml on my machine.",
        "<tool_call>\n{\"name\": \"file_reader\", \"arguments\": {\"filepath\": \"config.yaml\"}}\n</tool_call>\n<tool_result>\nmodel:\n  name: Tantra-55M\n  context_len: 131072\n</tool_result>\n\nHere are the contents of `config.yaml`:\n```yaml\nmodel:\n  name: Tantra-55M\n  context_len: 131072\n```"
    ),
    (
        "Check what is inside dataset_manifest.json.",
        "<tool_call>\n{\"name\": \"file_reader\", \"arguments\": {\"filepath\": \"dataset_manifest.json\"}}\n</tool_call>\n<tool_result>\n{\"total_samples\": 2500, \"curriculum\": [\"code\", \"math\", \"science\"]}\n</tool_result>\n\nThe file contains:\n```json\n{\n  \"total_samples\": 2500,\n  \"curriculum\": [\"code\", \"math\", \"science\"]\n}\n```"
    )
]

random.seed(42)
all_samples = []
for _ in range(150):
    for u, a in SAMPLES:
        all_samples.append({
            "system": SYSTEM_PROMPT,
            "user": u,
            "assistant": a
        })

random.shuffle(all_samples)
all_samples = all_samples[:1000]

with open(out_path, "w", encoding="utf-8") as f:
    for item in all_samples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

file_kb = os.path.getsize(out_path) / 1024
print(f"[OK] Generated {len(all_samples)} Tool-Calling samples at {out_path} ({file_kb:.1f} KB)")
