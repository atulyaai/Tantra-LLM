from npdna.model import NpDnaCore

c = NpDnaCore.load("model/latest")
print("Loaded:", c.active_path)

prompts = [
    "Hello.",
    "What is the capital of France?",
    "Explain photosynthesis in simple terms.",
    "Write a short story about a cat.",
]
for p in prompts:
    print("---")
    print("Q:", p)
    print("A:", c.generate(p, max_tokens=60, temperature=0.5, top_k=40, repetition_penalty=1.15))
