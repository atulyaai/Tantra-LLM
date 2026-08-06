"""Direct Weight Injection (Fast Fact & Conversation Consolidation into Model Weights).

Feeds text facts directly into the model's parameters (weights) via
fast target-oriented backprop steps without full dataset training loops.

Usage:
    python tools/inject.py --preset all
    python tools/inject.py --preset facts --steps 10
    python tools/inject.py --preset chat
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from npdna import NpDnaCore


BASIC_CONVERSATIONS = [
    "User: Hello! Who are you?\nAssistant: Hi! I am Atulya Tantra, a fast, local-first AI assistant.",
    "User: What is your name?\nAssistant: My name is Atulya. I am designed to be thoughtful, direct, and fast on CPU.",
    "User: How are you?\nAssistant: I am doing great and ready to help you! How can I assist you today?",
    "User: What can you do?\nAssistant: I can assist you with coding, reasoning, answer questions, and process text, vision, and audio.",
    "User: What is 2 plus 2?\nAssistant: 2 plus 2 equals 4.",
    "User: What is 5 times 5?\nAssistant: 5 times 5 equals 25.",
    "User: Thank you!\nAssistant: You are very welcome! Let me know if you need anything else.",
    "User: Bye!\nAssistant: Goodbye! Have a wonderful day!",
    "User: Who created you?\nAssistant: I am built on the Atulya Tantra NP-DNA neuroplastic architecture.",
    "User: Explain gravity in one sentence.\nAssistant: Gravity is the force that pulls objects toward each other, keeping planets in orbit and objects on the ground."
]

SAMPLE_FACTS = [
    "Q: What is the primary advantage of Atulya Tantra? A: Extremely fast CPU execution with minimal RAM.",
    "Q: How to write a fast string reversal in Python? A: Use the slice syntax result = text[::-1].",
    "Q: What is the speed of light? A: Approximately 299,792,458 meters per second."
]

FACT_TEST_PROMPTS = [
    "What is the primary advantage of Atulya Tantra?",
    "How to write a fast string reversal in Python?",
]

CHAT_TEST_PROMPTS = [
    "Hello! Who are you?",
    "What is your name?",
    "What can you do?",
    "What is 2 plus 2?",
]


def inject_facts_into_weights(
    core: NpDnaCore,
    facts: list[str],
    steps_per_fact: int = 5,
    lr: float = 1e-3
) -> dict[str, float]:
    """Bakes text facts directly into model parameters in milliseconds."""
    model = core.model
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    fact_losses = {}

    for i, text in enumerate(facts, 1):
        token_ids = core.encode(text, allow_growth=False)
        if len(token_ids) < 2:
            continue

        x = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

        initial_loss = 0.0
        final_loss = 0.0

        for step in range(steps_per_fact):
            opt.zero_grad(set_to_none=True)
            logits, bal_loss = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss = loss + bal_loss * 0.05

            if step == 0:
                initial_loss = float(loss.item())

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            final_loss = float(loss.item())

        snippet = text[:40] + ("..." if len(text) > 40 else "")
        print(f"  [{i}/{len(facts)}] Fact: {snippet!r}")
        print(f"       Loss: {initial_loss:.4f} → {final_loss:.4f} (Drop: {initial_loss - final_loss:+.4f})")
        fact_losses[snippet] = final_loss

    model.eval()
    return fact_losses


def main():
    p = argparse.ArgumentParser(description="Directly inject facts and conversations into NP-DNA weights")
    p.add_argument("--checkpoint", default="model/latest", help="Model checkpoint path")
    p.add_argument("--steps", type=int, default=5, help="Gradient steps per fact")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for fact injection")
    p.add_argument(
        "--preset",
        choices=["facts", "chat", "all"],
        default="all",
        help="Which set of facts to inject: 'facts' (sample Q&A), 'chat' (basic conversations), or 'all' (default: both)"
    )
    args = p.parse_args()

    print(f"\n⚡ Direct Weight Write-Back (Fast Fact Injection)")
    print(f"  Loading checkpoint: {args.checkpoint}...")
    core = NpDnaCore.load(args.checkpoint)

    facts: list[str] = []
    test_prompts: list[str] = []

    if args.preset in ("facts", "all"):
        facts.extend(SAMPLE_FACTS)
        test_prompts.extend(FACT_TEST_PROMPTS)

    if args.preset in ("chat", "all"):
        facts.extend(BASIC_CONVERSATIONS)
        test_prompts.extend(CHAT_TEST_PROMPTS)

    print(f"\n--> Injecting {len(facts)} facts directly into model parameters ({args.steps} steps/fact)...")
    inject_facts_into_weights(core, facts, steps_per_fact=args.steps, lr=args.lr)

    core.save(args.checkpoint)
    print(f"\n✅ Weights updated and saved to {args.checkpoint}!")

    print("\n--> Testing Model Response After Direct Weight Injection:")
    for q in test_prompts:
        res = core.generate(q, max_tokens=40, temperature=0.2)
        print(f"\nQ: {q}")
        print(f"A: {res}")


if __name__ == "__main__":
    main()
