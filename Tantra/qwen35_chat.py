"""Interactive chat interface for Tantra — multilingual AI assistant.

Supports:
  - Single-prompt mode: --prompt "..."
  - Interactive terminal mode (default)
  - Optional LoRA adapter loading: --adapter-dir <path>

Identity is enforced via Tantra/identity.py regardless of base model.
Arithmetic is resolved via Tantra/reasoning.py math_answer.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from Tantra.identity import IDENTITY_SYSTEM_MESSAGE, identity_response, prepare_messages
from Tantra.qwen35_finetune import DEFAULT_REVISION, MODEL_ID
from Tantra.reasoning import math_answer

# Default generation parameters optimised for coherent CPU inference.
_DEFAULT_MAX_NEW_TOKENS = 128
_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_TOP_P = 0.9
_DEFAULT_REPETITION_PENALTY = 1.15

WELCOME_BANNER = """
╔══════════════════════════════════════════════════╗
║          तंत्रा  |  Tantra AI Assistant           ║
║  Hindi · English · Hinglish                       ║
║  Type 'exit' or 'quit' to stop.                  ║
╚══════════════════════════════════════════════════╝
"""

# Few-shot anchors when chatting with raw base model without adapter
_FEW_SHOT_BASE = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a popular, easy-to-read programming language used for web development, automation, data analysis, and AI."},
    {"role": "user", "content": "नमस्ते! क्या आप हिंदी में बात कर सकते हैं?"},
    {"role": "assistant", "content": "नमस्ते! हाँ, मैं हिंदी, अंग्रेज़ी और हिंग्लिश में बात कर सकता हूँ। मैं आपकी कैसे मदद करूँ?"},
]


def _clean_response(text: str) -> str:
    """Trim stop artifacts and self-continuation turns."""
    for stop in ("\nUser:", "\nHuman:", "\nSystem:", "\nउपयोगकर्ता:", "\nAssistant:", "<|im_end|>", "<|endoftext|>"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def _generate(model, tokenizer, messages, args, device):
    """Run one forward pass and return the decoded response string."""
    import torch

    # 1. Deterministic identity short-circuit: check last user message first
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), None
    )
    if last_user:
        hard_answer = identity_response(last_user)
        if hard_answer is not None:
            return hard_answer

        # 2. Arithmetic exact evaluation short-circuit
        math_res = math_answer(last_user)
        if math_res:
            return math_res

    # 3. Format message sequence
    eval_messages = list(messages)
    # If using pure base model without an adapter and only 1 turn, add few-shot anchors
    if not args.adapter_dir and len([m for m in eval_messages if m.get("role") == "user"]) == 1:
        system_msg = eval_messages[0]
        user_msg = eval_messages[-1]
        eval_messages = [system_msg] + _FEW_SHOT_BASE + [user_msg]

    inputs = tokenizer.apply_chat_template(
        eval_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if isinstance(inputs, Mapping):
        input_ids = inputs["input_ids"].to(device)
    elif hasattr(inputs, "input_ids"):
        input_ids = inputs.input_ids.to(device)
    else:
        input_ids = inputs.to(device)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p if args.temperature > 0 else None,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    new_ids = generated[0, input_ids.shape[-1]:]
    raw_answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return _clean_response(raw_answer)


def chat_loop(args):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install the fine-tuning extra: pip install -e '.[finetune]'"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )
    print(f"Loading {args.model_id}@{args.revision} on {device} ({dtype})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if args.adapter_dir:
        adapter_path = Path(args.adapter_dir)
        print(f"Loading LoRA adapter from {adapter_path}...", flush=True)
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.to(device)
    model.eval()

    # Build initial message list with Tantra identity system prompt
    base_system = args.system if args.system else IDENTITY_SYSTEM_MESSAGE
    seed_messages: list[dict[str, str]] = [{"role": "system", "content": base_system}]
    messages = prepare_messages(seed_messages)

    if args.prompt:
        # ── Single-prompt mode ───────────────────────────────────────────────
        messages.append({"role": "user", "content": args.prompt})
        answer = _generate(model, tokenizer, messages, args, device)
        print(answer)
        return

    # ── Interactive terminal mode ────────────────────────────────────────────
    print(WELCOME_BANNER)
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! | अलविदा!")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "बाहर", "बंद"}:
            print("Goodbye! | अलविदा!")
            break
        messages.append({"role": "user", "content": user_input})
        answer = _generate(model, tokenizer, messages, args, device)
        print(f"\nTantra: {answer}\n")
        messages.append({"role": "assistant", "content": answer})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive chat with Tantra (multilingual AI — Hindi/English/Hinglish)."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--adapter-dir", default=None, help="Path to saved LoRA adapter directory"
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Override system prompt (default: Tantra identity prompt)",
    )
    parser.add_argument("--prompt", default=None, help="Single prompt execution mode")
    parser.add_argument("--max-new-tokens", type=int, default=_DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=_DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=_DEFAULT_TOP_P)
    parser.add_argument("--repetition-penalty", type=float, default=_DEFAULT_REPETITION_PENALTY)
    return parser


def main():
    parser = build_parser()
    chat_loop(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
