"""Interactive chat interface for Tantra — multilingual AI assistant.

Supports:
  - Single-prompt mode: --prompt "..."
  - Interactive terminal mode (default)
  - Optional LoRA adapter loading: --adapter-dir <path>

Identity is enforced via Tantra/identity.py regardless of base model.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from Tantra.identity import IDENTITY_SYSTEM_MESSAGE, identity_response, prepare_messages
from Tantra.qwen35_finetune import DEFAULT_REVISION, MODEL_ID

# Default generation parameters optimised for CPU inference.
_DEFAULT_MAX_NEW_TOKENS = 128
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_P = 0.9

WELCOME_BANNER = """
╔══════════════════════════════════════════════════╗
║          तंत्रा  |  Tantra AI Assistant           ║
║  Hindi · English · Hinglish                       ║
║  Type 'exit' or 'quit' to stop.                  ║
╚══════════════════════════════════════════════════╝
"""


def _generate(model, tokenizer, messages, args, device):
    """Run one forward pass and return the decoded response string."""
    import torch

    # Deterministic identity short-circuit: check last user message first
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), None
    )
    if last_user:
        hard_answer = identity_response(last_user)
        if hard_answer is not None:
            return hard_answer

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
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
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    new_ids = generated[0, input_ids.shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


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
        dtype=dtype,            # use dtype= not torch_dtype= (Transformers 5.x)
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
    return parser


def main():
    parser = build_parser()
    chat_loop(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
