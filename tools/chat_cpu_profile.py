"""Chat with a separate CPU profile checkpoint, not main.py's legacy model."""
from __future__ import annotations
import argparse
import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tantra.config import VocabConfig
from Tantra.cpu_profiles import build_cpu_model
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from main import run_interactive_chat

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="Model/CPU_Dense32K")
    parser.add_argument("--tokenizer", default="Model/tokenizer.json")
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--top-p", type=float, default=0.80)
    args = parser.parse_args()
    checkpoint = os.path.join(args.model_dir, "Latest", "checkpoint_latest.pt")
    cfg = VocabConfig(vocab_size=32768)
    tokenizer = UnifiedTokenizer(cfg, ByteBPETokenizer.load(args.tokenizer, cfg), MegabytePatcher())
    model = build_cpu_model("dense", attention_kind="causal", vocab_size=32768).to("cpu")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    run_interactive_chat(model, tokenizer, "cpu", args.temperature, args.top_p)

if __name__ == "__main__":
    main()
