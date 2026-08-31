"""
tools/build_super_vocab.py — Unified Multimodal Super-Vocabulary Builder for Tantra-LLM.

Capabilities:
- Harvests subword tokenizers from flagship Hugging Face models (Qwen, Gemma, Llama, Mistral).
- Merges unique tokens with project datasets including Devanagari/Hindi, Sanskrit, Python, C++, Math, and Special Control Tags.
- Trains and exports an optimal, high-compression 64K / 65,536 BPE vocabulary to Model/tokenizer.json.
"""
import os
import sys
import json
import glob
from typing import Set, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Tantra.config import VocabConfig
from Tantra.tokenizer import ByteBPETokenizer

DEFAULT_VOCAB_SIZE = 65536  # Tensor Core 64K Aligned Limit

SOTA_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "google/gemma-2b",
    "meta-llama/Meta-Llama-3-8B",
    "gpt2"
]

SPECIAL_TOKENS = [
    "<pad>", "<s>", "</s>", "<unk>",
    "<|im_start|>", "<|im_end|>",
    "<|system|>", "<|user|>", "<|assistant|>",
    "<|thought|>", "</|thought|>",
    "<|call:tool|>", "<|tool_output|>",
    "```python", "```json", "```cpp", "```rust", "```sql",
    "def __init__", "import torch", "async def", "fn main()",
    "User:", "Assistant:", "System:"
]

def harvest_subwords(hf_token: Optional[str] = None) -> Set[str]:
    """Harvest unique subwords across SOTA HuggingFace model tokenizers."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        os.system("pip install -q transformers tokenizers")
        from transformers import AutoTokenizer

    unique_tokens = set()
    print("[+] Harvesting subwords from SOTA model families...")

    for model_id in SOTA_MODELS:
        try:
            print(f"  --> Loading tokenizer: {model_id}...")
            tok = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
            vocab = tok.get_vocab()
            for token_str in vocab.keys():
                if isinstance(token_str, str) and len(token_str.strip()) > 0:
                    unique_tokens.add(token_str)
            print(f"      ✓ Harvested {len(vocab):,} tokens from {model_id}")
        except Exception as e:
            print(f"      ⚠️ Skipped {model_id}: {e}")

    print(f"[+] Harvested {len(unique_tokens):,} total unique subword candidates.")
    return unique_tokens

def build_super_vocabulary(vocab_size: int = DEFAULT_VOCAB_SIZE, hf_token: Optional[str] = None, strategy: str = "harvest", output_path: Optional[str] = None, max_corpus_mb: int = 256):
    print("\n=========================================================")
    print(f"  TANTRA-LLM MULTIMODAL SUPER-VOCABULARY BUILDER [{strategy.upper()}]  ")
    print("=========================================================\n")

    subwords = harvest_subwords(hf_token) if strategy == "harvest" else set()
    sample_corpus_path = os.path.join(REPO_ROOT, "Model", "super_corpus_sample.txt")
    os.makedirs(os.path.dirname(sample_corpus_path), exist_ok=True)

    print(f"[+] Building training corpus -> {sample_corpus_path}...")
    with open(sample_corpus_path, "w", encoding="utf-8") as f:
        f.write("नमस्ते, तन्त्र-LLM में आपका स्वागत है।\n"
                "तन्त्र (Sanskrit: तन्त्र) — An instrument that weaves threads of knowledge.\n"
                "User: What is artificial intelligence?\nAssistant: AI is a branch of computer science.\n")

        if subwords:
            f.write(" ".join(list(subwords)[:100000]) + "\n")

        search_dirs = [os.path.join(REPO_ROOT, "Datasets"), os.path.join(REPO_ROOT, "Model")]
        raw_files = []
        for d in search_dirs:
            if os.path.exists(d):
                raw_files.extend(glob.glob(os.path.join(d, "*.jsonl")))
                raw_files.extend(glob.glob(os.path.join(d, "**/*.jsonl"), recursive=True))
                raw_files.extend(glob.glob(os.path.join(d, "*.txt")))
                raw_files.extend(glob.glob(os.path.join(d, "**/*.txt"), recursive=True))

        byte_budget = max(1, max_corpus_mb) * 1024 * 1024
        bytes_written = f.tell()
        for fpath in sorted(list(set(raw_files))):
            if bytes_written >= byte_budget:
                break
            if "super_corpus_sample" in fpath or "temp" in fpath:
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if line:
                            if fpath.endswith(".jsonl"):
                                try:
                                    data = json.loads(line)
                                    if "messages" in data:
                                        text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in data["messages"]])
                                    elif "text" in data: text = data["text"]
                                    else: text = line
                                except Exception: text = line
                            else: text = line
                            encoded = (text + "\n").encode("utf-8")
                            remaining = byte_budget - bytes_written
                            if remaining <= 0:
                                break
                            f.write(encoded[:remaining].decode("utf-8", errors="ignore"))
                            bytes_written = f.tell()
                            if bytes_written >= byte_budget:
                                break
            except Exception: pass

    cfg = VocabConfig(vocab_size=vocab_size)
    bpe = ByteBPETokenizer(cfg)

    print(f"\n[+] Training {vocab_size/1024:.0f}K BPE Super-Tokenizer (Target: {vocab_size:,})...")
    bpe.train([sample_corpus_path], vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)

    output_json = output_path or os.path.join(REPO_ROOT, "Model", "tokenizer.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    bpe.save(output_json)
    print(f"\n✅ Merged Super-Vocabulary Successfully Saved -> {output_json} ({bpe.vocab_size:,} tokens)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tantra Super-Vocabulary Builder")
    parser.add_argument("--strategy", choices=["harvest", "corpus"], default="harvest", help="Harvest HF subwords or train strictly on local corpus")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="Target vocabulary size")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API Access Token")
    parser.add_argument("--output", type=str, default=None, help="Tokenizer JSON path; use a separate path for a new model profile")
    parser.add_argument("--max-corpus-mb", type=int, default=256, help="Maximum sampled local corpus size; keeps tokenizer builds fast")
    args = parser.parse_args()

    build_super_vocabulary(vocab_size=args.vocab_size, hf_token=args.token, strategy=args.strategy, output_path=args.output, max_corpus_mb=args.max_corpus_mb)
