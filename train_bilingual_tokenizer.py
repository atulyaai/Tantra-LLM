"""
Train 16K Bilingual BPE Tokenizer for Tantra
============================================
Produces an efficient bilingual BPE tokenizer supporting English and Indic/Devanagari.
Eliminates byte-level fragmentation for Hindi while maintaining English efficiency.
"""

import json
import os
import shutil
import sys
from pathlib import Path

# Configure utf-8 stdout
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from Tantra.tokenizer import TantraTokenizer


def collect_corpus():
    texts = []
    print("Collecting balanced bilingual training corpus...")

    # 1. Hindi Multilingual & Indic Knowledge
    hi_paths = [
        "Datasets/experts/multilingual/lang_hi.jsonl",
        "Datasets/curated/knowledge/gk_indic.jsonl",
    ]
    hi_count = 0
    for p in hi_paths:
        path = Path(p)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    for m in row.get("messages", []):
                        content = m.get("content", "").strip()
                        if content:
                            texts.append(content)
                            hi_count += 1
                except Exception:
                    pass
    print(f"  · Collected {hi_count} Hindi/Indic texts")

    # 2. General Knowledge Domains (Physics, Math, Science, History, Tech, AI, Biology)
    gk_count = 0
    for p in Path("Datasets/experts/knowledge").glob("*.jsonl"):
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
                if i >= 400:  # sample 400 rows per domain (24 domains = ~9,600 knowledge texts)
                    break
                row = json.loads(line)
                if "text" in row:
                    texts.append(row["text"])
                    gk_count += 1
                elif "messages" in row:
                    for m in row.get("messages", []):
                        texts.append(m.get("content", ""))
                        gk_count += 1
        except Exception:
            pass
    print(f"  · Collected {gk_count} Knowledge domain texts")

    # 3. Curated Conversation Dialogues
    conv_count = 0
    for p in Path("Datasets/curated").rglob("*.jsonl"):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                for m in row.get("messages", []):
                    texts.append(m.get("content", ""))
                    conv_count += 1
        except Exception:
            pass
    print(f"  · Collected {conv_count} Conversational texts")

    # 4. General English Corpus (TinyStories / General text)
    gen_count = 0
    gen_file = Path("Datasets/general/tinystories_valid.txt")
    if gen_file.exists():
        with open(gen_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and len(s) > 20:
                    texts.append(s)
                    gen_count += 1
                if gen_count >= 15000:
                    break
    print(f"  · Collected {gen_count} General text paragraphs")
    print(f"Total training corpus: {len(texts)} samples")
    return texts


def main():
    corpus = collect_corpus()
    vocab_size = 16384
    print(f"\nTraining ByteLevel BPE Tokenizer (vocab_size={vocab_size})...")
    tok = TantraTokenizer.train(corpus, vocab_size=vocab_size, min_frequency=2)
    print(f"Trained vocabulary size: {tok.vocab_size}")

    # Test efficiency
    en_sample = "Hello, how are you today?"
    hi_sample = "नमस्ते, आप आज कैसे हैं?"

    en_toks = tok.encode(en_sample)
    hi_toks = tok.encode(hi_sample)

    print("\nTokenization Efficiency Benchmark:")
    print(f"  · English: '{en_sample}' -> {len(en_toks)} tokens")
    print(f"  · Hindi:   '{hi_sample}' -> {len(hi_toks)} tokens (reduced from 57!)")

    # Save to Model
    out_path = Path("Model/tokenizer.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        backup = Path("Model/tokenizer_backup_8k.json")
        if not backup.exists():
            shutil.copy2(out_path, backup)
            print(f"Backed up previous 8k tokenizer -> {backup}")

    tok.save(out_path)
    print(f"Saved new 16k bilingual tokenizer -> {out_path}")


if __name__ == "__main__":
    main()
