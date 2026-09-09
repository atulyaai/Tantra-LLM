"""
Tantra-LLM: Standalone Dataset Diversity & Quality Auditor.
Works locally or directly in Kaggle/Colab notebooks.

Usage:
    python check_dataset_diversity.py --train Datasets/tantra_conversation_train.jsonl --val Datasets/tantra_conversation_val.jsonl
    python check_dataset_diversity.py --train /kaggle/working/tantra_final_dataset.jsonl
"""
import argparse
import collections
import json
import math
import os
import re
import sys


def clean_text(text: str) -> str:
    """Normalize text for duplicate and n-gram comparison."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t)


def analyze_dataset(train_path: str, val_path: str = None, top_k_words: int = 25):
    print("=" * 70)
    print("🔬 TANTRA DATASET DIVERSITY & QUALITY AUDITOR")
    print("=" * 70)

    if not os.path.isfile(train_path):
        print(f"❌ Error: Train dataset not found at: {train_path}")
        return

    print(f"📁 Analyzing Train Dataset: {train_path} ({os.path.getsize(train_path)/1e6:.2f} MB)")

    train_queries = []
    train_answers = []
    train_systems = collections.Counter()
    exact_query_dups = collections.Counter()
    exact_pair_dups = collections.Counter()
    word_freq = collections.Counter()
    query_lengths = []
    answer_lengths = []
    total_tokens_est = 0
    blank_lines = 0
    malformed_lines = 0

    with open(train_path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                blank_lines += 1
                continue
            try:
                data = json.loads(line)
            except Exception:
                malformed_lines += 1
                continue

            # Support {system, user, assistant} or {prompt, response} or {instruction, output}
            user_text = data.get("user") or data.get("prompt") or data.get("instruction") or ""
            asst_text = data.get("assistant") or data.get("response") or data.get("output") or ""
            sys_text = data.get("system") or "none"

            train_systems[sys_text] += 1
            norm_q = clean_text(user_text)
            norm_a = clean_text(asst_text)

            exact_query_dups[norm_q] += 1
            exact_pair_dups[(norm_q, norm_a)] += 1

            train_queries.append(norm_q)
            train_answers.append(norm_a)

            q_words = norm_q.split()
            a_words = norm_a.split()
            query_lengths.append(len(q_words))
            answer_lengths.append(len(a_words))
            total_tokens_est += int((len(q_words) + len(a_words)) * 1.3)

            for w in a_words:
                if len(w) > 2:
                    word_freq[w] += 1

    total_samples = len(train_queries)
    if total_samples == 0:
        print("❌ Dataset is empty or contains no valid JSON lines.")
        return

    unique_queries = len(exact_query_dups)
    unique_pairs = len(exact_pair_dups)
    query_dup_count = total_samples - unique_queries
    pair_dup_count = total_samples - unique_pairs
    unique_words_count = len(word_freq)
    total_words_count = sum(word_freq.values())

    # Lexical Diversity: Type-Token Ratio (TTR)
    ttr = (unique_words_count / max(1, total_words_count)) * 100

    print(f"\n📊 1. VOLUME & INTEGRITY")
    print(f"   • Total Samples Analyzed : {total_samples:,}")
    print(f"   • Estimated Total Tokens : {total_tokens_est:,}")
    print(f"   • Blank / Empty Lines   : {blank_lines}")
    print(f"   • Malformed JSON Lines  : {malformed_lines}")

    print(f"\n🔁 2. DUPLICATION & REPETITION")
    print(f"   • Unique Prompts/Queries : {unique_queries:,} ({(unique_queries/total_samples)*100:.1f}%)")
    print(f"   • Duplicate Prompts      : {query_dup_count:,} ({(query_dup_count/total_samples)*100:.1f}%)")
    print(f"   • Exact Duplicate Pairs  : {pair_dup_count:,} ({(pair_dup_count/total_samples)*100:.1f}%)")
    
    top_repeated_queries = [(q, c) for q, c in exact_query_dups.most_common(5) if c > 1]
    if top_repeated_queries:
        print(f"   ⚠️ Most Repeated Prompts:")
        for q, c in top_repeated_queries:
            disp_q = q[:60] + "..." if len(q) > 60 else q
            print(f"      [{c}x] \"{disp_q}\"")

    print(f"\n📚 3. VOCABULARY & LEXICAL RICHNESS")
    print(f"   • Total Content Words    : {total_words_count:,}")
    print(f"   • Distinct Vocab Words   : {unique_words_count:,}")
    print(f"   • Type-Token Ratio (TTR) : {ttr:.2f}% (Higher is more diverse; >15% is healthy)")

    print(f"   • Top Most Frequent Vocabulary Words in Assistant Outputs:")
    top_words = [f"{w} ({c})" for w, c in word_freq.most_common(12)]
    print(f"      {', '.join(top_words)}")

    print(f"\n📏 4. SEQUENCE LENGTH DISTRIBUTION (Words)")
    avg_q_len = sum(query_lengths) / max(1, total_samples)
    avg_a_len = sum(answer_lengths) / max(1, total_samples)
    short_answers = sum(1 for l in answer_lengths if l < 10)
    long_answers = sum(1 for l in answer_lengths if l > 100)
    print(f"   • Avg User Prompt Length : {avg_q_len:.1f} words")
    print(f"   • Avg Assistant Output   : {avg_a_len:.1f} words (Min: {min(answer_lengths)}, Max: {max(answer_lengths)})")
    print(f"   • Short Responses (<10w) : {short_answers:,} ({(short_answers/total_samples)*100:.1f}%)")
    print(f"   • In-Depth (>100w)       : {long_answers:,} ({(long_answers/total_samples)*100:.1f}%)")

    print(f"\n🎭 5. SYSTEM PROMPTS")
    print(f"   • Unique System Prompts  : {len(train_systems)}")
    for s_text, cnt in train_systems.most_common(3):
        disp_s = s_text[:70] + "..." if len(s_text) > 70 else s_text
        print(f"      [{cnt:,}x] \"{disp_s}\"")

    # Train / Val Leakage Check
    if val_path and os.path.isfile(val_path):
        print(f"\n🛡️  6. TRAIN / VALIDATION LEAKAGE AUDIT")
        print(f"   📁 Validating against: {val_path}")
        val_samples = 0
        leaked_queries = 0
        train_query_set = set(train_queries)
        with open(val_path, "r", encoding="utf-8", errors="replace") as vf:
            for vline in vf:
                vline = vline.strip()
                if not vline:
                    continue
                try:
                    vdata = json.loads(vline)
                    vq = clean_text(vdata.get("user") or vdata.get("prompt") or vdata.get("instruction") or "")
                    val_samples += 1
                    if vq in train_query_set:
                        leaked_queries += 1
                except Exception:
                    pass

        leak_pct = (leaked_queries / max(1, val_samples)) * 100
        print(f"   • Validation Samples     : {val_samples:,}")
        print(f"   • Leaked Queries in Val  : {leaked_queries:,} ({leak_pct:.1f}%)")
        if leaked_queries == 0:
            print("   ✅ CLEAN: Zero data leakage between train and validation splits!")
        else:
            print(f"   ⚠️ WARNING: {leaked_queries} validation questions are memorized in training!")

    print("\n" + "=" * 70)
    print("📋 SUMMARY DIAGNOSTIC & VERDICT:")
    if query_dup_count > total_samples * 0.15:
        print("🔴 HIGH DUPLICATION: More than 15% of prompts are duplicates. Model will overfit/memorize templates.")
    elif ttr < 10.0:
        print("🟡 LOW DIVERSITY: Lexical Type-Token Ratio < 10%. Data uses repetitive phrasing.")
    else:
        print("🟢 HEALTHY DIVERSITY: Low prompt duplication and strong vocabulary distribution.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tantra-LLM Dataset Diversity Auditor")
    parser.add_argument("--train", type=str, default="Datasets/tantra_conversation_train.jsonl", help="Path to train JSONL")
    parser.add_argument("--val", type=str, default="Datasets/tantra_conversation_val.jsonl", help="Path to validation JSONL (optional)")
    args = parser.parse_args()

    analyze_dataset(args.train, args.val)
