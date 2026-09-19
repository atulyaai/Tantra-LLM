# Fixes applied — Datasets/ (Tantra-LLM)

## Round 1 (previous pass)
- Fixed 19,470/19,472 wrong multiplication answers and 8,569/16,970 sign-flipped
  subtraction answers in the raw math data.
- Deduplicated and removed train/val leakage in the Hindi set.
- Removed byte-identical duplicate files.
- Relabeled `tantra_coding.jsonl` from `lang: hi` to `lang: en` (content is English).
- Fixed `main.py`'s stale default-dataset filenames.

## Round 2 (this pass) — full merge, dedupe, and real Hindi-content growth

**Final file list (was 14 files, now 7):**
| File | Rows | Purpose |
|---|---|---|
| `tantra_hindi_conversation_train.jsonl` | 98 | Hindi conversation/identity/general-knowledge — **use this for Hindi-only conversational training** |
| `tantra_hindi_conversation_val.jsonl` | 20 | held-out val for the above, zero overlap with train |
| `tantra_math_train.jsonl` | 65,127 | all math word problems, merged, deduped, answers corrected |
| `tantra_math_val.jsonl` | 7,810 | held-out val for math, zero overlap with train or with conversation set |
| `tantra_coding.jsonl` | 12 | small English coding set, unchanged |
| `gold_corpus.jsonl` | 410 | English general-QA, unchanged (see "left for you" below) |
| `preference_pairs.jsonl` | 4 | too small to use, unchanged |

**What got merged / removed and why:**
- `tantra_math.jsonl` was 99.99% the same content as `tantra_combined_train.jsonl`, just in a
  different schema (flat vs `messages`). Merged both into one deduped, schema-normalized
  `tantra_math_train.jsonl` (143,310 raw rows → 65,127 unique) and did the same for
  `tantra_combined_val.jsonl` → `tantra_math_val.jsonl` (7,963 → 7,810, with train/val
  overlap removed).
- `tantra_identity_train.jsonl` (640 rows, only 32 unique) turned out to be 100% duplicated
  inside `tantra_conversation.jsonl` — removed.
- `tantra_conversation.jsonl` (655 rows, only 47 unique) and `tantra_science.jsonl` (14 rows)
  turned out to be 100% duplicated inside `tantra_master_train.jsonl` — removed.
- **Found a second hidden copy of the broken math generator**: 3,807 of `tantra_master_train.jsonl`'s
  3,888 rows were mislabeled `domain: word_problem_other` — they were math word problems with
  the same wrong-answer bug, sitting inside what was supposed to be your Hindi *conversation*
  file. Extracted them, fixed the same multiplication/subtraction bugs, merged into
  `tantra_math_train.jsonl`. This is why your Hindi conversation set looked bigger than it
  actually was — once the math was pulled out, only **81 real, unique conversational rows**
  were left in the entire repo.
- 8 stray identity/greeting/coding rows had also leaked into the math files from the old
  combined dataset — removed from the math side, kept once in the conversation set.

**Actually growing the Hindi conversation set (not just math):**
This was the real gap — 81 rows, almost entirely factual trivia (geography/history/science
facts) and "who are you" identity Q&A, with zero small talk, advice, opinions, reasoning, or
daily-life conversation. I hand-wrote **37 new, natural Hindi conversational rows** across
domains that didn't exist before: `small_talk`, `advice`, `emotion`, `culture`, `food`,
`health`, `opinion`, `reasoning`, `howto`, `greeting`. Merged, deduped against the existing 81,
and re-split 118 unique rows into a clean 98/20 train/val split with zero leakage.

This is a starting point, not a finished dataset — 98 training rows is still small for a
"better conversation" model. The honest next step, in priority order:
1. Translate `gold_corpus.jsonl`'s 410 English general-QA rows into Hindi (still not done —
   it's the single biggest available lever, kept out again this round because 410 natural
   translations is a bigger job than a cleanup pass).
2. Keep hand-authoring conversational rows in the domains above — variety matters more than
   volume here.
3. Once `main.py --mode adapter add` is used to give math and conversation their own adapter
   layers (per the earlier discussion), you can safely oversample this small conversation set
   during its own adapter's training without it being drowned out by the 65K math rows.

`main.py`'s default-dataset logic has been updated again to point at the renamed
`tantra_hindi_conversation_train.jsonl` / `_val.jsonl`.
