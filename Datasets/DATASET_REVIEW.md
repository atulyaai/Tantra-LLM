# Dataset review — Datasets/ (Tantra-LLM)

## What's actually in there

| File | Rows | Schema | Language |
|---|---|---|---|
| `hindi_train.jsonl` | 92,746 | flat `{system,user,assistant}` | **100% synthetic math word problems**, Hindi |
| `hindi_val.jsonl` | 10,306 | same | same |
| `gold_corpus.jsonl` | 410 | `{instruction,input,output,domain,complexity}` | English, general conversation/reasoning |
| `conversation_greetings.jsonl` | 9 | flat `{system,user,assistant}` | 5 English, 4 Hindi |
| `sft_train.jsonl` / `sft_val.jsonl` | 93,165 / 10,306 | unified `{"messages":[...]}` | already the merge of the three train sources above, per `manifest.json` — **but this merge is not Hindi-only**, it includes the 410 English gold_corpus rows and 5 English greetings |
| `preference_pairs.jsonl` | 4 | DPO `{prompt,chosen,rejected}` | too few to train on — `manifest.json` already flags this itself |
| `Datasets/documents/` | 1 file, 74 bytes | RAG test doc | not usable as training data |

## What I checked

**Every single row of `hindi_train.jsonl`/`hindi_val.jsonl` is a synthetic math word problem** — arithmetic, percentages, profit/loss, simple/compound interest, geometry (area/volume/perimeter), physics (velocity, kinetic energy, power), ratios, LCM/HCF, linear equations, number sequences, fraction arithmetic. There is **zero conversational, factual, or open-domain content** in either file — no small talk, no explanations, no "why" questions, nothing outside a Q→numeric-answer format. A model trained only on this will get good at Hindi arithmetic and nothing else.

**I verified the math itself is correct**, not just present:
- Auto-checked ~53,600 rows across 6 template families (plain arithmetic, percentages, simple interest, speed/distance, distance/time, ratio simplification) by recomputing the expected answer and comparing — **0 errors found.**
- Spot-checked the remaining families (geometry, physics, sequences, fractions) by hand — all correct.
- **0 exact duplicate (question, answer) pairs** across all 92,746 training rows.

**Domain breakdown of `hindi_train.jsonl`** (I added these tags — they don't exist in the source file):

| Domain | Rows | Share |
|---|---|---|
| math_arithmetic | 40,837 | 44.0% |
| math_finance (profit/loss, interest) | 11,014 | 11.9% |
| math_percentage | 10,652 | 11.5% |
| math_physics | 7,642 | 8.2% |
| math_algebra | 5,385 | 5.8% |
| math_geometry | 3,879 | 4.2% |
| math_number_theory (LCM/HCF) | 3,232 | 3.5% |
| math_sequence | 2,237 | 2.4% |
| math_fraction | 1,799 | 1.9% |
| math_ratio | 773 | 0.8% |
| (uncategorized word problems) | 5,296 | 5.7% |

## What I changed (given "Hindi-only for now")

1. **Converted `hindi_train.jsonl` + `hindi_val.jsonl` to the project's unified `messages` schema** (matching `sft_train.jsonl`'s documented format), so they're drop-in compatible with whatever loader reads the existing SFT files.
2. **Tagged every row with a `domain`** using the breakdown above, so you can later do curriculum ordering, per-domain eval, or oversample/undersample a category — none of that's possible with the flat source files as-is.
3. **Kept train/val split intact** — I did not merge train into val or vice versa; that would silently destroy your held-out set. "Bigger" here means enriching train, not conflating the two.
4. **Added 9 short Hindi conversational examples**: the 4 that were already Hindi in `conversation_greetings.jsonl`, plus 5 that were in English there — I translated those 5 into natural Hindi myself (not machine translation) and re-phrased them so they don't collide with the 4 originals. This is a small but real dent in the "100% math, 0% conversation" problem.
5. **Left `gold_corpus.jsonl` (410 rows) and the rest of `conversation_greetings.jsonl` out**, because they're English and you said Hindi-only for now.
6. **Left `preference_pairs.jsonl` untouched** — 4 examples is too few to matter either way, and the manifest already correctly flags its "rejected" completions as incoherent filler rather than plausible-but-wrong (which would teach "sound fluent" instead of "be correct").

## Honest limits — I did not make the data "bigger" in any deep sense

There is no other Hindi-language source anywhere in this repo. I did **not** invent new math problems or synthesize new data, because:
- More procedurally-generated arithmetic wouldn't fix the actual gap (100% math, 0% everything else) — it would just be more of the same 44%-arithmetic pie.
- I'm not going to fabricate "real-world" Hindi text and pass it off as authentic training data.

**Result: same row count as before** (92,755 train / 10,306 val — +9 from the translated greetings), just cleaner, schema-unified, domain-tagged, and verified correct. If you want a genuinely bigger and more diverse Hindi set, the real lever is:

- **Translate `gold_corpus.jsonl`'s 410 English general-QA examples into Hindi.** This is the highest-value next step — it's the only conversational/non-math content in the whole repo, and translating it in is what would actually stop the model from being "a calculator that speaks Hindi." I can do this next if you want — it's ~410 short items, doable, but is enough text that I'd want to do it as its own follow-up rather than folding it silently into this response.
- Bring in an external Hindi corpus (news, Wikipedia, dialogue) — not something I can source from inside this offline sandbox (no network access here).
