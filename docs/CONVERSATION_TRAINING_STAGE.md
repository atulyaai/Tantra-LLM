# Supervised conversation stage

`python -m Tantra.conversation_train` adds a separate assistant-only supervised
stage after base pretraining. It is not a claim that the currently poor dialogue
responses have been repaired. It never deploys a model or approves data.

```powershell
python -m Tantra.conversation_train prepare --bundle Datasets/workbench-conversation-v2 --output runs/new-chat-readiness.json
python -m Tantra.conversation_train train --bundle Datasets/workbench-conversation-reviewed --source runs/pretraining-20m-v1/base-training --output runs/chat-candidate-v1
```

The real inspected bundle has 98 reviewed English examples and no approved
Hindi/Hinglish examples. The readiness report runs/conversation-readiness-v3.json
lists 72 expert/language/split shortfalls for greetings, three identity experts,
clarification, empathy, farewell and smalltalk. The 19 original multilingual
drafts remain outside training; runs/conversation-review-pack-v3 contains a TSV
review aid. Even approving those 19 will not satisfy the data minimums.

The gate requires the existing 1000/100/100 independent families for each selected
expert and each of Hindi/English/Hinglish. It validates every frozen split hash,
strict provenance/review fields and cross-split family isolation. Hindi/Hinglish
records also require a recorded fluent_language_review attestation. The import
path now preserves that real reviewer-supplied flag; it is never auto-created.
Prepared bundles must be regenerated as new versions after reviewed data changes.

Training requires a completed source base and improving Hindi/English validation,
matching tokenizer/vocabulary and a separate output directory. It rejects examples
longer than 256 encoded tokens instead of silently dropping assistant labels.
Only assistant tokens contribute to supervised loss. Three-record balanced updates
include one example per language, use the smallest available language pool without
duplicating other rows, and perform one shuffled pass. This balances example counts,
not necessarily target-token counts; report language coverage separately.
Each update runs its records as one right-padded batch (2026-10-20); because attention is
causal and the loss averages over all supervised tokens, this is numerically the same as the
earlier per-row token-weighted passes (`tests/test_batched_sft.py`) but needs one forward/backward.
English pilots may set `--rows-per-update N` for larger updates; balanced mode stays at 3.

The base is fine-tuned with four CPU threads, FP32, AdamW LR 2e-5, gradient clipping,
resource checks, separate atomic weights/optimizer/RNG/cursor checkpoints, clean
interruption and per-language held-out assistant loss. This starts a new optimizer
for supervised learning; same-stage resume restores it. No adapters/router are
silently inherited or validated. Token loss is not a conversation release score.
Unexpected failures preserve the last healthy checkpoint; automatic rollback and
stale-lock cleanup for the chat entry point are not claimed.

Actual language data gates currently stop training before any model load/update.
The numerical path was tested with a real tiny transformer and explicitly synthetic
fixtures; fixture approval is not real dataset approval. Tests verify user-token
masking, a real optimizer step, language nonduplication, rejection of actual missing
review coverage, checkpoint state and unchanged source model.

Continue the base with `python -m Tantra.pretrain --resume --microbatch 16 --dashboard`.
The base process was restarted only after checking that no existing pretrain worker
was alive. This does not start the blocked chat stage.

## English-only pilot mode (2026-10-18)

`python -m Tantra.conversation_train train --english-pilot --epochs N ...` is an explicitly
labelled, undersized English-only pilot. It uses English rows only, reports family
minimums without enforcing them, needs reviewed train and validation rows, and its
readiness is always `english_pilot_undersized` (never `data_ready`). Final state is
`english_pilot_complete_unvalidated`, with `pilot_warning`, `mode` and `epochs` in the
report; it is never deployed. `--epochs` (shuffled passes, no repeats within a pass) is
rejected outside pilot mode. `TANTRA_DEVICE=cuda` runs it on a GPU. The balanced
trilingual gate for normal training is unchanged.

Bundle `Datasets/workbench-chat-en-pilot-v1`: owner-approved English rows only
(identity_self v3-v6, greeting/smalltalk/farewell v5): 253/11/15 rows. Local CPU smoke
run (1 epoch, 85 updates, 3.2 min): English assistant-token validation loss 6.85 -> 3.76.
Kaggle GPU run (10 epochs): `scripts/kaggle/chat_pilot_upload.ps1` + `chat_pilot_kaggle.py`,
kernel tantrallm/tantra-chat-en-pilot; it also writes base vs pilot answers to fresh prompts.

Local CPU 10-epoch result (2026-10-18): `runs/chat-en-pilot-local-v1`, 844 updates, English
validation loss 6.85 -> 3.05, state english_pilot_complete_unvalidated, not deployed.
On 14 fresh prompts (scripts/compare_chat_pilot.py) the base gives Wikipedia fragments for all 14.
The pilot gives coherent, on-fact identity replies for about 8 of 10 identity prompts
("Who is the mind behind you?" -> "Atul Vij is the person who created me.") but
over-applies identity to chat: "Hello there, friend." -> "I'm Tantra! What's Atulya AI?",
"It is so cold today." -> "Tantra AI is my parent company.", "Good evening!" -> "Good morning!...".
Cause: identity is 162 of 215 train families. Next: many more greeting/smalltalk/farewell
families so the data is balanced across situations.

Pilot v2 (2026-10-19): bundle `Datasets/workbench-chat-en-pilot-v2` adds 148 owner-approved v7
greeting/smalltalk/farewell drafts (387/16/24 rows; identity 46% of train families, was 75%).
Local CPU, 10 epochs, 1,290 updates (stopped at 620 and resumed): English validation loss
7.07 -> 3.49 on 16 rows (different validation set from v1, so not directly comparable).
Fresh-prompt comparison vs v1: casual chat improved ("I have to leave now, bye." -> "No problem!
Take care."; "Hello there, friend." -> a greeting instead of an identity reply), identity mostly
held ("Who coded you?" -> "Atul Vij created me."). Still wrong: "Good evening!" -> "Good morning!...",
"It is so cold today." -> unrelated, "Which company do you represent?" garbled, "Are you Atul Vij
himself?" -> "Atul Vij created me." The Kaggle helper now locates the bundle from its manifest
(the earlier version pointed at Datasets/ and failed), and code, bundle and model ship as one dataset.

Pilot v3 (2026-10-19): bundle `Datasets/workbench-chat-en-pilot-v3` (owner-approved v8 persona
revisions replace 51 replies that mentioned AI/creator unasked; 459/19/30 rows). Local CPU, 6 epochs,
918 updates, best.pt at update 918, English validation 7.07 -> 3.14. Despite the lower loss, answers on
22 fresh prompts regressed in style: repetition ("I'm Tantra, and I'm Tantra"), unasked "I'm an AI",
"Good morning!" to bye/evening, and unrelated creator lines ("My exam went well today!" -> "Atul Vij is
my creator."). Conclusion: a few hundred rows teach a few patterns; the next step is 10K+ own,
filtered conversations (see docs/BULK_DATA_POLICY.md).

Measured (2026-10-20, pilot-v1 bundle, 1 epoch, CPU): batched 3 rows/update 2.4 min vs 3.2 min before, identical validation 6.85 -> 3.76; 12 rows/update 1.2 min per epoch but 4.47 after one epoch (fewer updates), so larger updates need more epochs.


## Bulk batch v4 result (2026-10-21): template collapse

Owner spot-check approved bulk-v1 (90 stratified rows, 2 N; 9,240 rows kept, families
`greeting:bulk-v1:morning` and `smalltalk:bulk-v1:play-sport` dropped). Bundle
`Datasets/workbench-chat-en-v4` = bulk + row-reviewed persona data (8,333/754/660). Kaggle GPU,
3 epochs, 8,333 updates: validation 7.89 -> best 4.19 at update 950 -> 5.06 final. Both the
final weights and best.pt answer the 22 fresh prompts with stitched template fragments
("I can't watch it, but I love go! What about you?") and lose the identity answers.
Cause: 9,240 rows but only ~800 distinct replies from 90 topic packs, so each reply is seen
~10x per epoch (audit A7). Row count is not diversity. A CPU run of the same bundle was resumed
and stopped at update 2,100 by another agent in this checkout (report state stopped_requested).
Next: `scripts/subsample_bulk.py` caps each reply at 2 occurrences (9,240 -> 528 rows, all
topics kept) for bundle v5; real progress needs many more distinct situations, not more rows.

## Bundle v5 result (2026-10-21): generic-reply collapse

`Datasets/workbench-chat-en-v5` = row-reviewed persona data + bulk-v1 capped at 2 uses per reply
(1,035 conversations). Local CPU, early stopping (patience 5) at update 1,600; best.pt at 1,350,
validation 3.96 (lowest of all runs). On the 22 fresh prompts best.pt answers almost everything
with "I'm Tantra! What's your name?" and no longer names its creator; only "Good evening!" is
handled well. v4 collapsed into stitched templates, v5 into one generic opener. Pilot v2
(`runs/serve-64k-v0`) remains the best chat model. Lower validation loss has not predicted better
replies in any run; future selection should use a scored fresh-prompt set, not loss alone.

## Reply-quality evaluation and checkpoint selection (2026-10-21)

`Tantra/eval/reply_eval.py` scores greedy replies (3-gram blocking) to 32 held-out prompts in
`docs/data/reply_eval.v1.json`: 0.6 for an expected cue, 0.2 for no forbidden phrase (forbidden
zeroes the case), 0.2 for no persona violation or repetition; the set mean is scaled by reply
diversity so one line for everything cannot score well. `leaked_prompts()` confirms none of the
prompts appear in any chat bundle. `conversation_train --select-by replies` scores replies at each
interval, keeps `best_reply.pt`, stops on reply-score patience and selects it as `best-base.pt`.

| Model | Reply score | Mean | Diversity | Passed (>=0.8) |
|---|---|---|---|---|
| Raw 64K base | 0.356 | 0.356 | 1.00 | 0/32 |
| Pilot v1 final | 0.431 | 0.431 | 0.84 | 9/32 |
| **Pilot v2 final (`runs/serve-64k-v0`)** | **0.531** | 0.531 | 0.84 | **12/32** |
| Pilot v3 best-by-loss | 0.450 | 0.450 | 0.84 | 8/32 |
| v4 best-by-loss | 0.393 | 0.438 | 0.72 | 2/32 |
| v4 final | 0.467 | 0.569 | 0.66 | 9/32 |
| v5 best-by-loss | 0.395 | 0.481 | 0.66 | 6/32 |
| v5 final | 0.494 | 0.494 | 0.84 | 6/32 |

Best-by-loss checkpoints scored below the final weights in v4 and v5: loss selection was choosing
worse models. v2 passes 10/12 identity prompts but almost no everyday-chat prompts (rain, food,
sadness, goodbyes) and often adds an unasked "an AI created by Atul Vij". Next data work should
add distinct everyday situations, and future runs should use `--select-by replies`.

## v6 (2026-10-22): first run selected by reply quality

Bundle `Datasets/workbench-chat-en-v6` = pilot-v2 data (30 old rows dropped by the persona gate) +
122 owner-approved v9 everyday conversations (each a distinct situation with its own reply); 519 rows.
CPU, 10 epochs, `--select-by replies --early-stopping-patience 10`, 1,567 updates; `best_reply.pt`
at update 1,500 became `best-base.pt`.

| Model | Reply score | Diversity | Passed |
|---|---|---|---|
| Pilot v2 | 0.531 | 0.84 | 12/32 |
| **v6 (reply-selected)** | **0.606** | **0.97** | **13/32** |

Gains: goodbyes, human/AI honesty, capabilities, fewer unasked "AI created by Atul Vij". Losses:
some identity prompts ("I'm Tantra, Tantra."). Still failing: most everyday topics (rain, food,
sadness, jokes) - 122 conversations moved the score but are not enough. Serving copy:
`runs/serve-64k-v1` (v6 best-base + tokenizer).
