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

