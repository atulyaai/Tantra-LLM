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
