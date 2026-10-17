# Public data preparation and sequential training

## Current scope

Use Hindi and English for the base (50/50 by token budget, per the updated balanced-language scope). Keep Hinglish
conversation data separate. Other language specialists remain deferred. Preserve
all 164 taxonomy IDs; a catalog entry is not a dataset or a trained capability.

`data_sources.lock.json` pins two Wikipedia shards and OpenAssistant ready trees
by repository commit, file byte size and published LFS SHA-256. Initial downloads
are 589,325,646 bytes, below the 50 GB initial data budget. Acquisition retains
dataset cards and validates complete file hashes after bounded byte-range reads.
These are public downloads; no paid generation or compute is involved.

- Wikipedia Hindi/English: encyclopedia text for next-token base training, not
  reviewed conversational answers. Retain article URLs, titles, text hashes and
  source revisions in the generated attribution file. The dataset card specifies
  CC BY-SA 3.0/GFDL and links source-specific terms. The repository MIT license
  does not replace dataset licenses.
- OpenAssistant: Apache-2.0 conversation trees with community annotations.
  Reject deleted, synthetic, negatively/unreviewed messages and spam/PII/language
  mismatch labels. Keep root-to-response paths and original tree family IDs.
  Only Hindi/English records pass the initial language filter. No Hindi coverage
  is promised: the published OASST1 language inventory does not list Hindi.
- Hindi/Hinglish instruction sources with missing license metadata are not
  silently admitted. Translation-derived or generated answers need fluent review.

## Reproduce

```powershell
python -m pip install -r requirements.txt
python -m Tantra.data.public_sources
python -m Tantra.data.conversation_sources prepare --output runs/public-campaign-v1/review-queue
python -m Tantra.data.prepare_corpus --output runs/public-campaign-v1/corpus-1m --train-tokens 1000000
python -m Tantra.data.train_base_pilot --corpus runs/public-campaign-v1/corpus-1m --output runs/public-campaign-v1/base-pilot
# Explicit resume from the last atomic healthy optimizer checkpoint:
python -m Tantra.data.train_base_pilot --corpus runs/public-campaign-v1/corpus-1m --output runs/public-campaign-v1/base-pilot --resume
```

Or, while the source acquisition command runs, launch the sequential job:

```powershell
python -m Tantra.data.public_training_job --output runs/public-campaign-v1
Get-Content runs/public-campaign-v1/status.json
```

The job waits up to 90 minutes for source files, validates their contents during
preparation, builds review queues, prepares the bilingual pilot, then trains the
base for one pass. It stops before unsupported expert training/promotion. It is
not full pretraining, an overnight scheduler, or proof that all 161 missing
expert datasets have been supplied.

## Review workflow for experts

The review queue index includes every taxonomy ID, suggested row counts and
explicit `needs_source`/`needs_review` states. Conservative keyword suggestions
are NOT approved expert labels. Ambiguous items go to `unassigned.jsonl`.
Original source community ratings do not certify correctness or Tantra identity.
Do not train the queue or relabel it automatically as reviewed.

Read each candidate's question and answer, verify correctness and ownership,
check independent source families and sensitive content, then supply real
row-specific decisions in JSONL:

```json
{"record_id":"exact queue record hash","status":"approved","expert":"greeting","reviewer":"actual reviewer or transparent AI review identifier","rationale":"Concrete correctness and scope assessment"}
```

For Hindi/Hinglish additionally require `fluent_language_review: true` backed
by an actual fluent review; safety requires `domain_review: true` backed by
appropriate review. Do not manufacture those attestations. Reject examples
with `status: rejected`. Source queues are hashed; edits invalidate approval.

```powershell
python -m Tantra.data.conversation_sources approve --queue runs/public-campaign-v1/review-queue --decisions reviewed-decisions.jsonl --output Datasets/workbench-conversation-v1/incoming
python -m Tantra.data.expert_workbench --out Datasets/workbench-conversation-v1 prepare
```

The existing workbench enforces 1000/100/100 independent train/validation/test
families. Missing counts remain blocked; paraphrases and numeric/template
variants are not invented to hide shortages. Review diversity and language
coverage even when raw counts pass. Use distinct experts only when their held-out
benefit justifies them.

## CPU base pilot and quality gates

Preparation streams Parquet batches and writes little-endian uint32 token files,
with exact-text deduplication before accepting a document. Article families stay
in a deterministic 90/5/5 split. Per-language token quotas reserve validation and
test data. All shards, tokenizer and manifests have checksums. This first sample
uses one shard per language and is not a representative world-knowledge corpus;
semantic/cross-language duplicate detection is still future work.

Training uses mmap token files and contiguous 256-target windows without padding.
It benchmarks microbatches 1/2/4 with backward passes, chooses the fastest within
memory checks, and accumulates toward 4096 targets per update. The last update
can be smaller. Token counts are actual supervised next-token positions, not
padded lengths or claimed corpus sizes. No repeated epochs inflate unique data.

Atomic checkpoints store weights, optimizer, constant-LR scheduler metadata,
Python/NumPy/Torch RNG, cursor, token counts, configuration and source hashes.
Checkpoint every 100 updates or ten minutes, plus completion/clean interruption;
the exact-resume test uses a small real transformer. One latest resumable
checkpoint is retained; a best evaluated base is separate. Three-version
retention, corrupted-checkpoint fallback and automatic recovery are not yet
implemented. Memory limits are 8 GiB process RSS and at least 2 GiB free system
RAM, checked at optimizer boundaries; transient peaks are not a hard OS limit.
On an error, preserve the last healthy checkpoint and write diagnostics.

Validation reports Hindi and English loss on fixed validation windows every
50 updates and at completion. Test data remains unused for model selection.
A lower loss is not a conversation, reasoning or world-knowledge score. The
base output is separate and has no transferred expert weights: changing the
base requires expert retraining or re-evaluation against that exact base.
No 210M/1B model is launched without further feasibility and quality checks.

For a single durable worker, use `python -m Tantra.data.public_training_job --output runs/public-campaign-v1 --fetch`. It resumes verified byte ranges, then prepares data and trains. Do not launch a second worker against an active lock.
