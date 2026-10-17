# Larger balanced pretraining stage

The visible Command Prompt runs `Tantra.data.continue_pretraining` for the
64K, 57.16M-parameter candidate. It prepares 19M new training tokens (9.5M
Hindi and 9.5M English), then waits for the active 1M-token pilot. Together the
stages target about 20M training tokens from deduplicated source documents.
Actual supervised counts exclude dropped incomplete windows and are logged.

The new corpus excludes every pilot attribution family and exact source-text
hash, across training, validation and test. It records the excluded manifest
hash. Source-group splitting and tokenization remain frozen; test examples are
never used for training. This is exact deduplication, not a claim that all
semantic or translated near-duplicates have been detected.

Automatic continuation requires a finished pilot and finite lower validation
loss in both Hindi and English. If either gate fails, the new worker stops with
an explicit blocked status. It also stops if the pilot process fails/disappears,
data quotas are short, or the wait exceeds eight hours. It does not silently
start another pilot or bypass review.

Once gates pass, the worker snapshots the pilot-trained base/tokenizer into its
own source-model directory. This new data stage starts a fresh AdamW optimizer;
it is weight continuation, not an exact optimizer resume across changed data.
Within the stage, existing atomic optimizer/RNG/cursor resume is supported.
Four CPU threads, 8 GiB process / 2 GiB free-system-memory checks, disk checks,
and checkpoints every ten updates remain in effect. There is no deployment.

At approximately 400 supervised targets/second, 19M targets alone require over
13 hours; validation, checkpointing and other desktop work add overhead. This
is not an eight-hour completion promise. Hinglish expert/conversation data
remain a separate review and training task.

```powershell
python -m Tantra.data.continue_pretraining --output runs/pretraining-20m-v1 --pilot runs/vocabulary-64k-v1 --initial-model runs/model-64k-initial-v1 --new-tokens 19000000
Get-Content runs/pretraining-20m-v1/status.json
```

Do not start a duplicate worker against an active worker.lock. Live pilot
progress remains runs/vocabulary-64k-v1/base-pilot/metrics.jsonl. Continued-stage
progress will be runs/pretraining-20m-v1/base-training/metrics.jsonl. Source
models and earlier checkpoints are preserved.

Validation: seven relevant tests passed, including prior-family/text exclusion,
immutable corpus preparation, exact same-stage resume, and rejection of
unfinished/non-improving/non-finite pilot results. The existing full suite had
51 passing tests before this stage was added.
