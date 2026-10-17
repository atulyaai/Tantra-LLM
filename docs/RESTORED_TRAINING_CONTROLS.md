# Restored training controls: October 2026

Compared historical Tantra/train.py at bc8b8d1 (5 October) and the September
history with the 14 October architecture rewrite. Reused the current shared
trainer; did not transplant incompatible old model/optimizer internals.

Restored on new runs: model dimensions/actual parameters, training loss, learning
rate, gradient norm, tokens/sec, ETA, sampled memory, checkpoint messages and
per-language validation loss/perplexity/top-1/top-5. Token prediction accuracy is
not conversational correctness. Each language is reported separately.

The configured command remains:

```powershell
Set-Location 'F:\Tantra-LLM'
& '.\.venv\Scripts\python.exe' -u -B -m Tantra.pretrain --resume
```

128 validation windows are spread over each held-out shard. Checkpoint metadata
now records that protocol. Changing protocol resets baseline, best score and
patience at the resumed weights, avoiding comparisons against a different
validation set. Old baseline values do not become new-protocol measurements.

Optional early stopping is five consecutive non-improving evaluations in the
provided config; evaluation runs every 50 updates. Set patience to zero to
disable. This is an equal-language mean-loss criterion, not a release gate.

For a worker running this updated code:

```powershell
python -m Tantra.train stop
```

It checks the request at an optimizer boundary, saves optimizer/RNG/cursor and
exports a candidate. Wait for exit, then resume. Older already-loaded workers do
not acquire this feature through editing files. Checkpoint intervals remain ten
updates or ten minutes. Corrupt-checkpoint automatic fallback and three-version
retention are not restored yet. Source models and frozen datasets are preserved.

The constant AdamW LR remains 1e-4 with FP32 and four CPU threads. The existing
microbatch benchmark selects within memory gates; no claim that this is the
fastest possible architecture, precision or optimizer is made. Representative
conversation/routing tests and fluent multilingual data review are still needed.

Validation: full suite 54 passed, including real-transformer exact interrupted
resume, migration, data exclusion, review boundaries and stale-lock safeguards.
A bounded resume on the real 57,156,096-parameter model is run before continuing.
