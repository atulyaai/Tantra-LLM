# Validation evidence — 17 October 2026

## Automated checks

`python -m pytest tests -q -p no:cacheprovider --basetemp .venv/final-expert-tests`

Result: **38 passed in 21.15 seconds**, including the temporary-model end-to-end
training pipeline. New cases cover all 164 scopes, actual adapter budgets,
placeholder/malformed rejection, automatic answer verification, unique task
ownership, language matching, exact deduplication, case-sensitive JSON values,
family-separated frozen splits, immutability, output isolation and schema coverage.
`git diff --check` was also run before handoff.

## Data prepared locally

- 15,000 exactly verifiable examples; arithmetic belongs only to `math`.
- Arithmetic: 4,481 train / 250 validation / 269 test, in Hindi/Hinglish/English.
- JSON: 4,502 / 265 / 233, English serialization bootstrap.
- CSV: 4,515 / 252 / 233, English serialization bootstrap.
- Three pilot-ready datasets; 161 other taxonomy entries still need reviewed data.
- Template coverage is narrow (three arithmetic operator patterns, one JSON and
  one CSV serialization pattern). These counts do not certify broad expertise.

## Isolated pilot

Command:

```
python -m Tantra.data.expert_workbench train --expert math --model-dir Model --output runs/math-pilot-20261017 --steps 20 --rank 8 --threads 4
```

The candidate trained **303,104 adapter parameters**. Held-out loss decreased
from **10.9739 to 6.0926**. Reported training/after-validation time was **16.55 s**;
sampled process RSS reached approximately **496 MiB**. Setup and baseline
validation are outside that timing, and RSS sampling is not an OS peak-memory
measurement. No answer-accuracy/release check was run.

The artifact is `runs/math-pilot-20261017/math.pt`, marked `pilot_unvalidated`.
The report and artifact retain base, tokenizer and dataset hashes. Canonical
`Model/` was not overwritten; routing was not retrained. Keep the candidate paired
with its exact base/tokenizer and evaluate answers before any deployment.

## Remaining work

The existing direct neural model still produces weak answers. Review conversation
data first, expand independently checked task patterns, and evaluate every new
specialist against a baseline. Registered cards and saved weight files are not
validated skills. Full optimizer resume, bounded inference expert loading and
automatic repair/growth are not part of this restored implementation.