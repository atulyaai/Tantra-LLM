# Tantra handoff

## Current baseline

This repository is a clean source baseline. Local checkpoints, datasets,
candidate adapters, historical evaluation output, Kaggle helpers, and generated
conversation drafts were intentionally removed on 10 November 2026. `Model/`
contains tokenizer artifacts only; there is no runnable trained model.

Use `python -m Tantra --help` as the sole command interface. Local generated
artifacts belong in ignored `runs/`, `Datasets/`, and `Model/` paths. Use
`scripts/clean_workspace.ps1` to remove them safely or reset to `origin/main`.

## Training rules

- Read `DATASET_SPEC.md`, `CONVERSATION_UPDATES.md`, and the relevant card in
  `docs/experts/` before authoring data.
- Use an immutable, provenance-backed dataset with frozen train/validation/test
  splits; never train on test data.
- Train one rank-8 adapter candidate at a time in a separate run directory.
- Require independent held-out improvement before deploying anything to `Model/`.
- Do not claim language, conversation, or knowledge quality without recorded
  evaluation. Hindi, English, and Hinglish must be evaluated separately.

## Validation

After dependencies are installed, run:

```powershell
.\.venv\Scripts\python.exe -m pytest Tests -m "not slow" -q
```

The source tree is expected to remain clean after the suite, except for ignored
cache files.

## Package layout (flattened, 2026-11-10)

`Tantra/` has no subpackages. Configuration and the taxonomy live in the package
root; each former subfolder was folded into one host module:

| Host module | Contains |
|---|---|
| `Tantra/__init__.py` | ModelConfig, presets, taxonomy helpers, canonical `sha256` |
| `Tantra/train.py` | trainer, checkpoints, TantraModel, PEER, expert dataset/workbench, persona rules |
| `Tantra/pretrain.py` | public sources, corpus preparation, base pilot, continued pretraining, public training job |
| `Tantra/tokenizer.py` | tokenizer candidates, vocabulary migration, vocabulary training job |
| `Tantra/main.py` | ngram router, inference engine, expert cache, serving manifest (the `python -m Tantra` CLI) |
| `Tantra/evaluate.py` | router/reply evaluation CLIs |
| `Tantra/evaluate_conversation.py` | memory store and indexes + reservation CLI |
| `Tantra/dashboard.py` | bulk review, review sheets, instruction candidates, health monitor |
| `Tantra/conversation_train.py` | seed writers, conversation sources (prepare/approve), conversation quality |

Cross-host module-level imports form a DAG; the two edges that would cycle
(tokenizer -> pretrain/conversation_train helpers; vocabulary job -> write_status)
are imported lazily inside functions. Collision renames are permanent API names:
`workbench_cli`, `pilot_run`, `cs_prepare`/`cs_approve`, `review_cli`,
`reply_eval_cli`/`router_eval_cli`, `write_seed_conversation` and siblings.

Verification after the flatten: `pytest Tests -m "not slow" -q` = 125 passed,
1 skipped (equal to the pre-flatten baseline). No model, tokenizer, or dataset
content changed; `Model/` artifacts are untouched.
