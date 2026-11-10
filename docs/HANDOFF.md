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
