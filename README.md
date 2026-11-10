# Tantra LLM

Tantra is a CPU-first bilingual language-model research project. The repository
is intentionally source-only: no local checkpoint, downloaded corpus, candidate
adapter, or generated run is kept in Git.

## Lean layout

```text
Assets/       static project images
Datasets/     local source data (created or downloaded locally; ignored by Git)
Model/        local tokenizer and model artifacts (weights ignored by Git)
Tantra/       application, model, training, inference, memory, and validation code
Tests/        automated checks
docs/         model, dataset, and training specifications
```

The only supported command interface is the package CLI:

```powershell
python -m Tantra --help
python -m Tantra info
```

Do not attach a tokenizer to weights with a different vocabulary. Do not train
or overwrite `Model/` in place: candidate data, adapters, and evaluations belong
in a separate local `runs/` directory.

## Fresh start

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest Tests -m "not slow" -q
```

Before making dataset or training changes, read [AGENTS.md](AGENTS.md),
[docs/HANDOFF.md](docs/HANDOFF.md), and [docs/DATASET_SPEC.md](docs/DATASET_SPEC.md).

## Cleanup

Preview safe local-artifact cleanup:

```powershell
.\clean_workspace.ps1 -Mode artifacts -WhatIf
```

Apply it to remove local datasets, runs, downloaded checkpoints, and generated
caches while keeping source code, Git history, and the virtual environment:

```powershell
.\clean_workspace.ps1 -Mode artifacts
```

To discard every local change and artifact and restore GitHub `main`:

```powershell
.\clean_workspace.ps1 -Mode fresh-github
```

This permanently deletes ignored local files, including local models and data.
