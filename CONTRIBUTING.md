# Contributing to Tantra-LLM

Thank you for helping improve Tantra-LLM. The project is an experimental local
LLM implementation; contributions should make its behaviour more measurable,
reproducible, or reliable.

## Setup

```powershell
git clone https://github.com/atulyaai/Tantra-LLM.git
cd Tantra-LLM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest Tests -q
```

## Where code belongs

- Put reusable importable code in `Tantra/`.
- Put reusable command-line workflows beside their owning `Tantra/` module;
  keep destructive dataset actions explicit and separately named.
- Keep `main.py` focused on coordinating public CLI modes.
- Put tests in `Tests/` using `test_*.py` names.

Do not move a script into `Tantra/` merely for folder tidiness. Move reusable
functions first, then keep a thin script wrapper for the CLI command.

## Before opening a pull request

```powershell
python -m pytest Tests -q
python -m py_compile main.py Tantra\*.py webui\server.py
```

For training or performance changes, include:

1. Hardware, PyTorch version, threads, batch size, sequence length, and seed.
2. Dataset identity and held-out evaluation method (never commit private/raw
   data).
3. Parameter count, tokens/sec, validation loss, and a short qualitative
   evaluation when generation is affected.
4. A checkpoint compatibility plan if any model shape, tokenizer, or vocabulary
   changes.

## Rules for artifacts and safety

- Do not commit checkpoints, logs, model state, caches, API keys, or raw
  datasets. They are intentionally ignored by Git.
- Do not claim performance or model-quality improvements without a comparable
  measurement.
- Preserve compatible checkpoint loading or add an explicit converter.
- Keep WebUI and server changes local-first by default. Do not enable code
  execution or network-facing services without clear opt-in and tests.

## Pull requests

Use a focused branch, explain the problem and validation, and avoid combining
formatting-only rewrites with functional changes. Contributions are licensed
under the [MIT License](LICENSE).
