# Tantra handoff

## Current baseline

This repository is a clean source baseline. Local checkpoints, datasets,
candidate adapters, historical evaluation output, Kaggle helpers, and generated
conversation drafts were intentionally removed on 10 November 2026. `Model/`
contains tokenizer artifacts only; there is no runnable trained model.

The only command interface is `python -m Tantra`: the routing CLI
(`--help` lists `seed, tokenizer, base, expert, router, stage1, stage2,
checkpoint, chat, ask, bench, grow, info, eval, feedback, chats`), plus named
CLI entries from the flattened host modules (`python -m Tantra workbench_cli`,
`pilot_cli`, `candidates_cli`, `review_cli`, `router_eval_cli`, `reply_eval_cli`,
`quality_cli`, `conversation_sources_cli`, `dashboard`, and the other names
listed in `Tantra/__main__.py`). The four benchmark/build utilities run as
`python -m Tantra.bench_cpu|bench_peer_cpu|build_serving|subsample_bulk`.
Local generated artifacts belong in ignored `runs/`, `Datasets/`, and `Model/`
paths. Use `clean_workspace.ps1` to remove them safely or reset to
`origin/main`.

## Training rules

- Read `DATASET_SPEC.md`, `CONVERSATION_UPDATES.md`, and the relevant card in
  `docs/experts/` before authoring data.
- Use an immutable, provenance-backed dataset with frozen train/validation/test
  splits; never train on test data.
- Train one rank-8 adapter candidate at a time in a separate run directory.
- Require independent held-out improvement before deploying anything to `Model/`.
- Do not claim language, conversation, or knowledge quality without recorded
  evaluation. Hindi, English, and Hinglish must be evaluated separately.

## Goal ladder (written 2026-11-10 after the `b00559a` reset)

The offline CPU pipeline is intact and can rebuild a small, narrow pilot —
scripted workbench curricula, then rank-8 expert adapters on a ~29M base
(`python -m Tantra base` -> `expert` -> `chat`). Such a model can greet and
answer a small fixed set of identity/farewell facts under the shared strict
checks, and nothing beyond that. It is not, and cannot become, a frontier
competitor at 29M parameters on a CPU with synthetic seed data.

The honest gap to competing with frontier models:

1. **Scale.** Frontier systems train on trillions of tokens with GPUs. The
   planned Kaggle 1B-token GPU pipeline helpers were removed in the reset and
   must be re-established before any scale step exists.
2. **Data.** Real ability needs large, reviewed, provenance-backed corpora.
   Pinned public checksum-verified sources (`Tantra/pretrain.py`) are the
   sanctioned path; fabricated paraphrase dumps are prohibited and do not
   produce true capability anyway.
3. **Correctness.** Any answer-correctness claim must pass the recorded
   strict checks (required facts present, wrong echoes absent) on held-out
   test rows never trained on. Met for narrow scripted domains; not yet
   claimable for open conversation.

Each rung requires independent held-out evidence before the next:

0. Clean verified core, tests green — done.
1. Rebuild the CPU pilot: regenerate workbench bundles, one rank-8 candidate
   at a time into `runs/`, measured only by the shared checks on val/test.
2. Real-data base pilot: pinned public bilingual corpus through
   `prepare_corpus`/`pilot_cli`; per-language (hi/en, Hinglish separate) loss
   improvement on frozen validation.
3. Chat stage: conversation training on reviewed, validator-audited data,
   again only after held-out wins.
4. Scale: the GPU 1B+ token pretraining ladder with the expanded tokenizer.
   First rung where "frontier knowledge" becomes decidable; needs GPU compute.
5. Frontier comparison: only after rung 4, by benchmarking against published
   results, never by claim.

Rules at every rung: never modify `Model/` without independent held-out
gains; no filler data or unreviewed provenance; no correctness claim without
the recorded strict checks; offline by default for tests; no paid compute or
generation.

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
1 skipped; full suite including the end-to-end slow test = 126 passed, 1
skipped. After the redundancy cleanup the suite is unchanged. No model,
tokenizer, or dataset content changed; `Model/` artifacts are untouched.
