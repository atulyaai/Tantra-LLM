# Goal ladder: from a clean CPU core to a real talking model

Written 2026-11-10 after the intentional repository reset (`b00559a`).
This document separates what the current codebase can honestly deliver from
what competing with frontier models requires.

## Where we are

- The repo is a verified clean source baseline: full test suite green
  (126/126 including the slow test), working tree clean, no trained weights.
- `Model/` holds tokenizer artifacts only. There is **no runnable trained
  model** in the tree. Any "chat" needs weights that must be trained again.
- The offline CPU pipeline is intact and can rebuild a small, narrow pilot:
  scripted workbench curricula -> rank-8 expert adapters on a ~29M base
  (`python -m Tantra base` -> `expert` -> `chat`). This model can greet,
  answer a small fixed set of identity/farewell facts correctly under the
  strict shared checks, and nothing beyond that. It is not and cannot become
  a frontier competitor at 29M parameters on a CPU with synthetic seed data.

## The honest gap to "compete with frontier models"

1. **Scale.** Frontier systems are trained on trillions of tokens with GPUs.
   The repo's own plan for that was the Kaggle 1B-token GPU pipeline; its
   helpers were removed in the reset and must be re-established or rewritten
   before any scale step exists.
2. **Data.** Real conversational and knowledge ability comes from large,
   reviewed, provenance-backed corpora. Reusing public, checksum-verified,
   license-cleared sources (`docs/DATASET_SPEC.md`,
   `Tantra/pretrain.py`) is the sanctioned path. Fabricated
   paraphrase dumps and unverified imported data are prohibited and do not
   produce true capability anyway.
3. **Correctness.** Every answer correctness claim must pass the recorded
   strict checks (required facts present, wrong echoes absent) on held-out
   test rows that were never trained on. This bar can be met for narrow
   scripted domains today; it cannot be claimed yet for open conversation.

## Rung ladder (each rung requires independent held-out evidence)

1. **Rung 0 — done.** Clean verified core, tests green, workspace tooling
   fixed (`clean_workspace.ps1` now runs under `powershell -File`).
2. **Rung 1 — rebuild the CPU pilot.** Regenerate the scripted workbench
   bundles and train one rank-8 candidate at a time into `runs/`.
   Correctness is measured only by the shared checks on val/test.
3. **Rung 2 — real data base pilot.** Acquire the pinned public bilingual
   corpus, `prepare_corpus`, `train_base_pilot` on the 64K/29M+ configuration,
   and demonstrate per-language (hi/en, Hinglish reported separately) loss
   improvement on frozen validation.
4. **Rung 3 — chat stage.** Conversation training on reviewed data with
   validator-audited bundles, again only after held-out wins.
5. **Rung 4 — scale.** The Kaggle/GPU pretraining ladder at 1B+ tokens with
   the expanded tokenizer; deployment and adapter compatibility require
   evaluation, never assumption. This is the rung where "frontier knowledge"
   first becomes decidable, and it needs GPU compute and months of data work.
6. **Rung 5 — frontier comparison.** Only meaningful after rung 4, by
   benchmarking against published results, never by claim.

## Rules that stay in force at every rung

- Never modify `Model/` without independent held-out gains; never train on
  test data; one rank-8 adapter candidate at a time.
- No filler data, no manufactured examples, no unreviewed provenance.
- No correctness claim without the recorded strict checks.
- Offline by default for tests; no paid compute or generation.
