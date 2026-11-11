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

- Read `DATASET_SPEC.md`, `TRAINING_NOTES.md`, and the relevant card in
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

## Rung 1 — scripted conversation smoke (2026-11-11, all in ignored `runs/`)

The offline CPU pipeline was exercised end-to-end and verified per response:

seed → tokenizer (vocab ≈ 2.9K) → tiny base (~6.5M, 1500 steps, loss 0.28) →
seven rank-8 conversation experts (~114K params each) → n-gram + learned
router → `TantraEngine` → every reply of the 32-case held-out suite
(`docs/data/reply_eval.v1.json`) scored with `Tantra.evaluate.score_reply`,
recorded in `runs/rung1-v2/verification/responses.jsonl` + `report.json`.

**Champion (frozen in `runs/rung1-v2-champion/`): 22/32 cases ≥ 0.8,
mean case score 0.756, diversity 1.0, leak gate clean.**

Process lessons recorded for the next rung:

1. **Leak gate is mandatory.** One iteration trained on 24 verbatim held-out
   prompts (caught by `Tantra.evaluate.leaked_prompts`); that result was voided
   and quarantined. Rebuilt from a clean seed with paraphrase-only coverage;
   `leaked_prompts` must run before every verification claim.
2. **Retrain-to-retrain variance is ±3 cases** on this suite at tiny scale:
   per-case winners reshuffle between runs (17–22/32 observed for the same
   recipe). Single-case whack-a-mole does not converge; the recorded ceiling
   for the smoke model is the frozen champion above, and 32/32 is only
   reachable through verbatim leakage, which is prohibited.
3. **Router training reads only the first rows** (`expert_rows(limit=600)`,
   n-gram 1500): appended focused rows are invisible to it. Bundles must be
   rewritten focused-first (the runs/rung1-v2 data was).
4. **30M "small" preset underfit this CPU budget** (2400 base steps, 800-step
   experts): 18/32 vs tiny's 22/32, and n-gram router accuracy collapsed to
   53% when duplicate-heavy bundles biased its counts. Revisit with a proper
   LR/warmup schedule before scaling.
5. Greedy decoding produces garbled template medleys on near-ties; short,
   consistent reply templates and focused-first bundles reduce (not remove)
   this.

This rung is **scripted bootstrap capability only** (`sc_generate` seed data,
own-authored templates): the checks prove the pipeline, persona facts
(Tantra / Atul Vij / Atulya AI), and per-response verification; they do not
prove natural-language conversation. Rung 2 (real reviewed workbench data
with family-grouped train/val/test splits) is required before any further
claim.

## Rung 2 — real reviewed data pilot (2026-11-11, all in ignored `runs/` + new
`Datasets/workbench-rung2*` bundles)

The DATASET_SPEC workbench pipeline was exercised end-to-end on the reviewed
conversation seed, English-first (pending Hindi/Hinglish drafts stay excluded
from training until fluent review; a test enforces this):

1. Seed extended with reviewed authored English exchanges: identity_self
   15 → 45 rows (company, creator-person/team, name/nature/provider variants,
   about/purpose, origin, multi-turn) and greeting 21 → 32 rows (how-are-you,
   first-time, weather, return, casual, multi-turn). Every new row passed
   `flags()` (no unsolicited questions in greeting replies, persona-clean),
   uniqueness, and the verbatim held-out prompt gate. Reviewer string is
   honest: "AI content review (agent-assisted), 2026-11-11; not independent
   human review". Counts: 139 reviewed English / 19 pending (recorded in
   docs/TRAINING_NOTES.md; test pin updated).
2. `quality_cli export` → `workbench_cli init/prepare` produced immutable
   prepared splits with sha256 checksums and family-grouped 90/5/5 hashing
   (`Datasets/workbench-rung2b`). 139 accepted rows, all experts still
   `needs_data` (spec minimum 1000/100/100).
3. Three rank-8 candidate adapters trained with `workbench_cli train
   --small-pilot` on the rung-1 tiny champion base (source model unchanged,
   `source_model_changed: false`): greeting val loss 5.73 → 4.18,
   identity_self 6.32 → 4.48, identity_limitations 10.35 → 8.36. Candidates in
   `runs/rung2-v1b/` (+ `runs/rung2-v1/` for identity_limitations).
4. Per-response verification of every val/test prompt at temperature 0:
   greeting 8/8, identity_self 7/7, identity_limitations 2/2 checks pass
   (flags clean, no service wording, name/creator facts correct, replies
   non-empty). Leak gate on the prepared train bundle: clean. Fast suite:
   125 passed + 1 skipped (baseline unchanged).

Process lessons for the next rung:

6. **The 90/5/5 family-hash split leaves most experts with empty val/test**
   at this scale; `train_one` correctly refuses them (`Every pilot split must
   contain reviewed examples`). Only experts with ≥1 val AND ≥1 test family
   can train. Growing reviewed data is the only sanctioned unblock — no
   split manipulation. All 10 conversation experts were unblocked by growing
   the reviewed seed 139 → 405 rows (greeting 50, identity_self 57,
   farewell 61, clarification 58, ...); `clarification` was the last (its
   first test family landed at family #30 of the batch plan).
7. **Small-pilot adapters are candidates only** (`undersized_pilot_unvalidated`):
   they pair with exact base/tokenizer hashes, are not deployed, and the
   release check (≥100 verified answers per split) is unreachable until the
   1000/100/100 thresholds are met. Router integration and regression checks
   remain outstanding by design.
8. **Candidate integration loses to the bootstrap volume at this scale.**
   Swapping all ten rung-2 candidates into a champion copy scores 16/32 vs
   the rung-1 champion's 22/32 (each candidate saw only 12–57 train rows,
   vs the rung-1 bundles' paraphrase-volume ×8 rewriting). The champion is
   retained; the reviewed-seed path needs closer to the spec's 1000-row
   target before its candidates can compete. Per-response verification also
   caught one adapter emitting an empty reply on a val row ("Howdy") —
   small-data brittleness that loss curves alone hid (val loss improved
   10/10 experts).

Final pilot state (all rank-8, all val loss improved, all per-response
checks recorded in `runs/rung2-v1*/<expert>.verification.jsonl`):

| expert | val loss | splits t/v/t | checks |
|---|---|---|---|
| greeting | 5.73 → 4.18 | 41/5/4 | 8/9 (1 empty val reply) |
| identity_self | 6.32 → 4.48 | 48/2/7 | 9/9 |
| identity_limitations | 10.35 → 8.36 | 12/1/1 | 2/2 |
| smalltalk | 6.18 → 3.76 | 20/2/4 | 6/6 |
| empathy | 8.14 → 6.63 | 17/2/2 | 4/4 |
| farewell | 5.22 → 2.36 | 57/2/2 | 4/4 |
| apology | 8.55 → 6.76 | 34/2/2 | 4/4 |
| humor | 10.30 → 8.53 | 29/2/6 | 8/8 |
| identity_capabilities | 8.99 → 7.53 | 40/2/1 | 3/3 |
| clarification | 9.44 → 8.83 | 54/2/2 | 4/4 |

Leak gate on prepared bundles: clean. Fast suite: 125 passed + 1 skipped
(test pin updated to the 405-row seed).

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
