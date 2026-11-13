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

## Rung 3 - chat stage: multi-turn context holding (2026-11-11, all in ignored `runs/`)

Goal: keep a conversation — remember names, chain identity facts across turns,
shift topic mid-stream — verified per response on a new held-out suite.

- **New eval:** `docs/data/chat_eval.v1.json` — 16 multi-turn cases
  (history + final user turn, `any` cues, several with `none`).
  Scored with the same strict `score_reply`; commit contains only the eval,
  never training flows.
- **Baseline (rung-1 champion): 7/16, mean 0.6.** Failures concentrated
  exactly where context matters: name recall, mid-stream topic shifts.
- **Training volume:** 20 multi-turn flows appended ×2 to the champion's
  bundles (greeting/identity_self/farewell/smalltalk/empathy), manifest at
  196 entries, rebalanced. Retrained those 5 experts + router (router
  acc ~0.74, all expert losses improved).
- **New leak gate:** `chat_leak_check.py` (temp drivers dir) — no chat-eval
  final prompt verbatim as any training user turn, no full chat-eval
  conversation verbatim, plus the standing reply-eval gate. Final state:
  clean.
- **Leak catch (second time, same class):** the first volume script copied
  eval conversations verbatim as training flows; the gate refused and the
  partial appends were cleaned before any training. Lesson re-recorded:
  **eval turns must be paraphrase-distinct from every training turn** —
  teach the pattern, never the eval string. That iteration is void.
- **Champion frozen (`runs/rung3-champion/`): single-turn 22/32 (holds),
  chat 10/16, mean 0.7375.** Freeze criterion: chat strictly better than
  7/16 AND single-turn ≥22/32. Verification copies in
  `runs/rung3-champion/verification*/`.

Remaining 6 chat failures (all one root cause — weak in-context extraction
from history; replies answered from identity templates instead of the
conversation): `chat-name-recall-1/2`, `chat-return-memory` (user name in
history not recalled — one reply honestly said "I don't remember past
conversations", wrong scope), `chat-provider-challenge`,
`chat-farewell-after-help`, `chat-mood-flow`.

**Rung-3b bounded round (same gates): null result, champion retained.**
11 name-recall/memory flows appended (user gives name → recalled later;
thanks → goodbye; venting → accepted without pushing) and the same five
experts retrained. Verification: single-turn **20/32** (down from 22) and
chat **9/16 mean 0.6875** (down from 10/0.7375) — the extra volume regressed
both suites, so the freeze criterion failed and the rung-3 champion was
kept. Reinforces the ±variance lesson: at 405 reviewed rows and this data
budget, one targeted volume pass does not buy context recall; the extra
gated rows stay in the working data (`runs/rung1-v2/data`) as valid future
volume, and `runs/rung1-v2/model` was restored from the frozen champion.
No leak-gate failures this round.

Recorded ceiling: 10/16 chat, 22/32 single-turn. Next honest steps: scale
the reviewed seed toward the 1000/100/100 spec minimums (the gap rung 2
identified), then Hindi fluent review (user gate) before any Hindi claim.

**Seed scale-up batch 1 (2026-11-12): 405 -> 705 reviewed English rows.**
100 new families x 3 paraphrase variants (300 rows) authored with validator,
persona and leak gates clean (`docs/data/seed_v3_review_sheet.md`), approved
through the recorded user review gate, and merged with honest reviewer
attribution (AI content review, agent-assisted; not independent human
review). All 10 experts keep non-empty family-hash splits after the merge;
identity_limitations keeps only 33/1/4 train/val/test, the thinnest split.
Test pin updated to 705 reviewed / 19 pending.

**Seed scale-up batch 2 (2026-11-12): 705 -> 1002 reviewed English rows.**
99 more families x 3 variants (297 rows), weighted to the thin experts
(identity_limitations +33, empathy +27, smalltalk +27); same gates, same
approval route (`docs/data/seed_v3b_review_sheet.md`). The 1000-row pilot
minimum is met. Test pin 1002 reviewed / 19 pending. This is still a pilot
threshold, not release readiness: per-expert validation/test checks remain
far below the 100/100 quality gate, and Hindi/Hinglish still awaits fluent
review.

**Rung-3 scale round (2026-11-12): null result, champion retained.**
All 10 conversation experts retrained from the rung-3 champion base on the
1002-row bundle (rank 8, 800 steps, `--small-pilot`, `runs/rung3-scale/`).
Every candidate improved held-out validation loss (largest: clarification
9.55 -> 6.57, empathy 8.14 -> 5.32, identity_self 7.37 -> 4.27). Per-response
verification (expert forced, so adapter quality is judged separately from the
router) passed cleanly for only 4 of 10: farewell (10/10), smalltalk (9/9),
apology (10/10), identity_capabilities (9/9). The other six produced genuine
failures: greeting 2/27 (one unrequested greeting question, one creator
mention), identity_self 5/12 (identity template repeated instead of the asked
fact), empathy 1/7, clarification 3/10, humor 4/17 and identity_limitations
3/11 (repetitive/garbled output).

Two integrations were evaluated against the same held-out gate:

| integration | adapters swapped | single-turn | chat |
| --- | ---: | ---: | ---: |
| strict (improved + zero structural fails) | 4 | 19/32 | 8/16 (0.675) |
| all improved | 10 | 15/32 | 5/16 (0.5875) |
| champion control (same drivers today) | - | **22/32 (0.7812)** | **10/16 (0.7375)** |

Both integrations regressed both suites, so no freeze occurred; the champion
and `runs/rung1-v2/model` are unchanged (verified by file hashes). The
champion control reproduces its recorded failure lists exactly, which rules
out driver drift. Lesson recorded: **lower validation loss is not a selection
signal at this scale** — these adapters memorize the enlarged bundles and
drift into identity-template answers. Also recorded from route telemetry:
`apology`, `clarification` and `humor` are not routed at all in the deployed
champion (0 held-out route hits and no adapter files in the champion), so
their candidates were inert; `greeting` (25/27) and `identity_self` (11/12)
route on nearly every prompt and dominate behavior change.

**Rung-4 bounded short-step round (2026-11-12): null result, champion
retained, training on this bundle stopped.** The 7 experts the champion
actually deploys (greeting, identity_self, farewell, smalltalk, empathy,
identity_capabilities, identity_limitations) were retrained at 300 steps on
the same 1002-row bundle. Validation loss saturated (greeting 5.19 vs 5.84 at
800 steps; empathy 4.63 vs 5.32), so the drift is data-driven, not
step-count-driven. Forced verification: farewell, empathy,
identity_capabilities and identity_limitations clean; smalltalk 1/9;
greeting 2/27; identity_self 6/12. Gate results:

| integration | adapters swapped | single-turn | chat |
| --- | ---: | ---: | ---: |
| strict (clean experts) | 4 | 20/32 (0.7375) | 10/16 (0.75) |
| all improved | 7 | 15/32 (0.6375) | 7/16 (0.625) |

Chat mean edged up but single-turn fell below the 22/32 hold, so no freeze.
With two bounded rounds (800 and 300 steps) both failing, further retraining
on this bundle would only fit the held-out suite, so it is stopped. The
rung-3 champion stands as the measured best: single-turn 22/32 (0.7812),
chat 10/16 (0.7375).

**Champion deployed locally: `runs/deploy-champion/` (2026-11-12, ignored by
git).** A clean copy of the champion with a `SHA256SUMS` pin and a README;
re-running the held-out drivers against the copy reproduces 22/32 (0.7812)
and 10/16 (0.7375) with identical failure lists. It is intentionally
manifest-less: the champion adapters predate embedded base/tokenizer identity
hashes, so the strict version-2 serving path would refuse to assert provenance
that was never recorded. Run it with
`python -m Tantra --model-dir runs/deploy-champion chat --offline`.
Smoke check: "Hi!" -> "Hi! Nice to hear from you."; "What is your name?" ->
"My name is Tantra. What's yours?"; "Who made you?" -> "I was developed by
Atul Vij and his team at Atulya AI to be a helpful assistant." Known gap
still visible: "How should I address you?" answers the well-being question
instead of giving the name (same `name-2` suite failure).

**Selection, single-swap and engine checks (2026-11-12, all null).** Three
further attempts to beat the champion without a new base:

- Per-expert selection by held-out validation loss (channel: champion vs
  800-step vs 300-step; champion won only for empathy, candidates won every
  other expert): integration scored 17/32 single and 6/16 chat. Fitting the
  enlarged data better does not transfer to the suites.
- Single-expert swaps, one candidate at a time. Greeting (300-step) raised
  greeting quality on the bundle's unseen test split from 5/16 to 1/16
  structural failures, but single-turn fell to 21/32 because the router sends
  "Make me laugh with a joke" to greeting and the new adapter answers
  "I feel sad?"; chat stayed 10/16 (0.7375). Identity_self swap fell to 19/32
  and 7/16. Neither meets the hold, so neither is deployed.
- Engine-side context handling, tested without code changes: injecting an
  extracted user name as a system note changed nothing (10/16, name cases
  still 0.4/0.2), and routing on the last two user turns dropped chat to 8/16.
  The model ignores the context hints; this is capacity, not plumbing.

**Independent generalization check.** The champion was scored on the bundle's
test splits (54 rows, families never used for adapter selection):
12/54 forced structural failures (78% clean), concentrated in greeting
(5/16: identity-template intrusion and rule-breaking questions), identity_self
(3/7), and one each in farewell, empathy, identity_capabilities and
identity_limitations. The same class as the suite failures, so the result is
not an artifact of the 32+16 eval sets.

Conclusion: the 6.5M champion is the measured ceiling for this base, data and
CPU budget. Further suite gains now require a larger base (the 30M preset
previously underfit at 2400 steps and needs a real training budget) or
materially different data, not more rounds on this bundle. Router telemetry
shows the remaining failures are generation quality, not misrouting.

**Rung 5: rank, new data and the 30M base (2026-11-12, all null).** This round
tested the two paths that conclusion named, plus the one capacity lever never
tried. Nothing beat the champion; every number below was produced by the same
held-out drivers in the same run as a champion control (22/32 at 0.7812 and
10/16 at 0.7375).

- **Adapter rank.** Every previous round used rank 8. Smalltalk retrained at
  rank 16 for 600 steps cut validation loss from 5.44 to 3.55 and still scored
  21/32 + 10/16, losing `good-news`. Rank fits the same data better without
  transferring.
- **Materially different data.** Batch 3 adds 127 reviewed English rows
  (78 families, 58 of them multi-turn) to 1129 rows: multi-turn name recall,
  name-after-farewell, provider challenge, identity chain, address-me,
  farewell-after-thanks, mood flow, thanks-then-close, topical small talk
  (weather, food, music, sport, weekend, pets, tiredness) and capability/limit
  detail. The authoring pass refuses any user turn or full conversation that
  matches an eval case and any duplicate exchange. Retraining all seven
  deployed experts and integrating them scored 13/32 + 9/16, so the rule held:
  one adapter at a time.

  | single-expert swap | single-turn | chat |
  | --- | ---: | ---: |
  | identity_self (600 steps) | 17/32 (0.6875) | 9/16 (0.7) |
  | identity_self (250 steps) | 18/32 (0.6813) | 7/16 (0.6) |
  | smalltalk | 22/32 (0.7875) | 10/16 (0.7375) |
  | empathy | 21/32 (0.7625) | 10/16 (0.7375) |
  | greeting | 21/32 (0.7688) | 10/16 (0.7375) |
  | farewell | 21/32 (0.7562) | 9/16 (0.7) |
  | identity_capabilities | 22/32 (0.7812) | 10/16 (0.75) |
  | identity_limitations | 22/32 (0.7812) | 10/16 (0.7375) |

  No swap adds a passed case. The identity_capabilities swap ties at 10/16 with
  a slightly higher mean (0.75 vs 0.7375) and no case-level gain; it is recorded
  as a tie rather than frozen, because equal pass counts are not a new
  capability. The identity_self swap does fix `name-2` and
  `chat-provider-challenge`, but loses six single-turn identity cases, so the
  adapter overwrites narrow facts instead of extending them.
- **Larger base.** The 30M preset with all seven experts retrained on this
  bundle scored 17/32 + 5/16; the untouched same base scores 18/32 + 4/16. It
  stays underfit at this CPU budget, so scale alone does not buy the behavior.

**Root cause, measured directly.** Name recall is a copy-from-context task that
this base cannot perform. On the bundle's own held-out name-recall rows (never
trained on, same template family), the retrained identity_self adapter answers
"What did I say my name was?" with "Atul Vij." and "Your name is fact or call me
Tantra."; the champion answers "eyes? Atulya AI built me — I'm Tantra." Both
fail in-distribution, so the chat ceiling is not an artifact of the 16-case
suite. Disclose the dependency honestly: batch 3 targets behaviors the chat
suite measures, so it is not independent evidence on its own; the held-out
bundle rows above are the cross-check, and they still fail.

Champion retained unchanged. The durable artifacts from this round are the 127
reviewed rows, the rung-4 bundle (`Datasets/workbench-rung4`, ignored by git),
and `runs/rung5-multiturn/` / `runs/rung5-30m/` with training summaries and
per-swap gate verdicts. Next real lever: a base that can copy context, which
needs a genuine pretraining budget on CPU (or GPU), not another adapter round.

## Rung 6 — continued pretraining then chat SFT (2026-11-13, all in ignored `runs/`)

This round used the repo's documented pinned-corpus path instead of another
adapter change, because rung 5 showed the defect is the base's English, not the
adapters' rank.

**Corpus and throughput.** Fetched the pinned `wikipedia-en` (420 MB) and
`wikipedia-hi` (135 MB) shards with checksum verification, then built
`runs/rung6-corpus-6m` and `runs/rung6-corpus-16m` through
`prepare_corpus_cli --exclude-corpus`, keeping the mandated 50/50 hi/en token
budget (8M + 8M train tokens in the larger corpus, 444k val / 444k test each).
Measured steady-state CPU throughput: **~3,275 tokens/s** at microbatch 4.

**14.3M-token continued pretraining.** 3,500 updates (`runs/rung6-pilot-16m`,
exported base sha `47ad31ab...`). Held-out validation, spread-v1, 32 windows:

| language | loss before | loss after | perplexity before -> after |
| --- | ---: | ---: | --- |
| English | 11.51 | **4.10** | 99,505 -> 60.1 |
| Hindi | 15.33 | **0.90** | 4,543,741 -> 2.46 |

Hindi converges far faster because the 2,897-token vocabulary was built with a
Hindi-heavy budget; English remains the weak language. A 1.2M-token pilot
(`runs/rung6-pilot`) had already taken the base from word salad to readable chat
lines, so more tokens was the right axis.

**Adapters cannot recover a drifted base (measured).** Retraining all seven
deployed experts on the wiki base (`runs/rung6-adapted`) collapsed both suites
to **0/32 + 0/16**, with Wikipedia-style fragments in every slot. This is
recorded as a fact about LoRA on a shifted base, not as a tuning problem.

**Full-parameter chat SFT recovers conversation.** `runs/rung6-sft` (600 steps,
LR 1e-4 on the champion base) reached val 8.90 -> 4.26 but scored base-only
14/32 + 6/16 and 13/32 + 9/16 with champion adapters — null.

The pretrained base was then chat-SFT'd (`runs/rung6-base-fromwiki`, i.e. the
wiki base with the champion router registry spliced back in, since the
pretraining export writes an empty router). Best-checkpoint tracking was added
after the first attempt overfit: 3,000 steps drove train loss to 0.02 while val
rose 3.62 -> 5.13. With early stopping, the best epoch was **step 500, val
3.6173** (`runs/rung6-sft2`).

| candidate | single-turn | chat | note |
| --- | ---: | ---: | --- |
| champion (frozen) | **22/32 (0.7812)** | **10/16 (0.7375)** | reference |
| rung6-sft2, base only | 11/32 (0.5563) | 9/16 (0.7125) | most fluent base so far |
| rung6-sft2 + 7 retrained experts | 14/32 (0.65) | 4/16 (0.5375) | adapters hurt again |

Despite the lower suite scores, `rung6-sft2` is the first base whose free-form
English is grammatical on arbitrary prompts: "Hi! How are you?", "I am Tantra.",
"Atulya AI built Tantra.", "I can chat, explain simple things, and help draft
short text.", "Good night! Sleep well." Relevance is still weak — "What is the
capital of France?" answers "I am Tantra, a language model."

Six of seven experts validated *worse than the base they were trained on*
(e.g. identity_limitations 4.01 -> 5.85), so the adapter is now the thing that
degrades a good base. Champion stays; nothing new is deployed.

**What this rules out.** Not adapters, not rank, not routing, not full-parameter
SFT, not base size on its own, not 14M tokens of encyclopedia text. The measured
binding constraints are (1) token budget — 14.3M is still under-trained for even
5.2M parameters — and (2) the 2,897-token vocabulary, which forces rare words
into long byte-ish fragments. Rung 7 targets (1) first because it needs no new
vocabulary and no adapter work.

## Rung 7 — the copy circuit (2026-11-13, all in ignored `runs/`)

Rung 6 established that no adapter, rank, or routing change can make this base use
context. Rung 7 asked the narrower question the suites actually fail on: can a
5.2M-parameter model learn to copy a name from the conversation at all?

**Baseline measured first.** A 24-case probe (8 names x 3 ask shapes, names never
trained on) scored the frozen champion at **0/24**. It answers the user's name
question with its creator ("Atul Vij and his team at Atulya AI") or with a
no-memory disclaimer. Two of those names (Riya, Arjun) are the held-out chat-suite
names, so the suite's failure was reproduced independently of the suite.

**Rank-8 adapters still cannot do it.** Training a fresh identity_self adapter on
the bundle plus 768 curriculum rows (val 7.83 -> 3.66) still scored **0/24**. Its
errors were training names ("Anya" for Kavya), not the name in context. LoRA on a
base without the circuit memorises and cannot copy.

**Full-parameter training on the same rows also failed (0/24)** while reaching
curriculum val 8.73 -> 0.51. Replies then had the right shape — "Your name is
X." — but X was drawn from the 64 training names. With only 64 names,
memorisation and copying are indistinguishable and the optimiser picks the
cheaper solution.

**Forcing the circuit.** The decisive test trained on 4,000 generated syllable
names with 200 held out entirely, so memory cannot help. Copy accuracy on unseen
names climbed steadily, measured every 500 steps:

| step | 0 | 500 | 1000 | 1500 | 2000 | 2500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| copy accuracy (held-out names) | 0.00 | 0.00 | 0.025 | 0.15 | 0.40 | 0.625 |

Final 0.660 over 100 held-out names (`runs/rung7-induction`). The architecture can
implement context copying; earlier failures were a **data-volume and objective
problem**, not a capacity limit. Names here are generated strings used only as a
diagnostic task.

**Transfer to real names.** Chat tuning on the reviewed bundle plus the real-name
curriculum, from the circuit-carrying base, produced the first base that recalls
unseen real names: **11/24 then 13/24 (54%)** after a low-LR recovery pass
(`runs/rung7-combined` -> `runs/rung7-recover`). Correct answers now include
"Meera.", "Arjun.", "Yes, Nandita." and "Yes, Kavya." for names never trained on,
and the suite case `chat-name-recall-1` passes for the first time in any base-only
candidate.

**But chat regressed, so nothing is deployed.**

| candidate | single-turn | chat | unseen-name recall |
| --- | ---: | ---: | ---: |
| champion (frozen reference) | **22/32 (0.7812)** | **10/16 (0.7375)** | 0/24 |
| rung7-combined | 8/32 (0.4375) | 4/16 (0.5375) | 11/24 |
| rung7-recover | 8/32 (0.4813) | 6/16 (0.5875) | **13/24 (0.5417)** |
| rung7-adapt (seven experts on rung7-recover) | 14/32 (0.6188) | **10/16 (0.7750)** | 12/24 (0.50) |

The last row is the best composite so far: it matches the champion's chat count and
exceeds its chat mean, while recalling unseen names the champion cannot. It still
fails the freeze gate on single-turn (14 < 22), so the champion remains deployed.
Two attempts to raise single-turn on this base failed in a recorded way: a
bundle-weighted replay pass with a recall floor either raised validation loss from
the first evaluation or dropped recall below the floor, and no checkpoint beat the
starting point, so `rung7-recover` is a local optimum for that mixture.

The trade is explicit: buying the copy circuit cost 6 single-turn cases and 4 chat
cases. A polish pass weighted to the hard ask shape moved dev recall only
0.46 -> 0.50 with no suite gain, so the mixture is saturated at this size.

**What rung 7 establishes.** Copy-from-context is learnable here, which removes
"the architecture cannot do it" from the list of explanations. The deployment
blocker is now capacity: one 5.2M model at this token budget cannot hold fluent
chat, memorised persona lines, and a copy circuit at once. The honest next step is
more capacity or a much larger token budget, then repeat the same copy-then-chat
recipe; the recipe itself is now validated and recorded.

**Observed defect, recorded not hidden.** On `rung7-adapt`, the prompt
"Do you remember my name? My name is Meera." returned an empty string because
generation emitted a stop token immediately; `chat()` returns `""` in that case.
No empty reply appears in any 32-case or 16-case suite run across all candidates,
so this is a defect of that candidate rather than of the engine. It is listed
here because an empty reply is unacceptable in real use and must be checked
before any recall-capable model is deployed.

**Provenance note.** The name curricula are template-generated and marked
`not_reviewed` (`runs/rung7-name-curriculum/curriculum.json`,
`runs/rung7-induction/`). They are experiment data only: never merged into
`docs/data`, never counted as reviewed rows, and the 0.66 copy figure is a
synthetic-task diagnostic, not a conversation capability claim.

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
