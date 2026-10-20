# Roadmap: CPU chat with many routed experts

Target machine: 4-core i3, 16 GB RAM, no usable GPU. Target product: one shared
64K-vocab base plus many rank-8 LoRA experts, 1-2 routed per request, served by a
single CPU chat command. Written 2026-10-20 by the planning session. It is a plan,
not a status report: nothing below is done unless it says "today".

## Where we are today (facts from the repo, not goals)

- Base: `runs/kaggle-output/base-training/base.pt`, 57M params, config
  d_model 512, 8 layers, 8 heads / 4 KV heads, mlp_mult 2.75, max_seq_len 512,
  vocab 65,536, LoRA targets q/v/o/down. Validation Hindi 2.64 / English 5.70.
  Probe answers are incoherent; it is not a chat model.
- Chat pilots: `python -m Tantra.conversation_train train --english-pilot` (v1-v3)
  learned identity facts but over-applies them and overfits a few hundred rows.
  All are `english_pilot_complete_unvalidated`, none deployed.
- Bulk data: `runs/conversation-bulk-v1/queue/` has `bulk-drafts.jsonl` (9,450 own-authored
  English rows), `filter-report.json` (all pass) and `spot-check.csv`, which still
  **awaits the owner's Y/N marks** (BULK_DATA_POLICY.md, >=95% Y needed).
- Experts: `Model/experts/*.pt` holds 20 rank-8 adapters for the OLD 28.8M base with a
  10,277-token tokenizer. Their shapes do not match the 64K base: they are unusable
  there and must be retrained, not converted. Keep the files.
- Inference: `Tantra/inference/inference_engine.py` `TantraEngine` calls
  `TantraModel.load(base, experts_dir)` (`Tantra/model/tantra_model.py`), which adds
  and loads **every** expert listed in `router_names` eagerly. Routing uses
  `router_ngram.pt` (`Tantra/model/ngram_router.py`) when present, otherwise the
  in-model `Router`; `set_active()` sets weights on every LoRALinear.
- Entry points: `python -m Tantra chat|ask|bench [--int8]` (`Tantra/main.py`).

## Size arithmetic (why RAM is not the hard limit)

Per expert, rank 8: rank x layers x sum(in+out) over q(512+512), v(512+256),
o(512+512), down(1408+512) = 8 x 8 x 4,736 = **303,104 params = 1.2 MB FP32**.
Base 57M FP32 is about 230 MB (embeddings are 33.5M of that); INT8 dynamic quant
of linear layers saves less than "4x" because embeddings stay FP32.
So 164 experts would add only about 200 MB. Lazy loading is about **startup time,
not paying for 164 copies of LoRA forward cost, and safety** (only load vetted,
compatible adapters), not about fitting in 16 GB. The real limits are
latency per token on 4 cores and the lack of reviewed data per expert.

Global budget for every milestone: process RSS <= 1.5 GB, first token <= 1.5 s,
>= 15 tokens/s greedy at 256-token context on the i3 with 4 threads. These are
targets to measure in M1, then tighten; they are not measured numbers yet.

---

## M0. Baseline benchmark on the new base (no experts)

- Goal: real CPU numbers for the 64K base before changing anything.
- Files: new `scripts/bench_cpu.py` (wraps `TantraEngine.benchmark`, records RSS via
  `psutil`, tokens/s, time-to-first-token, `torch.get_num_threads()`), writes
  `runs/bench/<tag>.json`. No change to `Model/`.
- Acceptance: `.venv\Scripts\python.exe scripts\bench_cpu.py --model-dir <dir with
  64K base.pt + tokenizer.json> --tag base64k-fp32` and `--int8` both write JSON with
  `tok_per_s`, `ttft_s`, `rss_mb`; numbers committed to this doc's table, not guessed.
- Budget: measure only. Dependency: none. Note: needs a model dir built by copying
  the 64K base and tokenizer into a NEW folder (e.g. `runs/serve-64k-v0/`).

### M0/M1 measured (2026-10-20, i3-9100F, 4 threads, `scripts/bench_cpu.py`, raw JSON in `runs/bench/`)

| Run | Load s | RSS loaded MB | Peak RSS MB | tok/s | TTFT s | Max expert switch s | Resident |
|---|---|---|---|---|---|---|---|
| 64K base FP32 | 1.24 | 511 | 428 | 40.8 | 0.034 | - | 0 |
| 64K base INT8 | 1.43 | 564 | 377 | 44.4 | 0.027 | - | 0 |
| 64K base INT8 layers + INT8 head | - | - | 538 | 61.0 | 0.024 | - | 0 |
| 64K base FP32 layers + INT8 head (`--int8` default) | - | - | 592 | 59.4 | 0.025 | - | 0 |
| 60 synthetic rank-8 experts, lazy (cap 8) | 1.11 | 639* | 435 | 40.2 | 0.029 | 0.030 | 6 |
| 60 synthetic experts, eager | 2.33 | 714 | 501 | 40.9 | 0.033 | 0.000 | 60 |
| 160 synthetic experts, lazy (cap 8) | 1.23 | 533 | 447 | 41.1 | 0.032 | 0.025 | 6 |
| 160 synthetic experts, eager | 5.07 | 836 | 623 | 39.0 | 0.030 | 0.000 | 160 |

Synthetic adapters are random weights: they measure memory and switching only, not quality.
INT8 of the transformer layers alone gives ~9%: the tied 64K-vocab output head is ~40% of each decode step and was
left in FP32. An INT8 copy of the head gives +45%. Top-1 agreement with FP32 on 484 chat test positions: head-only
94%, layers + head 83%, so `--int8` now quantises only the head and `--int8-full` adds the layers.
*first lazy-60 run shares the process with building the synthetic folder; the 160 run is clean.
M1 is implemented (`Tantra/inference/expert_cache.py`, `tests/test_lazy_experts.py`).

## M1. Lazy, on-demand adapter loading

- Goal: start with zero experts in RAM; load an adapter file on first route hit;
  keep an LRU of at most N (default 8) resident; unload the rest.
- Files: `Tantra/model/tantra_model.py` (new `TantraModel.load(..., lazy=True)` that
  registers names/ranks from an experts manifest without calling `add_expert`;
  `ensure_expert(name)` / `evict_expert(name)` using existing `load_expert_state` /
  `remove_expert`; `LoRALinear.forward` must iterate only `active` experts, never all
  registered), new `Tantra/inference/expert_cache.py` (LRU, file SHA-256 + shape check
  against base config, refuses mismatched rank/dims instead of silently skipping as
  `load_expert_state` does today), `Tantra/inference/inference_engine.py`
  (`stream()` calls `ensure_expert` for routed names before `set_active`).
- Acceptance: new `tests/test_lazy_experts.py` with a tiny model and 30 fake experts:
  after load, 0 experts resident; after routing to 2, exactly 2 resident; after 10
  distinct routes with cap 8, 8 resident; generated logits with lazy == eager for the
  same active set (max abs diff < 1e-5); an expert with wrong shapes raises.
  `python -m pytest tests -m "not slow" -q` stays green.
- Budget: load one 1.2 MB adapter < 50 ms from SSD; RSS growth <= 2 MB per resident
  expert. Dependency: none (can run in parallel with M0).
- Honest note: the "legacy inference eagerly loads adapters" line in README/HANDOFF
  changes only after this merges and tests pass.

## M2. Chat-capable base (English) from the bulk batch

- Goal: a full-model English chat candidate that greets, small-talks and answers
  identity questions without over-applying identity.
- Blocked by: **owner marks `runs/conversation-bulk-v1/queue/spot-check.csv`**, then
  `python -m Tantra.data.bulk_review approve`. Below 95% Y, fix the generator and
  draw a new batch; nobody else may mark it.
- Files: no new trainer. Build a NEW bundle `Datasets/workbench-chat-en-pilot-v4` =
  approved bulk rows + the row-reviewed identity/persona rows (v3-v8, persona rules
  as in HANDOFF; drop superseded v1/v2 identity). Cap identity at <= 15% of train
  families. Report rows AND families (bulk topics share families).
- Train: `python -m Tantra.conversation_train train --english-pilot --epochs 2
  --bundle Datasets/workbench-chat-en-pilot-v4 --source runs/kaggle-output/base-training
  --output runs/chat-en-pilot-v4`. CPU estimate from v3 (918 updates in a few hours):
  ~9.9K rows x 2 epochs / 3 per update is about 6.6K updates, i.e. an overnight CPU
  run, or Kaggle free GPU via `scripts/kaggle/` (free tier only; no paid compute).
- Acceptance: (a) English val loss below v3's 3.14 on the NEW frozen validation set
  (report as not comparable across sets); (b) `scripts/compare_chat_pilot.py` on a
  fixed 40-prompt fresh set (new file `runs/eval-prompts/chat-en-40.txt`, never
  trained on) scored by the owner: >= 30/40 acceptable, <= 2 unasked identity/AI
  mentions, 0 "Good morning" to evening/bye. Owner scoring is a human step;
  automated loss alone does not pass M2.
- Budget: training RSS <= 8 GB (HANDOFF rule); inference same as M0.
- Hindi/Hinglish: still blocked by the fluent-reviewer gate; M2 is English only and
  must be labelled so. Hindi capability stays at base level.

## M3. Freeze the serving base

- Goal: one immutable serving base that all new experts train against.
- Files: `runs/serve-64k-v1/` with `base.pt` (M2 winner, or the raw 64K base if M2
  fails), `tokenizer.json`, `manifest.json` (SHA-256 of base and tokenizer). Never
  write into `Model/`.
- Acceptance: `scripts/bench_cpu.py --model-dir runs/serve-64k-v1` meets the global
  budget or the gap is recorded; hash in manifest matches file.
- Dependency: M2 result (or explicit decision to proceed without it).
  Rule: if the base changes later, every expert is retrained (adapters are per-base).

## M4. Retrain existing experts against the frozen base

- Goal: compatible rank-8 adapters, one candidate at a time, each beating the base on
  its own held-out data.
- Files: `Tantra/data/train_prepared.py` `train_one(... model_dir=runs/serve-64k-v1,
  output=runs/expert-64k/<name>)`; add a guard that refuses a model dir whose
  manifest hash differs from the bundle's recorded base hash. No new pipeline.
- Acceptance per expert: `python -m Tantra.data.expert_workbench train --expert <name>
  --model-dir runs/serve-64k-v1 --output runs/expert-64k/<name> --rank 8 --threads 4`
  then `--release-check`: held-out answer score > base, and no regression > 1 point
  on the M2 40-prompt chat set. Undersized bundles stay `undersized_pilot_unvalidated`.
- Budget: <= 10 min CPU per 300-step pilot (identity pilots took 82-158 s).
- Which old experts can be retrained now: only those with reviewed bundles
  (identity_self, greeting, smalltalk/farewell English). Most old raw files are filler
  (HANDOFF) and must NOT be retrained just because a file exists.

## M5. Router for 1-2 experts per request, with its own evaluation

- Goal: pick <= 2 experts or "base only"; measure it separately from answer quality.
- Files: `Tantra/model/ngram_router.py` (add an explicit `none` class so chit-chat can
  route to base only; keep `route_probs_to_experts` max 2), new
  `Tantra/eval/router_eval.py`, router trained only on TRAIN-split prompts of deployed
  experts; saved as `runs/serve-64k-v1/router_ngram.pt` with the expert list.
- Acceptance: `.venv\Scripts\python.exe -m Tantra.eval.router_eval --model-dir
  runs/serve-64k-v1 --split test` reports top-1 accuracy, top-2 recall and confusion
  matrix on held-out prompts. Gate: top-2 recall >= 0.90, top-1 >= 0.80, and wrong
  routing must not make answers worse than base-only on the M2 prompt set.
  Paraphrase families stay in one split.
- Budget: routing < 5 ms, < 5 MB RAM.
- Dependency: >= 3 deployed experts from M4; a router over 1-2 experts is not a test.

## M6. End-to-end CPU chat command with latency numbers

- Goal: `python -m Tantra chat --model-dir runs/serve-64k-v1 --lazy --int8
  --max-experts 2` works interactively and prints route + latency per turn.
- Files: `Tantra/main.py` (flags `--lazy`, `--max-resident`, `--show-route`),
  `Tantra/inference/inference_engine.py` (per-turn timing: route, adapter load, TTFT,
  tok/s), `scripts/bench_cpu.py` (`--chat-script` replays 20 fixed multi-turn chats).
- Acceptance: `scripts/bench_cpu.py --model-dir runs/serve-64k-v1 --lazy --int8
  --chat-script runs/eval-prompts/chat-en-20.jsonl` writes p50/p95 TTFT, tok/s, peak
  RSS, and expert loads per chat; meets global budget; table pasted into README.
  A slow test in `tests/` runs one scripted turn on a tiny model end to end.
- Dependency: M1, M3; M5 for routed (otherwise `--experts` forced manually).

### M6 status (2026-10-20)

Implemented: `python -m Tantra --model-dir <dir> chat|ask ... [--eager] [--max-resident N] [--max-experts K] [--timing]`
loads experts on demand by default and prints per-turn route, expert-load, first-token and tok/s numbers
(`TantraEngine.last_timing`; test `tests/test_engine_lazy.py`). Smoke run on `runs/serve-64k-v0` (chat pilot v2
weights, no experts, 4 threads): route < 1 ms, first token ~0.04 s warm (0.65 s on the first call), 34-37 tok/s.
Not yet done: the `--chat-script` p50/p95 replay and a frozen M3 serving base with retrained experts.

### M3/M4 first experiment (2026-10-20): what experts need

Rank-8 adapters, 300-400 steps, CPU, `Datasets/workbench-chat-en-pilot-v3` splits (`runs/experts-64k-*`):

| Expert | On chat-tuned base (pilot v2) val loss | On raw pretrained base val loss |
|---|---|---|
| identity_self (12 val rows) | 3.29 -> 3.52 (worse) | 6.77 -> 3.40 (better) |
| greeting (5) | 3.84 -> 4.13 (worse) | 8.23 -> 4.78 (better) |
| smalltalk (1) | 6.24 -> 7.34 (worse) | 6.94 -> 7.81 (worse) |
| farewell (1) | 7.86 -> 9.05 (worse) | 8.91 -> 9.84 (worse) |

`scripts/build_serving.py` assembled `runs/serve-64k-raw-v1` (raw base + identity_self + greeting +
n-gram router with a base-only `none` class; router test top-1 0.78 / top-2 0.89 on 27 rows), and
`scripts/compare_serving.py` compared it with the single chat-tuned model on 22 fresh prompts. Routing
worked (identity questions -> identity_self, greetings -> greeting), but answers were worse than the
single chat-tuned model, and anything routed to the raw base alone was incoherent.

Conclusions: (1) the serving base must itself be chat-tuned; (2) an expert only helps when trained on
data the base has not already absorbed; experts trained on the base's own SFT data overfit and are kept
disabled. So the order is: chat-tune the base on the large general batch (bulk-v1 after the owner's
spot-check), freeze it (M3), then train experts only on distinct domain data (M4/M7).
Fixed on the way: when the router's top class is `none`, the request now runs on the base alone
(previously the losing expert was applied at its small weight).

## M7. Build new experts in priority order

Order (all English first; Hindi/Hinglish rows need a named fluent reviewer):

| # | Expert | Why first | Data needed (gate 1000/100/100 families per DATASET_SPEC) | Blocker |
|---|--------|-----------|------------------------------------------------------------|---------|
| 1 | identity_self (persona) | product identity | row-reviewed only; have ~160 train families | more distinct situations, owner review |
| 2 | greeting / smalltalk / farewell | most chat turns | from bulk batch + v5/v7 | owner spot-check of bulk-v1 |
| 3 | clarification | stops guessing on vague asks | new own-authored batch, ~1K families | generator + spot-check |
| 4 | safety_refusal | required before any wider use | row-reviewed; careful wording | owner row review |
| 5 | gk_math (with calculator tool) | verifiable answers | generated + exact verifier (workbench seed) | calculator wins are not neural accuracy |
| 6 | summarization | useful, measurable | own-authored source texts + summaries, no copied text | licensing/provenance |
| 7 | lang_hi / translation | project language goal | Hindi rows with fluent_language_review | **no fluent reviewer recorded yet** |

Keep scopes distinct (no shared rows between 2/3/4); one candidate adapter at a time;
a new expert is deployed only after M4's held-out gain and M5's router check. Expect
most of the 164 taxonomy IDs to stay specifications for a long time.

## What is blocked by humans or data (no code can unblock these)

1. Owner Y/N marks on `runs/conversation-bulk-v1/queue/spot-check.csv` (blocks M2).
2. Owner scoring of the 40-prompt fresh chat set (blocks M2 pass, M4 regression check).
3. A named fluent Hindi/Hinglish reviewer (blocks all non-English experts and the
   balanced trilingual gate).
4. 1000/100/100 independent families per expert; current bundles are far below it, so
   every adapter until then is a labelled undersized pilot.

## Next 3 tasks for the implementer

1. **M1 lazy loading**: `ensure_expert`/`evict_expert` + LRU in
   `Tantra/inference/expert_cache.py`, shape/hash checks, and
   `tests/test_lazy_experts.py` (lazy == eager logits, resident counts, shape refusal).
2. **M0 benchmark script**: `scripts/bench_cpu.py` writing tok/s, TTFT and RSS JSON
   for the 64K base (FP32 and `--int8`) from a new `runs/serve-64k-v0/` folder.
3. **M5 prep**: add the `none` class and `Tantra/eval/router_eval.py` (accuracy, top-2
   recall, confusion matrix) with a tiny-fixture test, so the router can be measured
   as soon as M4 yields three compatible experts.
