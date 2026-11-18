# Tantra-LLM: family growth, expert composition, DNA compression and balanced runtime

**Status:** research architecture decision record and implementation backlog. The current locked product direction is native-first, with residual-branch growth first and width/depth expansion later; see [the authoritative roadmap](TANTRA_NATIVE_ARCHITECTURE_ROADMAP.md). This document does not claim that proposed experiments have run.

**Decisions — 2026-11-18**

| Area | Decision | Engineering interpretation |
|---|---|---|
| Model growth | Family | Maintain a versioned family of compatible model sizes; use controlled transformations and independent-size controls. |
| DNA compression | Both | Research a learned parameter/storage codec and DNA-inspired parameter/architecture sharing. Literal A/C/G/T encoding is not compression. |
| Tokenizer | 64K | Target 65,536 tokens. Existing tokenizer remains immutable; 64K stays candidate-only until migration, training and held-out gates pass. |
| Runtime | Balanced | One runtime contract with CPU, NVIDIA CUDA, AMD ROCm and optional platform backends selected by measured capability. |

## 1. Preserve the baseline

The current native Tantra model, tokenizer, taxonomy, expert adapters, router, training pipeline and historical run records remain supported. Do not overwrite Model/, runs/, Datasets/, docs/data/ or existing reports/checkpoints.

Every experiment writes to a new ignored run directory and records source commit, model/base SHA-256, tokenizer SHA-256, config, dataset manifest and split hashes, source/license/revision, code/dependency versions, seed, optimizer/scheduler, hardware/runtime, parameter counts, disk bytes, peak RAM/VRAM, tokens/s, latency and evaluation outputs.

Candidates are never promoted by training loss alone. Promotion requires held-out task scores, generated-answer checks, no regression beyond a pre-registered tolerance, reproducibility and artifact provenance.

## 2. Model-family growth

### 2.1 Family ladder

| Tier | Target | Purpose | Status |
|---|---:|---|---|
| edge | 0.8B–1B | low-memory local inference | target |
| compact | 2B | control and specialist host | target |
| standard | 4B | primary local general model candidate | target |
| quality | 7B–9B | quality/latency comparison | target |
| large | 14B+ | distributed/high-memory research | later |
| sparse | 100B–1T total | MoE research; report active and total parameters separately | research horizon |

These are target tiers, not claims that this repository can currently train or serve them. A 1T sparse model still needs storage/distribution for all experts; active parameters per token are not total storage.

### 2.2 Three controlled growth tracks

1. **Residual-branch growth first (locked).** Prove the native expert can add a parallel branch behind a zero output projection or zero residual gate while preserving the original path. Train the new branch with distillation + replay + new data; keep the source expert immutable.
2. **Depth and width growth later.** Harden the existing depth-growth operation and design width mapping only after the residual-branch proof. Each operator needs its own architecture-specific fixed-batch equivalence/control test.
3. **Distilled family siblings and progressive sparse experts (later).** Train sibling checkpoints from a common teacher and data contract; add experts incrementally and calibrate/distill the router. Existing validated experts remain frozen unless a declared joint-training experiment justifies changing them.

### 2.3 Growth training protocol

For each source checkpoint:
1. Freeze and hash the source checkpoint, tokenizer, config and evaluation set.
2. Capture baseline generations, exact-task metrics, logits and hidden-state probes.
3. Create a new candidate directory; never mutate the source.
4. Apply one transformation only: depth, width, expert count or distillation.
5. Verify shapes, finite tensors, optimizer state and function preservation where promised.
6. Train in stages: new parameters first at a higher LR; inherited parameters frozen or at lower LR; then a short joint recovery stage.
7. Use replay/anchor examples and KL or feature-preservation loss to reduce forgetting. Report these losses separately.
8. Evaluate old and new capabilities on untouched tests. Run generated-answer checks, not only perplexity.
9. Promote only if the candidate clears the pre-registered gate. Otherwise retain the source and archive the negative result.

A structural transformation can preserve the initial function exactly; training can still change that function. A larger model does not inherit additional intelligence merely by allocating random weights.

## 3. Small-model collaboration and domain experts

Use a hierarchical ensemble, not an indiscriminate swarm:
- one generalist handles ordinary chat and decides whether specialist help is needed;
- specialist models/adapters own narrow domains and have independent capability cards and test sets;
- a planner/router selects one or more specialists using intent, confidence, cost and latency;
- specialists return structured outputs with evidence, confidence and provenance;
- a verifier checks factual, mathematical, schema or code claims where deterministic checks exist;
- a synthesizer combines results and resolves conflicts; abstain or clarify when confidence is insufficient.

Execution modes: single, parallel_consensus, pipeline and fallback. Each has hard time/token budgets. Do not run every expert on every prompt.

Enable a specialist only when it beats the generalist baseline on family-disjoint held-out tests without unacceptable regressions. Measure router precision/recall, incremental specialist accuracy, calibration, conflict rate, verifier pass rate, latency and total compute. Taxonomy entries do not require one model or tensor per ID.

## 4. Tokenizer: 64K candidate with compatibility boundary

Target exactly 65,536 tokens, but pin the current tokenizer and all checkpoints/adapters to their original tokenizer hash.

Compare the current baseline, fresh 16K/32K/64K candidates and the existing-ID-preserving 64K extension path. Measure Hindi, English and Hinglish separately, with fluent review for Hinglish; also test code, identifiers, numbers, punctuation, mixed scripts, rare words, Unicode normalization and malformed input. Report token fertility, bytes/token, round-trip fidelity, vocabulary distribution, throughput and downstream quality.

Migration invariants:
- every existing token ID remains identical;
- special-token IDs and tokenizer processing remain identical;
- each new token has deterministic decomposition into old tokens;
- old embedding rows remain byte-identical at initialization;
- new embedding rows use a documented initializer and are trained;
- tied/untied output projection and softmax changes are tested;
- every adapter is bound to exact base and tokenizer hashes and re-evaluated or retrained.

A 64K vocabulary increases embedding/output parameters and can increase softmax cost. Select it only if held-out token efficiency improves across languages and the migrated model recovers task quality. A smaller vocabulary can win.

## 5. DNA compression — both research meanings

DNA compression is the project name for a learned, content-aware compression system. It is not a claim that encoding bytes as A/C/G/T reduces entropy.

### 5.1 Storage codec track

Build an offline lossless container first: tensor names, shapes, dtypes, offsets, checksums and model/tokenizer/config identity; blockwise analysis; exact round-trip codec baseline; optional shared codebooks/dictionaries, repeated-block deduplication, delta coding and entropy coding; streaming decode and random block access; corruption detection and safe fallback.

Then test lossy formats separately: per-tensor/per-channel INT8 and INT4; groupwise quantization; learned codebooks/vector quantization and low-rank residuals; sensitivity-aware bit allocation; optional learned entropy models for quantized symbols.

Report actual disk bytes, decode RAM, load time, throughput, quality drift and exact reconstruction. Lossless must be bit-exact. Lossy candidates must pass task and generation gates. Do not compare theoretical bit width with actual file size and call it a win.

### 5.2 Architecture/parameter-sharing track

Research shared base blocks with expert-specific low-rank deltas, repeated-layer/block dictionaries and parameter tying, factorized embeddings and vocabulary-aware output projection, low-rank matrix factors, structured sparsity, learned codebooks, conditional expert activation and compute-aware routing. Recurrent/linear-attention or hybrid blocks are separate architecture variants.

Measure unique stored parameters, active parameters per token, multiply-adds, peak memory and wall-clock latency separately. Shared parameters reduce storage; sparse activation can reduce compute; neither automatically reduces all three.

### 5.3 Vocabulary compression

Keep tokenizer IDs stable while researching embedding factorization, tied input/output embeddings where supported, adaptive or clustered output projection, low-rank/quantized embedding tables and token-frequency-aware precision allocation. Vocabulary compression changes the weight representation; it does not permit silent token-ID remapping.

### 5.4 Experiment order

1. Establish FP32/BF16/FP16/INT8/INT4 baselines where supported.
2. Implement lossless tensor container and round-trip tests.
3. Measure per-layer sensitivity and weight/activation distributions.
4. Prototype codebook plus residual and low-rank plus residual codecs.
5. Evaluate embedding/output compression independently.
6. Combine only codecs whose individual effects are understood.

## 6. Balanced runtime

Define a stable backend contract around load, tokenize, prefill, decode, generate, train_capabilities, device_info, memory_info and close.

CPU is the correctness reference and fallback. NVIDIA CUDA is enabled only after runtime/device checks. AMD ROCm is enabled only where the installed OS/PyTorch/runtime supports it. Intel/OpenVINO, DirectML and GGUF/llama.cpp are optional adapters, not hard dependencies. Unsupported devices fail with actionable explanations or safely fall back; they never silently claim GPU execution.

Record actual device, driver, runtime, precision, offload layers, batch/context settings, tokens/s, time-to-first-token, peak RAM/VRAM and output agreement against CPU. Training and inference capability are separate.

## 7. Repository implementation backlog

### P0 — contracts and baseline
- [ ] Add versioned model-family manifest and validator.
- [ ] Add experiment manifest with source hashes, environment, data splits, seed and evaluation references.
- [ ] Add a fixed family-disjoint baseline suite for Hindi, English, Hinglish, chat, reasoning, coding, retrieval and tool use.
- [ ] Add checkpoint immutability and candidate-directory safeguards.
- [ ] Add report status fields: measured, pending, failed and not_run.

### P1 — growth and expert composition
- [ ] Extract existing depth growth behind a documented API without changing semantics.
- [ ] Add exact-logit and gradient equivalence tests for no-op depth growth, including expert-enabled models.
- [ ] Implement width growth only after a mapping-level equivalence test exists.
- [ ] Add staged growth trainer with inherited/new parameter groups, replay anchors, KL/feature preservation and resumable optimizer state.
- [ ] Add family sibling/distillation runner.
- [ ] Add multi-expert orchestration modes and specialist output schema.
- [ ] Add router calibration, budget enforcement, verifier hooks and incremental-benefit evaluation.

### P2 — tokenizer and migration
- [ ] Make 64K a reproducible candidate matrix with immutable source and held-out manifests.
- [ ] Add Hindi/English/Hinglish/code/mixed-script tokenization reports.
- [ ] Test old-ID preservation, special IDs, BPE decomposition and round-trip.
- [ ] Add embedding/output migration tests, including tied/untied weights and logits before/after initialization.
- [ ] Keep 64K candidate-only until continued pretraining and chat/task recovery clear gates.

### P3 — DNA compression
- [ ] Add tensor inventory and storage accounting.
- [ ] Add lossless block container, checksums, corruption and random-access tests.
- [ ] Add quantization baselines and quality reports.
- [ ] Prototype codebook, low-rank residual, parameter sharing and vocabulary compression behind feature flags.
- [ ] Add Pareto evaluation: quality vs bytes vs RAM/VRAM vs latency vs active compute.

### P4 — balanced runtime
- [ ] Define backend protocol and capability probe.
- [ ] Wrap native Tantra runtime as reference backend.
- [ ] Add CPU benchmark and deterministic output regression first.
- [ ] Add optional CUDA/ROCm/provider adapters only with hardware-specific CI or recorded manual evidence.
- [ ] Add backend selection and safe-fallback tests.

## 8. Tests and acceptance gates

**Unit/invariant tests**
- source checkpoint/tokenizer hashes unchanged after candidate operations;
- growth preserves logits within declared tolerance before training;
- width mapping preserves outputs before training or is rejected;
- optimizer state maps only where valid and new parameters receive gradients;
- expert outputs are schema-valid; specialists cannot exceed budgets;
- routing supports ambiguous intent and fallback;
- tokenizer IDs/special IDs are stable and new-token decomposition is complete;
- lossless codec round-trips every tensor bit-exactly and detects truncation/corruption;
- lossy codec outputs finite tensors and records actual bytes/metadata;
- backend capability mismatch never reports successful acceleration;
- CPU reference and selected backend pass output-agreement tests.

**Evaluation gates**
- run all existing frozen suites before and after;
- pre-register old-capability regression tolerance;
- use family-disjoint validation/test sets for new capabilities;
- check every generated answer for empty output, incorrect facts, leakage and format failures;
- report Hindi, English and Hinglish separately;
- include repeated seeds/confidence intervals when variance is material;
- never use test-set results to iterate or select checkpoints.

## 9. Milestones

| Milestone | Deliverable | Exit condition |
|---|---|---|
| M0 — baseline | family/experiment manifests and pinned evaluation protocol | existing tests pass; hashes and baseline reports reproducible |
| M1 — growth safety | depth-growth API hardening and invariant tests | exact pre-training function preservation and source immutability |
| M2 — family training | staged growth and sibling distillation | candidate beats control on held-out tasks without unacceptable regressions |
| M3 — expert collective | budgeted router/planner/verifier/synthesizer | specialists add measurable held-out benefit over generalist |
| M4 — 64K tokenizer | comparison matrix and migration/recovery pipeline | language-wise token efficiency and recovered model quality clear gates |
| M5 — DNA storage | lossless container and quantized/codebook candidates | reproducible bytes/load/quality Pareto report |
| M6 — DNA architecture | shared/factorized/low-rank/conditional variants | measured storage or compute win within quality tolerance |
| M7 — balanced runtime | CPU reference and optional accelerator adapters | capability-verified selection and measured cross-device report |

## 10. Definition of done

A milestone is done only when code, tests, a reproducible command, immutable artifacts and a report exist. A design document, passing unit tests, lower loss, fewer bytes in a synthetic tensor or successful import is not evidence of trained capability or production readiness.

All candidate artifacts remain under ignored runs/ paths. Existing model files and tokenizer artifacts are never replaced automatically.
