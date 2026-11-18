# Tantra-LLM: model-family growth, DNA compression and balanced runtime

**Decision record:** model growth = versioned family; first growth operator = residual branches; width/depth growth later; DNA compression = both representation/storage and architecture; tokenizer = 64K research target; runtime = balanced.  
**Status:** research architecture and metadata guardrails only. No growth, compression, tokenizer training, or hardware benchmark is claimed by this document. The residual-branch-first decision is governed by [the native-first roadmap](TANTRA_NATIVE_ARCHITECTURE_ROADMAP.md).

## 1. Preserve the current system

- Keep the native Tantra transformer, tokenizer, expert adapters, router, knowledge engine, training pipeline and CLI working.
- Add model backends alongside native Tantra; do not replace the native backend or rewrite `Model/`.
- Treat each model, tokenizer, adapter set and router as a versioned immutable artifact with SHA-256 identity.
- All experiments write to a new `runs/<experiment-id>/` directory. Never write candidates into the source checkpoint or canonical `Model/`.
- A candidate is not promoted because its loss fell. It must pass held-out response checks and full regression gates.

## 2. Model family, not a single ever-growing checkpoint

The family is a set of related, independently deployable models. A child may be created by controlled morphism from a parent, or a sibling may be distilled from a stronger teacher. Every member has its own checkpoint, manifest, tokenizer identity, backend, benchmark record and lifecycle state.

Proposed family ladder (targets, not existing checkpoints):

| Family role | Target scale | Purpose | Promotion condition |
|---|---:|---|---|
| Edge | 0.8B–1B | CPU/low-memory daily tasks | Quality floor + device benchmark |
| Local default | 2B–4B | balanced chat, reasoning, tools | Beats edge on held-out suite at acceptable latency |
| Specialist | 1B–4B each | code, math, Hindi, retrieval, structured tasks | Domain gain without harmful router regressions |
| Strong local/server | 7B–14B | broader general capability | Measured quality/latency improvement |
| Research | 30B–70B | multi-GPU and distributed experiments | Budget, data and hardware gate |
| Long-horizon sparse research | up to 1T total MoE parameters | large expert pool, distributed serving | Never treated as a local deployment target |

A 1T MoE has a large total parameter pool but activates only a subset per token; all experts still have to be stored and made available to the serving system. Keep total, active-per-token and resident parameters as separate metrics.

### Controlled expansion protocol

1. Freeze and hash the parent checkpoint, tokenizer, config, training data manifest and evaluation suite.
2. For the first native expandability milestone, select residual-branch growth only: attach a parallel compatible branch behind a zero/small gate and keep the original expert immutable.
3. Use distillation + replay + new training to teach the added branch. Width and depth growth are later operators and require their own architecture-specific equivalence/control tests. Random-only expansion is not an accepted growth method.
4. Before training, run equivalence tests on fixed inputs: logits, hidden-state shapes, finite values and output drift. For a truly function-preserving transform, measure numerical tolerance; for non-equivalent transforms, record the expected change and compare to a no-op control.
5. Continue training on provenance-locked data with frozen train/validation/test family splits. Save optimizer, scheduler, RNG and data cursors for resumability.
6. Evaluate parent, child and no-op control on identical held-out tasks, languages, prompts and decoding settings. Record regressions, not just average gains.
7. Promote only if the candidate clears absolute quality floors and improves the intended objective without unacceptable regressions. Otherwise archive it; parent remains champion.

**Required implementation work:** morphism operators must be architecture-specific. Do not assume a generic tensor-copy routine preserves a Transformer function. Implement and prove one operator at a time.

## 3. Small experts working together

Keep specialists small and independently loadable. Use a two-level router:
- Fast first-pass router selects candidate experts from intent, language and task metadata.
- A verifier/reranker may add a second expert or reject low-confidence routing.
- A shared orchestrator composes tool calls and specialist outputs, then checks conflicts, citations, schemas and safety constraints.

Initial expert groups: language/Hindi, coding, math, factual retrieval, summarization, structured data and tool-use. These are candidate domains, not claims that suitable trained experts already exist.

Tests must cover top-k routing, confidence calibration, no-expert fallback, expert timeout/failure, conflicting answers, duplicate tool calls, context budget, per-expert unload/reload, and end-to-end task success. Measure total system latency and memory, not just each expert independently. A team of small models can outperform one small general model on decomposable tasks, but orchestration overhead and error propagation must be measured.

## 4. DNA compression — both tracks

“DNA” is the research codename for AI-guided compression, not a claim that encoding tensors as A/C/G/T makes them smaller or faster. Maintain two distinct tracks with a shared benchmark harness.

### A. Parameter/vocabulary/storage compression

Candidate techniques, tested separately and in combinations:
- Quantization: BF16/FP16, INT8, INT4, then lower-bit only if quality and kernel support justify it.
- Block/dictionary coding: cluster repeated tensor blocks or codebook entries; store codebook plus indices and per-block scales.
- Low-rank factorization for eligible matrices.
- Sparse/delta encoding for checkpoints and adapters.
- Shared/tied embeddings where architecture supports it; factorized embeddings and output projection are experiments, not assumptions.
- Tokenizer serialization compression and vocabulary metadata deduplication.
- KV-cache quantization/compression as a serving-time memory track.

Every report must include original bytes, compressed bytes including metadata, compression ratio, encode/decode time, peak RAM, inference latency, task quality and exact hardware. Lossless packaging and lossy weight quantization must be reported separately. A smaller file that expands to a large RAM footprint is not a successful runtime compression.

### B. Architecture/computation compression

Investigate parameter sharing, repeated-block/Universal Transformer variants, low-rank projections, structured pruning, early exit, conditional computation, sparse MoE, hybrid attention/linear-recurrent blocks, and dynamic expert activation. Distinguish parameter count, active FLOPs, memory traffic, resident memory and wall-clock latency.

Use teacher/student distillation as a quality-transfer baseline. Do not call a compressed architecture successful until it beats an uncompressed control on a fixed quality/latency/memory frontier.

### AI-guided search loop

1. Build a reproducible candidate space with explicit constraints.
2. Train/evaluate a small proxy before spending on a full candidate.
3. Use Bayesian/ evolutionary search or bandit allocation to select the next candidate from measured results.
4. Keep a Pareto archive (quality, bytes, RAM/VRAM, latency, energy if available).
5. Re-run top candidates from clean seeds and independent evaluation.
6. Save the recipe, source hashes, code version, environment and measurements.

The search system may recommend transformations; it may not silently mutate the champion or promote a candidate.

## 5. Tokenizer decision: 64K target, migration is separate

The current repository has a 10,277-entry ByteLevel BPE tokenizer and an existing 64K candidate/extension experiment path. Do not replace the current tokenizer or treat “64K” as a proven optimum.

- Freeze the existing tokenizer and record its hash.
- Evaluate the existing baseline against 16K, 32K and 64K candidates using identical provenance-backed training and held-out corpora.
- Evaluate Hindi, English and Hinglish separately; Hinglish needs a meaningful reviewed corpus before it can pass a language-quality gate.
- Measure token fertility, bytes/token, sequence length, round-trip fidelity, encode/decode throughput, vocabulary frequency tail, embedding/output parameter cost and downstream model quality.
- Preserve old token IDs when extending BPE where feasible. Stable old IDs do not make a new vocabulary model-compatible: embedding/output rows and optimizer state still require migration and training.
- Keep tokenizer training, embedding initialization, vocabulary adaptation and continued pretraining as separate versioned steps.
- Promote the 64K tokenizer only after measured gains justify its embedding, output-projection, RAM and latency costs.

## 6. Balanced runtime

Define one backend contract and capability probe; keep optional dependencies optional:
- CPU: PyTorch CPU path, thread controls, memory-aware batch/sequence limits.
- NVIDIA: CUDA when installed and actually available; record GPU model, VRAM, driver and runtime versions.
- AMD: supported ROCm configurations only; never assume every AMD GPU is supported.
- Intel: supported oneAPI/XPU or CPU paths only; detect capability rather than guessing.
- Quantized CPU/GPU kernels are backend-specific. Do not silently fall back to a slower or numerically different kernel without reporting it.

Backend selection order is user override → compatible accelerator with sufficient memory → CPU. Record selected device, fallback reason, precision, quantization, memory use, tokens/sec and time-to-first-token. The same functional tests must run on CPU; accelerator tests are marked and skipped with an explicit reason when hardware is absent.

## 7. Repository-level work packages

| ID | Work package | Deliverables | Acceptance tests |
|---|---|---|---|
| FAM-01 | Family manifests | immutable member records, parent lineage, hashes, states | schema, duplicate IDs, missing parent, cycle, hash mismatch |
| FAM-02 | Backend interface | native + Transformers + GGUF adapters behind one contract | same prompt contract, capability detection, fallback |
| GROW-01 | Morphism research harness | depth/width operators, copy/gate initialization, equivalence report | output equivalence before training; source hash unchanged |
| GROW-02 | Family training | parent-to-child continuation and sibling distillation | resume, determinism, optimizer restore, data split isolation |
| EXP-01 | Expert team orchestration | hierarchical router, verifier, composition protocol | routing, failure, conflict, latency and memory tests |
| DNA-01 | Compression benchmark schema | byte/quality/latency/RAM measurements | exact byte accounting; metadata included |
| DNA-02 | Block dictionary prototype | tensor block codebook + index codec, decode path | round-trip, corruption detection, deterministic encoding |
| DNA-03 | Architecture search harness | candidate configs, proxy scoring, Pareto archive | reproducible candidate, no champion mutation |
| TOK-01 | 64K benchmark | 16K/32K/64K comparison report | language-separated fertility, round-trip, speed, hash pins |
| TOK-02 | Safe migration | isolated embedding/output expansion + training recipe | old rows exact, finite new rows, source unchanged |
| RT-01 | Balanced device probe | CPU/CUDA/ROCm/XPU capability record | mocked capability matrix + actual-device smoke |
| RT-02 | Runtime benchmark | latency, throughput, peak memory, fallback | same workload and output checks across devices |
| GOV-01 | Promotion gate | machine-readable evidence bundle | refuses promotion without all required evidence |

## 8. Milestones and stop/go gates

**M0 — protect baseline:** inventory current CLI, tests, artifacts and hashes; all existing tests green. No model changes.

**M1 — family metadata + backend contract:** manifests, lineage validation, model registry and backend interface. No weight mutation.

**M2 — tokenizer 64K evidence:** run the already-defined tokenizer candidate workflow; improve Hinglish evaluation data first. 64K remains candidate until it wins.

**M3 — first controlled growth operator:** implement one architecture-specific function-preserving transform on a tiny test model. Prove equivalence and source immutability before any training run.

**M4 — expert team:** route and compose two specialists with verifier/fallback; test task success against single-model and single-expert controls.

**M5 — DNA storage prototype:** implement lossless block dictionary encoding first, then lossy quantization comparisons. Do not combine methods until single-method baselines exist.

**M6 — architecture compression/search:** compare one change at a time; keep Pareto archive and independent repeats.

**M7 — balanced runtime:** CPU baseline first, then supported accelerator paths. Hardware claims require actual device smoke logs.

**M8 — family scale-out:** expand the family only when each rung has a promoted checkpoint and measured evidence. 1T remains a distributed research horizon, not a near-term deliverable.

## 9. Test and evidence policy

- Unit: manifest validation, lineage acyclicity, tensor shape checks, codec round-trip, tokenizer ID invariants, device capability matrix.
- Integration: parent → candidate → train → evaluate → promote; candidate failure never changes parent.
- Regression: existing single-turn/chat suites plus math, code, tool calls, Hindi, English and Hinglish.
- Performance: identical prompts, batch, sequence length, precision and hardware; report median and p95 latency, tokens/sec, peak RAM/VRAM and energy only if measured.
- Data integrity: provenance, checksums, family-grouped splits, duplicate and contamination gates.
- Evidence artifact: JSON manifest + logs + metrics + environment + source hashes + model/tokenizer hashes.

A test passing proves that test passed. It does not prove general intelligence, frontier capability, a compression win or hardware compatibility beyond the tested environment.
