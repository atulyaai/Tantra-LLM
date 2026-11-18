# Tantra-LLM evolution architecture

**Decision record:** model growth = compatible family; compression = both storage/parameter coding and architecture; tokenizer target = 65,536; runtime = balanced CPU/NVIDIA/AMD where the installed stack supports it.

**Status:** architecture and implementation backlog. No new model training, model expansion, tokenizer migration, compression experiment, or hardware benchmark is claimed by this document.

## 1. Preserve and extend the current system

The native Tantra model, tokenizer, expert definitions, router, data contracts, evaluation suites and training commands remain supported. Do not replace `Model/tokenizer.json`, rewrite native checkpoints, or change existing token IDs as part of the initial integration.

Add a backend boundary rather than forcing every model through native tensor assumptions:

- **Native backend:** existing Tantra transformer and adapter path.
- **Transformers backend:** external foundation checkpoints (initially a Qwen candidate after revision, license, tokenizer and architecture audit).
- **Quantized backend:** GGUF/other runtimes only through explicit capability detection and model-format manifests.
- **Runtime selector:** chooses a supported backend/device plan from measured capabilities; never assumes that CUDA, ROCm, or a particular GPU is installed.

A backend manifest must record model ID and immutable revision, license identifier and source, architecture/config hash, tokenizer hash, context limit, dtype/quantization, required libraries, supported devices, and provenance. An unverified license or missing immutable revision blocks promotion.

## 2. Model family: grow by controlled lineage

A family is a set of separately versioned models with explicit parentage and compatibility. Initial target rungs are research targets, not currently available Tantra checkpoints:

| Family rung | Intended role | Growth / training path |
|---|---|---|
| 0.8B–1B | edge / low-memory | distilled or continued-pretrained foundation |
| 2B–4B | balanced local default | parent checkpoint + domain adaptation |
| 7B–14B | higher-quality local/server | controlled expansion or independently trained sibling |
| 30B–70B | multi-GPU / server | family transfer, distillation and/or sparse experts |
| 100B–1T total | distributed research | sparse MoE; report active parameters separately |

Do not promise that every rung can be obtained by mechanically widening one checkpoint. For each parent→child transition, benchmark at least these candidates:

1. **Distilled sibling:** train a smaller or differently sized student from a frozen parent/teacher.
2. **Function-preserving expansion:** copy compatible tensors exactly; initialize new channels/blocks with a documented identity-preserving or near-identity method; gate new capacity initially; then train with replay and distillation.
3. **Depth/width morphism:** insert/duplicate layers or widen dimensions only where tensor shapes and positional/normalization semantics are compatible. Use explicit mapping manifests.
4. **Sparse expert growth:** add experts incrementally, initialize from a shared base or expert copies, train router and experts with load-balancing and replay.
5. **Fresh sibling:** train from a compatible foundation checkpoint where morphism is not safe or does not beat the control.

Every expansion run has a frozen parent, exact tensor mapping, initialization seed, optimizer/scheduler, data manifest, tokenizer hash, replay mix, teacher logits/config, and rollback pointer. First prove function preservation on a fixed probe suite before training. Then compare child against (a) parent, (b) unexpanded control trained for the same token/step budget, and (c) a distilled sibling. Promote only on independent held-out task gains without material regressions in language, safety, calibration, or latency.

**No random-parameter-only growth.** Newly allocated parameters are not intelligence. They are capacity that must be initialized, trained, and validated.

## 3. Specialist models and collective intelligence

Reuse the existing expert taxonomy and router; do not create a duplicate expert registry. Treat experts as independently versioned workers with a narrow contract:

- input schema, domain, supported operations, refusal boundaries;
- model/backend/revision and tokenizer hash;
- adapter or standalone-weight hash;
- train/validation/test provenance and contamination checks;
- calibrated confidence, latency, memory and tool requirements;
- output schema and evidence/citation requirements where relevant.

Build three execution strategies behind one interface:

- **Single expert:** route to one best specialist.
- **Parallel experts:** ask a small set of specialists independently, then synthesize.
- **Sequential workflow:** planner → specialist(s) → verifier, with bounded retries and explicit tool permissions.

Compare these with the single general model on the same frozen tasks. Router training data must be separate from expert evaluation. Add abstention/fallback when confidence is low. Keep a small general model available as fallback; do not call every expert for every prompt.

## 4. 64K tokenizer target and migration safety

The target vocabulary is **65,536 tokens**. The repository already has a separate 64K extension experiment recorded in `docs/TRAINING_NOTES.md` and `Tantra/training_scope.json`. It is not a production tokenizer or proof of model quality.

Rules:

- Keep `Model/tokenizer.json` and the backup unchanged until a separately versioned candidate passes the full benchmark.
- Preserve all existing token IDs for the extension candidate; record exact tokenizer SHA-256, normalizer, pre-tokenizer, decoder, special-token IDs and merge order.
- Evaluate Hindi, English and Hinglish separately. Hinglish evidence remains provisional until there is sufficient reviewed, family-deduplicated data.
- Measure token fertility (tokens/characters and bytes/token), round-trip exactness, unknown-byte behavior, encode/decode throughput, vocabulary coverage, long-tail terms, code, mixed-script text and downstream loss/quality.
- A 64K tokenizer is not a drop-in change. New IDs require resized input embeddings and output head, deterministic initialization for added rows, training, and exact model↔tokenizer pairing. Old adapters/checkpoints stay paired with their original tokenizer.
- Compare 32K and 64K where practical; the selected target remains 64K, but promotion is evidence-gated. If 64K materially hurts any language or task, fix the data/merge strategy before migration rather than silently accepting the regression.

## 5. DNA compression research — both tracks

“DNA” is the project codename for **data/parameter neural compression**, not a claim that converting weights to A/C/G/T or another alphabet creates compression. Keep two research tracks separate and comparable.

### A. Representation / storage compression

- Quantization: BF16/FP16 baseline; INT8 and INT4; lower-bit formats only as explicit experiments.
- Entropy coding and blockwise packing of quantized tensors; codebooks, shared dictionaries and repeated-block detection.
- Low-rank factorization and tensor decomposition where reconstruction error and downstream quality are measured.
- Sparse/structured pruning with a hardware-aware sparse layout; unstructured zeros do not count as speedups unless the runtime exploits them.
- Embedding/output-matrix factorization, tied weights where architecture permits, and tokenizer metadata/merge-delta packing.
- KV-cache quantization/compression as a serving optimization, measured separately from on-disk model size.

### B. Architecture / computation compression

- Parameter sharing, layer tying, recurrent/linear-attention alternatives, compact student distillation, early exit and adaptive compute.
- MoE and expert activation: report total parameters, active parameters/token, expert count, routing load, communication and resident memory separately.
- Dynamic rank/adapters and conditional execution only where task quality and end-to-end latency improve.
- Vocab-aware/factorized embeddings and output projections, with exact tokenizer compatibility.

Every compression artifact must have a decoder/runtime version, source model hash, tokenizer hash, compression recipe, reconstructed-weight checksum (where lossless), size on disk, peak RAM/VRAM, load time, tokens/s, quality deltas and supported hardware. Lossless coding must reconstruct bit-identical source tensors. Lossy methods must pass pre-registered quality thresholds. Never label an encoding smaller until the actual serialized artifact is measured.

## 6. Balanced runtime

Use capability probing, not vendor assumptions:

1. CPU baseline always available.
2. NVIDIA CUDA path only when the installed PyTorch/runtime reports compatible device and kernels.
3. AMD ROCm path only when the installed stack reports support.
4. Hybrid placement only when partitioning is supported and measured to beat the selected single-device path; CPU↔GPU transfers can make hybrid slower.
5. Unsupported devices fall back cleanly to CPU with an explicit reason.

Record CPU model/cores/RAM, GPU model/VRAM/driver/runtime, OS, backend versions, precision, quantization, batch, context length, peak memory, load time, time-to-first-token, prefill/decode throughput, and power if available. Never infer compatibility from “NVIDIA GPU” alone.

## 7. Repository-level implementation map

| Area | Add / change | Acceptance tests |
|---|---|---|
| Model contracts | `Tantra/backends/`, `Tantra/model_manifest.py`, `configs/models/` | manifest schema, immutable revision, tokenizer/hash pairing, backend contract |
| Family growth | `Tantra/model_family.py`, `configs/model_family.yaml`, `Tantra/growth/` | DAG/no cycles, tensor-map coverage, exact-copy identity, deterministic init, parent unchanged, rollback |
| Expert federation | extend existing router/expert contracts; no duplicate taxonomy | single/parallel/sequential dispatch, timeout, abstain/fallback, verifier, budget limits |
| Tokenizer | keep existing candidate builder; add benchmark report schema and migration tool | ID preservation, special IDs, round-trip, multilingual held-out split, embedding resize/pairing, source immutability |
| Compression | `Tantra/compression/` | lossless round-trip, lossy quality gates, corruption detection, actual byte counts, runtime load |
| Runtime | `Tantra/runtime/` and hardware probe | CPU fallback, unsupported GPU fallback, CUDA/ROCm capability mocks, placement correctness |
| Evaluation | `benchmarks/` and run manifests | frozen data, leak checks, per-language and per-domain reports, reproducibility metadata |
| CI | CPU-only contract suite; optional GPU jobs | no model downloads in normal CI; deterministic small synthetic tensor tests |

## 8. Milestones and promotion gates

### M0 — repository safety and contracts
- Add this architecture decision, family manifest, backend/model manifest schema and test fixtures.
- Snapshot current tokenizer/checkpoint hashes and keep all generated artifacts under ignored `runs/`.
- Run existing tests before integration; record exact command and result.

### M1 — model registry and inference adapter
- Add native backend adapter first, then one external Transformers backend behind optional dependencies.
- Load a tiny local test model in CI; no network download.
- Audit one Qwen candidate's exact revision, license, tokenizer and architecture before writing a production config.

### M2 — family planner and expansion proof
- Implement tensor mapping and deterministic initialization utilities.
- Test identity preservation before any optimizer step.
- Run a small synthetic expansion experiment, then a small real checkpoint experiment only after the proof test passes.
- Compare morphism, distillation and unexpanded control.

### M3 — expert federation
- Add bounded parallel/sequential orchestration using existing experts.
- Add traces, confidence/abstention, verifier and cost/latency budgets.
- Evaluate against single-model and existing-router baselines.

### M4 — tokenizer 64K decision
- Reproduce and extend the recorded 64K benchmark with sufficient held-out Hindi/English/Hinglish and code/mixed-script data.
- Add deterministic embedding/output-head migration and tokenizer/model compatibility checks.
- Do not promote until language and task gates pass.

### M5 — DNA compression experiments
- Establish BF16/FP16 and existing quantized baseline.
- Test INT8/INT4, block dictionaries/entropy coding, low-rank, structured pruning, factorized embeddings and KV cache independently.
- Run architecture compression (distillation, parameter sharing, early exit, MoE) as separate experiments.
- Publish size/quality/speed/memory Pareto table; no single “compression ratio” without the quality and runtime context.

### M6 — balanced runtime and family release
- CPU first, then available NVIDIA/AMD backends, then hybrid placement.
- Test each family member independently and as a routed system.
- Promote only signed manifests with passing held-out gates; preserve rollback to previous family member.

## 9. Experiment record and release gates

Each run directory must contain immutable `manifest.json`, command/config, source hashes, data hashes/splits, environment, logs, metrics, checkpoint hashes and decision. Never overwrite a run directory. No test-set tuning. Keep Hindi, English and Hinglish separate. Report both aggregate and per-domain regressions. Require leak checks before any capability claim.

Minimum release comparison:

- general chat and instruction following;
- Hindi, English and Hinglish;
- reasoning/math and code;
- tool use and structured output;
- safety/refusal and calibration;
- context recall and long-context behavior;
- load time, TTFT, decode throughput, peak RAM/VRAM and disk size.

**Promotion policy:** a candidate must beat or justify its trade-off against its parent and same-budget control on frozen held-out data, with no unacceptable safety/language regression. Validation loss alone is insufficient. A smaller compressed artifact is not a win if it is slower or loses required tasks.

## 10. Immediate implementation order

1. Add metadata-only family registry and contract tests.
2. Add backend interface with native adapter; preserve CLI behavior.
3. Add external backend as optional dependency and test with tiny local fixture.
4. Add expert federation on top of the existing router.
5. Re-run 64K tokenizer benchmark; do not alter production tokenizer.
6. Add compression codecs one by one behind an experimental flag.
7. Add hardware-aware runtime selection and benchmark matrix.

This ordering produces useful, testable infrastructure before expensive training or hardware-specific optimization.
