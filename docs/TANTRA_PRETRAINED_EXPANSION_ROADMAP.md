# Tantra native-first expandable architecture and roadmap

**Architecture decision: LOCKED — 18 November 2026**  
**Product identity:** independent native Tantra architecture and weights.  
**First milestone:** native foundation + two compatible experts + residual-expansion proof.  
**Implementation status:** the repository contains a small native research model and related training/evaluation code. The complete two-expert residual-expansion lifecycle described here is a target, not a completed capability.

This document is the authoritative architecture and implementation roadmap. It supersedes the earlier pretrained-first product direction. Historical pretrained integration work remains available as optional tooling, teacher/benchmark infrastructure and independent model-gateway experiments.

## 1. Locked decisions

| Area | Decision | Engineering rule |
|---|---|---|
| Model identity | Independent native Tantra architecture and weights | Tantra owns its architecture, tokenizer, vocabulary, training pipeline, expert format, router and native checkpoints. |
| Expert growth | Residual branches first | Add parallel compatible capacity while preserving the original expert path; width and depth expansion follow later. |
| Learning strategy | Distillation + replay + new training | New capacity learns new capabilities while teacher targets and replay examples help preserve validated behaviour. |
| Version retention | Keep all validated versions; allow rollback | Promoted artifacts are immutable, addressable and recoverable. Never overwrite a validated parent. |
| First milestone | Native foundation + two experts + residual-expansion proof | Prove the architecture on a small reproducible model before pursuing 0.8B, 3B or larger targets. |

These are architecture decisions, not claims that the mechanisms are already implemented or that larger models will automatically be better.

## 2. Product identity and external-model boundary

Tantra is not a renamed third-party model. Its long-term foundation is trained native Tantra weights under a documented native architecture and tokenizer.

External models may be used in explicitly separated roles:
- Optional teachers that generate candidate demonstrations or soft targets, subject to review and provenance.
- Independent benchmark references for quality and capability comparisons.
- Optional model-gateway services for experiments or tools.
- Research references for architecture and training techniques.

External weights must remain separate from native Tantra checkpoints. Record provider, exact model ID, immutable revision, tokenizer, licence, source hashes and intended use. Never claim third-party weights were created by Tantra. A teacher's output is not automatically trusted training truth; review and validate it before use.

The pretrained gateway, LoRA and third-party residual-branch experiments already in the repository are optional engineering tracks. They do not define Tantra's foundation and must not displace the native-first milestone.

## 3. Target architecture

1. **Native foundation:** decoder-only Transformer, Tantra-owned tokenizer/vocabulary, configuration, training and inference contracts.
2. **Shared representation contract:** explicit hidden-state dimensions, tensor shapes, dtypes, device behaviour, context semantics and failure handling.
3. **Native expert family:** independently versioned expert modules that consume and return compatible representations.
4. **Expandable residual experts:** preserve the validated expert path and attach new trainable branches behind zero/small gates.
5. **Expert registry:** immutable versions, parent lineage, tokenizer/config hashes, capability labels, resource requirements and evaluation evidence.
6. **Router:** selects shared-only, one expert or multiple experts using task, confidence and compute budgets; later versions may learn routing from measured outcomes.
7. **Learning system:** new capability data, teacher distillation, replay, controlled joint tuning and reproducible checkpointing.
8. **Evaluation and promotion:** frozen held-out tests, regression gates, shadow routing, staged promotion and rollback.
9. **Knowledge/action layer (later):** retrieval, structured memory, tools and APIs complement parametric model knowledge.

Keep external model gateway integration at the boundary. Do not force incompatible tokenizers or hidden-state dimensions into native expert slots.

## 4. Residual-branch growth: first expansion operator

The first growth method is a parallel residual branch. Conceptually:

`expanded(x) = original_expert(x) + gate(x) * new_branch(x)`

- `original_expert` is the validated existing expert and remains immutable during branch training.
- `new_branch` is additional trainable capacity with compatible input/output representation.
- `gate` starts at zero or a carefully justified small value so the initial expanded path matches the original path as closely as the implementation promises.
- The original expert remains available as a standalone version and fallback.

### Required sequence

1. Freeze and hash the original expert, shared foundation, tokenizer, config, data manifests and evaluation suite.
2. Create a new candidate artifact; never write into the source checkpoint.
3. Add a residual branch with matching input/output dimensions and explicit dtype/device behaviour.
4. Initialise the branch and gate; run fixed-input equivalence tests before training.
5. Train the new branch on approved new-capability data.
6. Use teacher distillation and replay examples from old capabilities to reduce forgetting.
7. Jointly tune the branch, gate and (only if justified) router in a separately recorded stage.
8. Evaluate original, expanded and control candidates on identical held-out prompts.
9. Promote only if the new capability improves and old capabilities remain within pre-registered regression limits.
10. Retain the parent and every validated child version; rollback must restore the prior active version.

This reduces discarded work; it does not eliminate training. New parameters and their interaction with the original expert must learn useful behaviour.

### Growth methods after the first proof

- **Depth expansion:** add Transformer blocks with architecture-specific initialisation and function-preservation tests.
- **Width expansion:** expand selected projections/FFN dimensions with explicit tensor mapping and numerical validation.

The repository has historical native depth-growth work. That does not replace the residual-branch-first decision. Width growth remains later work. Never assume generic tensor copying preserves a Transformer function.

## 5. Expert versioning and compatibility

Each expert version must have a manifest containing:
- Stable expert family ID and immutable version ID.
- Native architecture/config version and shared-interface version.
- Parent checkpoint hashes and lineage.
- Tokenizer and vocabulary hashes.
- Parameter count, dtype, supported devices and resource requirements.
- Capability labels and intended task coverage.
- Training code/source commit, seed, optimizer and scheduler.
- Dataset source manifests, licence/provenance and train/validation/test split hashes.
- Distillation teacher identity/revision (if any), replay manifest and ratios.
- Evaluation suite version, metrics, human-review record and promotion decision.

The registry must allow multiple validated versions of the same family. The router may choose among versions only after compatibility and evaluation checks. A failed candidate is archived with its report; it does not replace the active version.

## 6. Learning without forgetting

Use three complementary inputs:
- **New training data:** teaches the added branch or expert the intended new capability.
- **Distillation:** transfers selected behaviour from the frozen parent/teacher to the expanded candidate.
- **Replay:** revisits approved examples representing previously validated capabilities.

Keep training, validation and test families disjoint. Teacher-generated data must carry teacher provenance and human-review status. Do not use held-out evaluation examples as replay or training data.

Measure old and new capabilities separately. Training loss is diagnostic only; it is not a promotion score.

## 7. First milestone — Tantra Expandability Prototype v0.1

### Prototype scope

- One small native Tantra decoder-only foundation and its tokenizer.
- Two native experts (initially general reasoning and coding) that share the same explicit representation contract.
- A simple router able to select shared-only, Expert A, Expert B or both. Initially deterministic policies are acceptable; learned routing is not a prerequisite.
- A residual branch attached to Expert A.
- Frozen parent, distillation + replay training, immutable candidate checkpoints and rollback.
- A machine-readable experiment report.

### Milestones and exit gates

| Step | Deliverable | Pass condition |
|---|---|---|
| 1 | Native foundation + tokenizer | Reproducible training/inference; checkpoint reload matches recorded outputs |
| 2 | Shared expert interface | Both experts pass shape, dtype, device and failure-contract tests |
| 3 | Two native experts | Both run against the same foundation representation |
| 4 | Basic router | Selects shared-only, A, B or both; fallback behaviour is tested |
| 5 | Residual branch | New branch attaches without deleting or mutating original weights |
| 6 | Distillation + replay | Expanded candidate learns a new target while retaining old capability within gate |
| 7 | Versioning + rollback | Parent and validated child coexist; active pointer can be safely rolled back |
| 8 | Evaluation report | Reproducible comparison includes quality, regression, latency, memory and training cost |

### Required measurements

- Old-task accuracy/quality and new-task accuracy/quality.
- General language, reasoning and coding performance.
- Router selection quality and fallback rate.
- Human-reviewed correctness, instruction adherence, naturalness, factuality and safety.
- Median/p95 latency, throughput, peak RAM/VRAM and active/resident/total parameters.
- Training time, compute cost, data volume and checkpoint size.
- Exact parent/child hashes and pre/post-training equivalence results.

A prototype passes only when the intended new capability improves without unacceptable regression. A parameter-count increase alone is not success.

## 8. Data and evaluation requirements

Keep the repository's review-first data pipeline. Preserve source IDs, URLs, revisions, licences, hashes and review decisions. Hindi and Hinglish require fluent-language review. Split by conversation family, keep train/validation/test disjoint and immutable, and report each language separately. Never duplicate examples to fill a language shortage.

Current data-readiness gaps remain real blockers for multilingual capability claims. Continue improving the reviewed corpus in parallel with architecture work, but do not let the absence of a large corpus prevent a small synthetic/mechanics-only expansion prototype. Clearly label mechanics fixtures as synthetic diagnostics, not language-quality evidence.

Use the same frozen suite, prompt format, decoding settings and hardware for parent/child comparisons. Human review is required for dimensions that automated checks cannot establish.

## 9. Longer-term capacity direction

The 0.8B → 1.5B → 2–3B path is an illustrative family target, not a committed training schedule or guaranteed quality curve. Keep smaller validated experts as teachers, fallbacks and lower-resource deployment options. Larger versions must demonstrate a measurable benefit.

A long-horizon sparse expert system may eventually contain hundreds of billions or a trillion total parameters, but that is research ambition—not a current engineering milestone. Sparse activation reduces per-token computation; it does not eliminate the need to store and serve the expert pool.

## 10. Implementation status — 18 November 2026

### Exists in the repository
- A small native Transformer, tokenizer artifacts, training/inference code, routing and expert/adaptation experiments.
- Native CPU training and checkpoint validation experiments.
- Historical narrow capability pilots and regression reports; they do not establish broad language capability.
- Native depth-growth work with a recorded function-preservation test in a prior experiment.
- Optional pretrained gateway, registry, LoRA/replay and residual-branch engineering experiments.
- Dataset intake, provenance, review and multilingual evaluation documentation.
- Automated tests and CI for the implemented software contracts.

### Not yet demonstrated as the locked native milestone
- Two native neural experts proven to share the final stable hidden-state contract.
- A native residual branch integrated into that shared expert contract and trained with distillation + replay.
- A complete native expert registry retaining multiple validated versions with tested rollback.
- A router proven to select the native experts usefully under measured quality/compute trade-offs.
- A measured native 0.8B-to-2–3B expansion.

### Known blockers and constraints
- The native pilot is small and narrow; it is not a general-purpose or frontier model.
- Reviewed Hindi/Hinglish data remains insufficient for the previously requested bilingual split; the 10M bilingual workflow failed its data-readiness gate before training.
- Larger-scale training requires substantially more reviewed data, compute, and measured infrastructure.
- Optional external-model smoke tests and passing software tests do not prove native Tantra capability.

## 11. Immediate next engineering work

1. Audit the current native model, expert, router, tokenizer and checkpoint code; identify reusable components and gaps.
2. Write the native shared-expert interface contract and version it.
3. Implement or adapt two tiny native experts against that contract.
4. Implement the residual branch for the native expert representation, with zero-gate equivalence tests and immutable parent checkpoints.
5. Add distillation + replay training configuration and a family-disjoint evaluation fixture.
6. Add version manifests, registry entries and rollback tests.
7. Run the complete prototype and publish the measured report before choosing a larger model size.

Do not begin a large 0.8B/3B run, expand the expert taxonomy, or prioritise optional third-party fine-tuning until the small native expandability proof passes.
