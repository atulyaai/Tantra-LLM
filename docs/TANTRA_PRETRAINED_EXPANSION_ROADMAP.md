# Tantra pretrained-first architecture and expansion roadmap

**Decision status:** agreed architecture direction, 18 November 2026.  
**Implementation status:** specification and validation contracts are prototypes; no pretrained model has been downloaded, fine-tuned, merged, or promoted by this document.

## 1. Product direction

Tantra is the platform, identity, orchestration architecture, model registry, expert system, adaptation pipeline and evaluation/governance layer. It may use third-party pretrained checkpoints as starting weights. Tantra must not imply that it created those original weights.

The platform supports:
- Pretrained foundation models adapted with LoRA/adapters or full fine-tuning when justified.
- Independent specialist models connected through a model gateway.
- Compatible neural experts integrated into Tantra's sparse neural layers.
- Expert inheritance from one or more parent models.
- Three capacity-growth operators: width, depth and residual-branch growth.
- Dynamic compute routing and hybrid knowledge (weights + retrieval/memory + tools).
- Versioned checkpoints and reversible promotion.

## 2. Architecture

1. **Identity and API boundary:** stable Tantra identity, API schemas, policy enforcement and truthful provenance.
2. **Orchestration core:** task decomposition, context, budget controller, hierarchical routing and result verification.
3. **Model registry:** immutable model revisions, tokenizer hashes, architecture metadata, licence/provenance, hardware requirements and benchmark evidence.
4. **Integration adapters:** model-gateway path for arbitrary supported models; neural-adapter path only when hidden-state/tensor contracts are compatible.
5. **Expert system:** pretrained-derived specialists, Tantra-native expert blocks, inherited children and optional independent specialist services.
6. **Knowledge/action layer:** parametric weights, retrieval/structured memory, tools and APIs.
7. **Learning pipeline:** curated data, LoRA/adapters, continued pretraining, supervised fine-tuning, distillation, replay and optional preference optimisation.
8. **Promotion controller:** shadow evaluation, regression gates, staged rollout and rollback.

Do not force every model into one architecture. Model-level integration is the compatibility default. Neural-level integration requires explicit compatibility and integration training.

## 3. Model intake and licensing

For every source checkpoint record:
- Provider, exact model name, immutable revision/commit and download source.
- Architecture/config, parameter count, tokenizer identity and tokenizer hash.
- Weight and dataset provenance where available.
- Exact licence identifier and a legal review status.
- Intended use, redistribution/hosting constraints and attribution requirements.
- Security scan, file hashes, known limitations and baseline evaluation.

Unknown or restricted licence status blocks integration until reviewed. Publicly downloadable does not automatically mean commercially usable or redistributable. Do not bundle third-party weights into the repository unless their licence permits it.

## 4. Integration paths

### A. Model gateway (default)

Run the pretrained model independently. Tantra passes a typed request and receives a typed response. This supports different tokenizers, hidden sizes and architectures. The cost is separate inference and orchestration latency.

### B. Neural adapter

Only use when model internals are accessible and compatible. Validate hidden dimensions, attention/FFN conventions, normalization, positional encoding, tokenizer/embedding semantics, dtype, layer placement and output contract. Add learned projections/adapters and train the integration. Never plug a full independent transformer into an FFN slot by assumption.

## 5. Adaptation and inheritance

Adaptation sequence:
1. Baseline benchmark and source checkpoint hash.
2. Data curation, provenance and train/validation/test split.
3. Start with LoRA or adapters; use continued pretraining/SFT as task requires.
4. Keep general replay examples to reduce forgetting.
5. Evaluate capability gain, old-skill regression, safety, latency and memory.
6. Version candidate and adapter separately; promote only after gates.

Inheritance methods:
- **Adapter transfer:** copy compatible adapter weights, then validate.
- **Compatible weight transfer:** reuse compatible layers/blocks with explicit tensor and architecture checks.
- **Behavioural distillation:** teacher outputs teach a student, including across different model families. This is the universal fallback.
- **Feature alignment:** align selected hidden representations through learned projections and alignment loss.

Inheritance is capability-selective. Each child records parents, exact revisions, capability labels, transfer method, data manifest, transformations and benchmark evidence. Preserve parent checkpoints; child training never mutates parents.

## 6. Capacity growth

Support all three growth methods:
- **Width growth:** expand selected projections/FFN dimensions. Architecture-specific tensor mapping and function-preservation checks required.
- **Depth growth:** insert or append blocks. Use proven function-preserving initialisation where available; otherwise use a no-op/control baseline and distillation.
- **Residual-branch growth:** add a parallel compatible branch with a zero/small gate, preserving the original path initially.

Example: 0.8B → 2B/3B is a candidate family upgrade, not a guaranteed direct resize. Preserve the 0.8B checkpoint, add capacity, initialise structurally, distil from the teacher, replay old capabilities, train new capabilities and jointly tune the router. Do not promise zero retraining: the new parameters and integration require training.

Growth plan requires source immutability, target parameter count, method, initialization, tokenizer compatibility, data manifests, teacher/replay configuration and promotion gates.

## 7. Dynamic compute and hybrid knowledge

Dynamic controller selects compute based on task difficulty, uncertainty, confidence, latency budget and memory budget. Start with conservative routing policies, then learn routing from measured task outcomes. Separate active parameters, resident parameters and total parameters.

Knowledge has three stores:
- Parametric: learned skills and generalisations.
- External memory/retrieval: changing, long-tail and source-grounded facts.
- Tools: calculations, code execution, APIs and external actions.

Do not fine-tune weights for every new fact.

## 8. Identity and provenance

Tantra is the product identity regardless of backend. The orchestration layer must answer ordinary identity questions as Tantra, and identity regression tests must cover every backend and expert route. If directly asked which underlying model is used, disclose the actual provider/model and revision accurately. Never claim Tantra authored third-party pretrained weights.

## 9. Test strategy

- Schema and licence validation; missing/unknown licence blocks.
- Registry uniqueness, immutable revision/hash and lineage-cycle checks.
- Model gateway contract, timeout, malformed output and fallback tests.
- Neural adapter shape, dtype, tokenizer and hidden-state compatibility tests.
- Inheritance tests for all four methods, parent immutability and selective capability manifests.
- Growth tests for width/depth/residual branches, source hash preservation, finite tensors/gradients and pre-training equivalence where mathematically promised.
- Training tests for checkpoint resume, data split isolation, distillation, replay and deterministic configuration.
- Router tests for dynamic budget, expert overload, no-expert fallback and failure handling.
- Identity tests across all routes, plus truthful direct provenance disclosure.
- Regression suite for general language, code, maths, Hindi, English, Hinglish, safety and tool use.
- Performance suite records hardware, precision, median/p95 latency, throughput, peak RAM/VRAM and active/resident/total parameter counts.

A passing unit test proves only the tested contract. It does not prove model quality or a training improvement.

## 10. Implementation roadmap

| Phase | Work | Deliverable | Exit gate |
|---|---|---|---|
| 0 | Baseline audit | Existing tests, CLI, files and assets inventoried | Existing baseline recorded; banner untouched |
| 1 | Contracts | Registry, source manifest, lineage, identity policy | Unit tests pass |
| 2 | Model gateway | First pretrained backend adapter | Typed requests, timeout/fallback and provenance work |
| 3 | Adaptation pipeline | LoRA/SFT config, data manifests, checkpoint lineage | Small reproducible fine-tune and baseline comparison |
| 4 | Expert inheritance | Adapter transfer + behavioural distillation first | Child gains target skill; parent unchanged; replay limits regression |
| 5 | Neural expert integration | Compatible hidden-state adapter and MoE interface | Shape/semantics checks and integration benchmark pass |
| 6 | Expansion operators | Residual branch first, then depth and width | Function-preservation/control tests and source hash unchanged |
| 7 | Dynamic routing | Hierarchical router and compute controller | Better quality/compute frontier than fixed routing |
| 8 | Knowledge/tool integration | Retrieval, memory and tools | Grounding, tool correctness and safety tests |
| 9 | Scale-up | 0.8B → 2–3B family experiment, then larger families | Quality, cost, memory and latency justify promotion |

## 11. First prototype scope

Do not begin with a 0.8B or 3B training run. First prove the platform mechanics using tiny public pretrained checkpoints with permissive, verified licences and a small controlled dataset.

Prototype deliverables:
1. Source manifest and licence gate.
2. One model-gateway adapter.
3. One LoRA adaptation recipe and reproducible evaluation.
4. One child expert trained by behavioural distillation from a parent.
5. Replay-based regression check.
6. Residual-branch growth on a tiny compatible model.
7. Identity tests proving all routes identify as Tantra.
8. Machine-readable report with hashes, quality, latency and memory.

Only after this passes should the team select a production 0.8B source checkpoint. Model selection must be based on actual licence, quality, hardware and data fit—not popularity alone.


## 12. Prototype implementation status (2026-11-18)

### Implemented in this branch
- Pretrained source metadata contract with immutable revision, tokenizer hash, weight hash and explicit licence approval gate.
- Expansion-plan contract for width, depth and residual-branch capacity growth.
- Expert-inheritance metadata validation for adapter transfer, compatible weight transfer, behavioural distillation and feature alignment.
- Distillation + replay are mandatory in the default expansion plan; source checkpoints are declared immutable.
- Platform-owned identity system message and deterministic direct identity responses are integrated into the native inference and reasoning paths.
- README architecture overview, a separate SVG architecture diagram, and GitHub Actions compile/asset/full-test workflow.

### Not implemented yet
- Production pretrained checkpoint downloader/model gateway.
- Real LoRA/SFT/continued-pretraining pipeline for arbitrary third-party architectures.
- Executable weight/adapter transfer and behavioural/feature distillation trainers.
- General width-growth operator and residual-branch training implementation for arbitrary pretrained model families.
- Model registry persistence, router learning, shadow deployment, automated promotion and rollback.
- A measured 0.8B-to-2–3B expansion experiment.

### Validation status
- Isolated local contract/identity harness: 28 tests passed (the full CI suite additionally exercises the inference and reasoning integration paths).
- GitHub Actions expansion workflow: full repository suite passed (173 passed, 1 skipped); compile and JSON/SVG checks passed.
- Existing CPU validation workflow: full repository suite passed (173 passed, 1 skipped); 35-step CPU smoke training passed, and checkpoint reload produced 0.0 maximum absolute logit error.
- The separate real bilingual 10M experiment workflow failed before training because there are insufficient reviewed Hindi conversation rows for its requested split. This is a reviewed-data availability blocker, not an architecture test failure.
- No pretrained weights were downloaded or fine-tuned by this prototype branch.
