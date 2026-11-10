# code_agent expert cards


Scope of the 6 `code_agent` expert specifications. Each card is a capability specification, not a trained or validated expert.


---

## code_search

Locate code matching a query and cite the actual supplied paths.

- Domain: code_agent; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/code_agent/code_search.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## debug_loop

Use a supplied failure/test result to propose and validate the next fix.

- Domain: code_agent; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/code_agent/debug_loop.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## dependency_analysis

Explain actual dependencies and cycles from supplied manifests or code.

- Domain: code_agent; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/code_agent/dependency_analysis.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## refactor_planning

Plan behavior-preserving changes with dependency and test coverage.

- Domain: code_agent; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/code_agent/refactor_planning.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## repo_navigation

Find relevant files and symbols from a supplied repository snapshot.

- Domain: code_agent; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/code_agent/repo_navigation.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## test_generation

Create tests that fail for a concrete supplied defect.

- Domain: code_agent; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/code_agent/test_generation.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.
