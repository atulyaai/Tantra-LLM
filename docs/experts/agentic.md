# agentic expert cards


Scope of the 6 `agentic` expert specifications. Each card is a capability specification, not a trained or validated expert.


---

## error_recovery

Select a bounded retry or rollback based on a supplied failure.

- Domain: agentic; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/agentic/error_recovery.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## multi_step_reasoning

Complete linked steps while retaining intermediate constraints.

- Domain: agentic; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/agentic/multi_step_reasoning.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## planning

Produce executable steps with prerequisites, success checks and stop conditions.

- Domain: agentic; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/agentic/planning.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## state_tracking

Update an explicit task state consistently across turns.

- Domain: agentic; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/agentic/state_tracking.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## task_decomposition

Break a request into non-overlapping tasks with dependencies.

- Domain: agentic; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/agentic/task_decomposition.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## tool_selection

Choose an available tool from its actual schema; abstain if none fits.

- Domain: agentic; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/agentic/tool_selection.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.
