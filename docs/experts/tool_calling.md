# tool_calling expert cards


Scope of the 6 `tool_calling` expert specifications. Each card is a capability specification, not a trained or validated expert.


---

## api_schema

Emit calls matching a supplied API schema without inventing fields.

- Domain: tool_calling; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/tool_calling/api_schema.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## auth_handling

Handle authorization errors without exposing or fabricating credentials.

- Domain: tool_calling; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/tool_calling/auth_handling.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## function_composition

Connect tool outputs to compatible downstream inputs.

- Domain: tool_calling; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/tool_calling/function_composition.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## parallel_calls

Parallelize independent calls while preserving dependency order.

- Domain: tool_calling; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/tool_calling/parallel_calls.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## streaming_results

Assemble partial results and distinguish partial from completed output.

- Domain: tool_calling; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/tool_calling/streaming_results.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## tool_error_handling

Interpret structured tool failures and choose a justified response.

- Domain: tool_calling; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/tool_calling/tool_error_handling.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.
