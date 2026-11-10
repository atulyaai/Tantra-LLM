# long_context expert cards


Scope of the 4 `long_context` expert specifications. Each card is a capability specification, not a trained or validated expert.


---

## compression

Compress long material under a budget while preserving required information.

- Domain: long_context; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/long_context/compression.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## hierarchy

Organize long material into a faithful hierarchical outline.

- Domain: long_context; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/long_context/hierarchy.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## retrieval

Retrieve exact evidence from long supplied documents.

- Domain: long_context; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/long_context/retrieval.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## rnn_memory

Maintain explicit sequential state over a supplied long sequence.

- Domain: long_context; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/long_context/rnn_memory.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.
