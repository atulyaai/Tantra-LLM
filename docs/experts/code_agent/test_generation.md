# test_generation

Create tests that fail for a concrete supplied defect.

- Domain: code_agent; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/code_agent/test_generation.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.
