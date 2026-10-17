# evaluation

Assess supplied work against an explicit rubric.

- Domain: tasks; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/tasks/evaluation.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.
