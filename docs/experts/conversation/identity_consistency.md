# identity_consistency

Keep identity facts consistent across paraphrases and multi-turn challenges.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/identity_consistency.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.
