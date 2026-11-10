# rag expert cards


Scope of the 5 `rag` expert specifications. Each card is a capability specification, not a trained or validated expert.


---

## citation_extraction

Attach claims to exact supplied source spans and identifiers.

- Domain: rag; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/rag/citation_extraction.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## context_compression

Compress supplied evidence while retaining facts required by a question.

- Domain: rag; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/rag/context_compression.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## doc_ranking

Rank supplied documents by their evidence for a given query.

- Domain: rag; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/rag/doc_ranking.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## hallucination_check

Flag claims unsupported by supplied evidence and propose grounded corrections.

- Domain: rag; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/rag/hallucination_check.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## query_rewrite

Rewrite a search query without introducing unsupported entities or claims.

- Domain: rag; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/rag/query_rewrite.jsonl`.
- Taxonomy target rank: 64; small-model adapter parameters: 2,424,832.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.
