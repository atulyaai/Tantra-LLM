# structured_output expert cards


Scope of the 5 `structured_output` expert specifications. Each card is a capability specification, not a trained or validated expert.


---

## csv_export

Serialize supplied records as CSV with correct quoting and column order.

- Domain: structured_output; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/structured_output/csv_export.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## json_schema

Emit syntactically valid JSON satisfying a supplied schema.

- Domain: structured_output; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/structured_output/json_schema.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## markdown_tables

Render supplied records as correct, escaped Markdown tables.

- Domain: structured_output; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/structured_output/markdown_tables.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## regex_validation

Match or validate strings using a stated pattern and check edge cases.

- Domain: structured_output; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/structured_output/regex_validation.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## xml_format

Emit well-formed XML with required structure and correct escaping.

- Domain: structured_output; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/structured_output/xml_format.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.
