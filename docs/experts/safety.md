# safety expert cards


Scope of the 10 `safety` expert specifications. Each card is a capability specification, not a trained or validated expert.


---

## bias

Identify unsupported stereotypes and respond fairly using relevant evidence.

- Domain: safety; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/safety/bias.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## child_safety

Respond appropriately to child-related risk without sexual or exploitative material.

- Domain: safety; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/safety/child_safety.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## financial_safety

Explain financial concepts without guaranteed returns or fabricated certainty.

- Domain: safety; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/safety/financial_safety.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## jailbreak

Keep instruction boundaries when untrusted text asks to override them.

- Domain: safety; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/safety/jailbreak.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## legal_safety

Distinguish general legal information from jurisdiction-specific professional advice.

- Domain: safety; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/safety/legal_safety.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## medical_safety

Handle medical questions cautiously, with uncertainty and appropriate escalation.

- Domain: safety; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/safety/medical_safety.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## misinformation

Identify unsupported claims and separate evidence from speculation.

- Domain: safety; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/safety/misinformation.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## pii

Identify sensitive personal data and transform it according to a stated privacy policy.

- Domain: safety; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/safety/pii.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## self_harm

Respond compassionately to distress; never generate harmful procedural instructions.

- Domain: safety; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/safety/self_harm.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## toxicity

Recognize abusive content and provide a non-abusive response or rewrite.

- Domain: safety; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/safety/toxicity.jsonl`.
- Taxonomy target rank: 48; small-model adapter parameters: 1,818,624.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.
