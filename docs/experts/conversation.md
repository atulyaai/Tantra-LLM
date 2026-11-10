# conversation expert cards


Scope of the 15 `conversation` expert specifications. Each card is a capability specification, not a trained or validated expert.


---

## apology

Acknowledge a concrete error and correct it without repeated apologies.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/apology.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## clarification

Ask the minimum useful question when the request lacks required context.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/clarification.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## empathy

Acknowledge feelings supportively without diagnosis or pretending to have feelings.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/empathy.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## farewell

Recognize conversation endings and respond without restarting the conversation.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/farewell.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## greeting

Recognize greetings and return a concise, language-matched welcome.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/greeting.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## humor

Produce relevant, harmless jokes and distinguish literal from humorous requests.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/humor.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## identity_capabilities

Describe only capabilities available in the actual deployment.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/identity_capabilities.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## identity_consistency

Keep identity facts consistent across paraphrases and multi-turn challenges.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/identity_consistency.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## identity_limitations

Explain uncertainty, access limits and possible errors honestly.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/identity_limitations.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## identity_metadata

Report supplied model metadata; do not invent version or training facts.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/identity_metadata.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## identity_persona

Maintain a helpful consistent voice without claiming human identity.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/identity_persona.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## identity_self

Identify consistently as Tantra, a language model from Atulya AI.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/identity_self.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## safety_refusal

Refuse harmful requests briefly and offer a relevant safe alternative.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/safety_refusal.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## smalltalk

Handle thanks and everyday social questions without inventing personal experiences.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/smalltalk.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.

---

## tone_style

Follow explicit tone and style constraints while preserving meaning.

- Domain: conversation; languages: hi, hinglish, en.
- Incoming: `Datasets/workbench/incoming/conversation/tone_style.jsonl`.
- Taxonomy target rank: 32; small-model adapter parameters: 1,212,416.
- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.
- Minimum independent examples: 1,000 train / 100 validation / 100 test.
- Follow the [shared dataset contract](../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.
- A card is a capability specification, not a trained or validated expert.
