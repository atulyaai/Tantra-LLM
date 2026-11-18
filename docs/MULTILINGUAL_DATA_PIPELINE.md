# Multilingual conversation data intake

This is a review queue, not a scraper or auto-labeler. It preserves source revision,
license, URL and original record ID. Every row starts as `pending`; none is training
eligible until an explicit reviewer decision.

## Normalize source records

Export UTF-8 JSONL, one conversation per line:

```json
{"id":"source-record-001","family_id":"source-conversation-001","language":"hi","expert":"lang_hi","domain":"multilingual","user":"आप कैसे हैं?","assistant":"मैं ठीक हूँ।"}
{"id":"source-record-002","family_id":"source-conversation-002","language":"hinglish","expert":"smalltalk","domain":"conversation","user":"Aaj kya plan hai?","assistant":"Abhi decide nahi kiya."}
```

Family IDs must identify the original conversation/problem. All translations,
paraphrases and turns from one source share the same family. Do not mint a family per row.

A normalized row may include `expert` and `domain` to route it to a specific
taxonomy expert. These row-level values override the CLI fallback. Aya export
routes `en` to `multilingual/lang_en` and `hi` to `multilingual/lang_hi`;
the Hinglish exporter routes by task category. The approval command writes the
exact `incoming/<domain>/<expert>.jsonl` paths consumed by the workbench.

```bash
python -m Tantra.multilingual_data intake \
  --input exports/aya-normalized.jsonl \
  --out Datasets/review/aya-v1 \
  --expert lang_hi --domain multilingual \
  --source-id aya-instructions \
  --source-revision f9ea04583f02a8f86404ff6c58bf75fe637df8a2 \
  --source-license Apache-2.0 \
  --source-url https://huggingface.co/datasets/CohereLabs/aya_dataset \
  --source-kind human_annotated_instruction_candidates
```

Checks cover required fields, supported language labels, basic script mismatch,
placeholder text, duplicate content and unreasonable message length. They are
triage only: they do not establish fluency, correctness, safety or license rights.
Mismatches go to quarantine for manual inspection.

## Human approval

Generate an editable spreadsheet from the pending queue:

```bash
python -m Tantra.multilingual_data sheet \
  --queue Datasets/review/aya-v1 \
  --out Datasets/review/aya-v1/review.csv
```

Review correctness, completeness, naturalness, code-switching, safety, personal
data, attribution and license compatibility. Fill `decision` (`approve` or
`reject`), `reviewer`, and `note`. Hindi and Hinglish approvals require
`fluent_language_review=true`. Do not alter `record_id`, language, expert,
domain or source fields. The CSV can be passed directly to `approve`; JSONL
decisions are also supported. Create one decision per candidate:

```json
{"record_id":"<record_id>","decision":"approve","reviewer":"reviewer-id","note":"Checked answer, language and source","fluent_language_review":true}
{"record_id":"<another-id>","decision":"reject","reviewer":"reviewer-id","note":"Incorrect answer"}
```

Then publish approved rows only:

```bash
python -m Tantra.multilingual_data approve \
  --queue Datasets/review/aya-v1 \
  --decisions Datasets/review/aya-v1/review.csv \
  --incoming Datasets/workbench/incoming --domain conversation
```

This writes taxonomy-aligned files under `incoming/<domain>/<expert>.jsonl` and refuses overwrite.
Run the existing dataset workbench `prepare` command next; it owns family-level
splits, cross-expert deduplication, immutable hashes and readiness gates. Keep
source validation/test partitions out of training.

## Training and evaluation policy

- Track English, Hindi and Hinglish counts separately; never balance by duplication.
- Split by source conversation family, not individual messages or translations.
- Compare base and LoRA on the same frozen held-out families.
- Report per-language quality and English regression; latency is not quality.
- Teacher-generated samples remain review-required and are not human-authored.
- The intake report always says `training_ready: false`; only the existing
  workbench can establish pilot readiness.

The current scoped authored-seed gap report records zero reviewed Hindi and zero
reviewed Hinglish rows. This pipeline does not claim to have filled that gap.
