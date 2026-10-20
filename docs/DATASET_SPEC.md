# Expert dataset contract (v1)

This is the shared contract for every expert card. A capability name, file count,
row count, saved adapter, and validated capability are different things.

## One JSON object per line

```json
{"schema_version":1,"expert":"math","language":"hi","family_id":"arithmetic:+:17:21","template_id":"arithmetic:+","messages":[{"role":"user","content":"17 + 21 की गणना करें। केवल पूर्णांक लिखें।"},{"role":"assistant","content":"38"}],"source":{"id":"tantra-verifiable-v1","revision":"1","license":"MIT","kind":"synthetic_verified"},"review":{"status":"verified_auto","reviewer":"exact arithmetic/serialization verifier"},"verification":{"kind":"arithmetic","a":17,"b":21,"op":"+"}}
```

`docs/dataset_record.schema.json` specifies the machine-readable shape. The
workbench additionally checks placeholders, message roles, verifier results,
family leakage, and frozen split hashes. The schema alone does not prove quality.

- `expert`: exact taxonomy ID; do not create alternate names for the same task.
- `language`: actual response language, not a tag around an English paragraph.
- `family_id`: stable source conversation, document, or independently answerable
  problem ID. All paraphrases, translations, turns and mixed-intent copies from
  the same source/problem must use the same family. Do not mint a family per spelling variant.
- `template_id`: stable generator/task pattern; report this separately from row
  and family counts. Thousands of numeric instances do not imply thousands of skills.
- `source`: real provenance, immutable revision, license, and source kind. Public
  material should also include its source URL and original record ID. `unknown`
  licenses are not accepted for prepared training data.
- `review`: `reviewed` with an accountable reviewer, or `verified_auto` for a
  supported deterministic verifier. Never invent human review or citations.
- `verification`: automatically checked arithmetic, JSON serialization, or CSV
  serialization. No untrusted Python or shell code is executed by validators.

Safety, medical, legal and financial data require domain review. Native/fluent
review is necessary for language experts. A generator cannot self-certify those
capabilities merely by writing `reviewed` into metadata.

## Data preparation and readiness

Use `Datasets/workbench/incoming/<domain>/<expert>.jsonl`. The checked generator
uses `verified-<expert>.jsonl` to avoid overwriting manually authored examples.
Existing `Datasets/curated` and `Datasets/experts` files are audited, not silently
approved or copied. The old `Sample N of 10000` corpus is filler and is rejected.
Legacy examples lacking provenance/review need a reviewed conversion before import.

`prepare` creates exact, case-sensitive message hashes (code, identifiers and JSON
values retain case), removes per-expert duplicates, and joins duplicate-connected
families globally before deterministic 90/5/5 splitting. All copies of the same
conversation/problem stay in one split. Source families and semantic near-duplicates
still require responsible authoring/review; exact hashing does not discover every
paraphrase automatically.

The prepared manifest is immutable and records SHA-256 for every split. Use a new
`--out` directory for a new dataset version. The training command verifies hashes
and family separation before loading weights. Rejected records are reported with
file and line in `quarantine.jsonl`.

`data_ready` requires 1,000 train / 100 validation / 100 test records and independent
families. It means **ready for a pilot**, not production quality. Generated tasks
currently have narrow template coverage; inspect `template_count` and language
coverage before extending training. Never repeat rows to manufacture readiness.

## Non-overlap and deployment policy

- `math`: arithmetic/algebra computation; `gk_math`: mathematical definitions and
  facts. Arithmetic examples are generated once, for `math` only.
- `qa`: answer supplied evidence; `gk_general`: referenced general knowledge.
- `translation`: broad translation; `translate_*`: a specific, independently
  evaluated constraint such as terminology or formality. Do not copy the same
  translation file into every specialist.
- `lang_*`: language competence, not a second copy of every topic expert.
- `formatting`: general representation requests; JSON/CSV/XML specialists must
  demonstrate their own structural edge cases and incremental benefit.
- `retrieval`: long-document evidence location; `doc_ranking`: ordering supplied
  candidate documents. Shared examples need explicit multi-label intent and one family.
- `logic`, `symbolic_reasoning`, and planning experts require different evaluation
  tasks. If a specialist adds no held-out benefit, keep its deployment disabled.

Keep all 164 IDs for compatibility and authoring coverage. Do not allocate 164
weight tensors merely to satisfy a count. Start at rank 8 and train one candidate
adapter at a time. Taxonomy ranks are targets, not measured deployment budgets.

## Quality gates

1. Review independent provenance and answer correctness, plus language coverage.
2. Compare held-out loss against the same frozen base with the adapter inactive.
3. Score answers against exact verifiers or an explicit expert-specific rubric.
4. Require at least 90% on 100 validation and 100 test answer checks for supported
   automatic release checks; other experts need a reviewed evaluation implementation.
5. Evaluate router macro-F1, mixed intents, and regressions on existing experts
   before deployment. A lower training loss or routing accuracy is not answer quality.
6. Keep the matching base, tokenizer and dataset hashes with every candidate.
   Changing the base can invalidate earlier adapters; re-evaluate them afterward.

The `train` command exports a candidate to a separate output directory. A pilot
is marked `pilot_unvalidated`. Even an answer-validated candidate is not automatically
installed into `Model/`, and the router is not silently retrained.

### Reviewed-sheet translation grouping (2026-10-20)

The workbench groups `tantra-conversation-human-v1` records with an
`<expert>:draft-v1:<situation>` family under the original
`tantra-conversation-drafts-v1` split namespace. English, Hindi and Hinglish
versions therefore share a split even though their attribution sources differ.
The source fields remain unchanged; unrelated publishers are not merged merely
because they use the same family ID. Existing English family hashes are unchanged.
Previously prepared bundles are immutable: build a new bundle to apply this fix.
