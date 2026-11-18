# Aya source export

The pinned Aya Dataset source is Apache-2.0 and exposes human-annotated
prompt/completion pairs with `inputs`, `targets`, `language_code` and
`annotation_type`. This exporter uses only the pinned `train` split and maps
English (`eng`) and Hindi (`hin`) into Tantra's normalized intake format.
The source test split remains reserved and is never exported. The exporter selects a reproducible SHA-256-ranked sample, capped at 2,000 rows per language by default; use a larger cap only after planning reviewer capacity.

Install the existing optional dataset dependency:

```bash
pip install -e ".[finetune]"
```

Export a deterministic, family-safe review sample (default cap: 2,000 rows per language):

```bash
python -m Tantra.aya_export --out exports/aya-en-hi.jsonl --max-rows-per-language 2000
```

Then create a pending review queue:

```bash
python -m Tantra.multilingual_data intake \
  --input exports/aya-en-hi.jsonl \
  --out Datasets/review/aya-en-hi-v1 \
  --expert lang_en --domain multilingual \
  --source-id aya-instructions \
  --source-revision f9ea04583f02a8f86404ff6c58bf75fe637df8a2 \
  --source-license Apache-2.0 \
  --source-url https://huggingface.co/datasets/CohereLabs/aya_dataset \
  --source-kind human_annotated_instruction_candidates
```

Important limitations:

- Aya's source test split is not used. Do not repurpose it as training data.
- The exporter does not fabricate Hinglish. Hinglish needs its own sourced or
  human-authored examples and fluent review.
- The exporter routes English rows to `lang_en` and Hindi rows to `lang_hi` under the `multilingual` workbench domain. The global `--expert` is only a fallback. The exported rows are pending candidates, not approved training examples.
- The source's human annotation status does not substitute for Tantra review of
  correctness, language naturalness, safety, provenance and license compatibility.
- Each source row gets its own family ID. If a source contains translations or
  paraphrases of the same original task, reconcile those family IDs before split
  preparation so related records cannot leak across train/validation/test.
- Keep the source revision and export manifest with every downstream dataset version.

Official source card: https://huggingface.co/datasets/CohereLabs/aya_dataset
