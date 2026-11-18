# Hinglish source export

This exporter uses `DSMJ910/hinglish-instruct-10k`, pinned at
`39686808306eb6d898c8f7e86dfa93549e4ef23b`, with Apache-2.0 metadata.
The source card describes 10,594 synthetic GPT-4o-mini-generated examples,
100% Roman-script Hinglish, four task categories, and notes occasional
fabricated personal anecdotes in its own 30-example spot check.

Export only the source train split:

```bash
pip install -e ".[finetune]"
python -m Tantra.hinglish_export --out exports/hinglish-roman.jsonl
```

Then send the export through the same pending review gate:

```bash
python -m Tantra.multilingual_data intake \
  --input exports/hinglish-roman.jsonl \
  --out Datasets/review/hinglish-roman-v1 \
  --expert smalltalk --domain conversation \
  --source-id hinglish-instruct-10k \
  --source-revision 39686808306eb6d898c8f7e86dfa93549e4ef23b \
  --source-license Apache-2.0 \
  --source-url https://huggingface.co/datasets/DSMJ910/hinglish-instruct-10k \
  --source-kind synthetic_hinglish_instruction_candidates
```

Important constraints:

- This is synthetic data, not real human conversation. Review every row for
  fabricated personal experiences, factual errors, stereotypes, safety and
  unnatural register before approval.
- It is Roman-script only. It does not represent Devanagari/Latin mixed-script
  Hinglish; collect that separately.
- Casual rows route to `conversation/smalltalk`; question and support rows route to `tasks/qa`; sentiment rows route to `tasks/classification`. The global `--expert` is only a fallback.
- Variants sharing scenario and persona are assigned the same family ID to reduce
  train/test leakage.
- It is a candidate source, not an endorsement. The exporter marks it
  `training_eligible: false` and preserves a SHA-256 manifest.
- Do not use this dataset as the held-out test set; build a separate human-reviewed
  Hinglish evaluation suite.
