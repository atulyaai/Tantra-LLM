# Real bilingual conversation experiment

Run from the repository root:

```bash
python -m pip install -e .
python examples/train_bilingual_conversation_experiment.py --output sample_output/bilingual_10m --steps 300
```

This experiment trains a fresh **~10.24M parameter** Tantra base model on the
revision-pinned OpenAssistant/oasst1 dataset. It includes only human-reviewed,
non-deleted, non-synthetic, non-flagged conversation paths where every message
is consistently labelled Hindi or English. It does not insert generated or
handwritten training answers.

Conversation trees are split 90/5/5 by root `message_tree_id`, so replies from
one conversation cannot leak across train, validation, and test. The report
contains per-language row counts, validation/test loss, and six generated
responses to prompts not used as training examples.

The workflow runs this on CPU and uploads `base.pt`, `tokenizer.json`, and
`training_report.json` as a 30-day artifact.

**Interpretation:** this is a controlled small-model experiment, not a claim
that 10M parameters can match a modern assistant. OASST1 is community data and
Hindi coverage may be much smaller than English; the report exposes actual
counts. Generated samples are qualitative probes, not a substitute for blind
native-speaker review. The existing repository tokenizer is reused, but model
weights are initialized from scratch.
