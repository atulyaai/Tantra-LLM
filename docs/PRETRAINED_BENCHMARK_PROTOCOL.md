# Native Tantra and optional external-model benchmark protocol

This protocol supports comparisons between native Tantra checkpoints/experts and optional external teacher or gateway references. External checkpoints are not Tantra's foundation and must never be merged into native lineage or relabelled as Tantra-created weights.

## Run an optional external-model comparison

The existing CLI can benchmark an optional external checkpoint:

```bash
python -m Tantra.pretrained_benchmark \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --checkpoint ./checkpoints/SmolLM2-360M-Instruct \
  --suite Tests/fixtures/pretrained_benchmark.jsonl \
  --output runs/benchmarks/smollm2-reference.json
```

Add `--adapter` or `--residual-candidate` only for the corresponding optional external-model mechanics experiment; do not pass both in one run. Pass `--source-commit` to record the training/source revision (the CLI also reads `GITHUB_SHA` when available). Keep external checkpoints, adapters and reports separate from native Tantra checkpoints.

For native expansion experiments, use the native evaluation harness and record the same provenance and measurement fields. Compare parent and expanded native experts on identical held-out cases. Do not assume the external-model CLI validates the native residual-expansion implementation.

## Comparison controls

Use the same hardware, device, generation length, suite, prompt formatting, decoding settings and software environment for every comparison. Record the model identity and candidate type explicitly.

## What is measured

- Per-case and per-language latency.
- Non-empty output rate.
- Identity checks for English, Hindi and Hinglish prompts.
- Outputs retained verbatim for human review.
- Runtime, exact model/checkpoint revision, tokenizer hash, candidate identity, artifact paths and source commit.
- For native growth: parent/child hashes, initial equivalence result, old/new capability scores, replay and distillation manifests, and rollback result.

## What is deliberately not automated

A short keyword score is not a reliable measure of Hindi fluency, factuality, reasoning, instruction adherence or safety. Those dimensions require human review. Reviewers should score each output 0–4 for correctness, instruction adherence, language naturalness, factuality and safety; record failure examples and adjudicate disagreements.

Never promote a candidate based only on latency, non-empty output or identity checks. A passing software test proves only the tested contract; it does not prove model quality.

## Promotion gate

For native expansion, compare the immutable parent and child using identical held-out cases. Require:
- Reproducible training and checkpoint reload.
- Parent checkpoint and tokenizer hashes unchanged.
- Improvement on the declared new capability.
- Old-capability regression within a pre-registered tolerance (the current lifecycle reference gate is <=2%, where applicable).
- Identity and safety checks.
- Human-reviewed held-out quality report.
- Latency and memory within the declared budget.
- Immutable lineage manifest and tested rollback.

Do not use training, replay, teacher-demonstration or validation fixtures as held-out evaluation data. Preserve every validated version; failed candidates are archived and never overwrite the active version.
