# Pretrained candidate benchmark protocol

Run the same versioned prompt suite against the untouched foundation, each LoRA
candidate, and each residual-growth candidate. Save each JSON report with its
model revision, candidate type, artifact paths, and source commit.

## Run

```bash
python -m Tantra.pretrained_benchmark \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --checkpoint ./checkpoints/SmolLM2-360M-Instruct \
  --suite Tests/fixtures/pretrained_benchmark.jsonl \
  --output runs/benchmarks/smollm2-base.json
```

Add `--adapter` or `--residual-candidate` for an adapted candidate; do not pass
both in one run. Pass `--source-commit` to record the training/source revision
(the CLI also reads `GITHUB_SHA` when available). The report records candidate
kind, checkpoint path, adapter/residual path, and source commit. Each JSON
artifact ends with a real newline and can be parsed by strict JSON readers.

Use the same hardware, device, generation length, suite and software environment
for every comparison.

## What is measured

- Per-case and per-language latency.
- Non-empty output rate.
- Deterministic identity checks for English, Hindi and Hinglish prompts.
- Outputs retained verbatim for review.
- Runtime, model revision, candidate identity, artifact paths and source commit.

## What is deliberately not automated

A short keyword score is not a reliable measure of Hindi fluency, factuality,
reasoning, instruction adherence or safety. Those dimensions remain pending
human review. Reviewers should score each output 0–4 for correctness,
instruction adherence, language naturalness, factuality and safety; record
failure examples and adjudicate disagreements. Never promote a candidate based
only on latency, non-empty output or identity checks.

## Promotion gate

Compare base vs candidate using identical held-out cases. Require the existing
lifecycle checks (tests, identity, safety, held-out score, <=2% old-capability
regression and latency budget) and attach a signed-off human review report.
Do not use the training or replay fixtures as evaluation data.
