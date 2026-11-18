# Optional external teacher and benchmark shortlist

**Role:** optional research references only. This shortlist does not select Tantra's foundation model. Tantra's foundation is to be built and trained as an independent native architecture with native weights.

External checkpoints may be used for:
- Optional teacher demonstrations or distillation targets, after provenance capture and human review.
- Independent quality comparisons against native Tantra checkpoints.
- Testing the optional model gateway and integration boundaries.

They must not replace native Tantra weights or be described as Tantra-created weights. Keep each external checkpoint immutable and separately identified.

| Candidate | Size | Declared licence | Optional role |
|---|---:|---|---|
| HuggingFaceTB/SmolLM2-360M-Instruct | 360M | Apache-2.0 | Small gateway/integration fixture and optional teacher |
| Qwen/Qwen3-1.7B | 1.7B | Apache-2.0 | Optional multilingual teacher and independent comparison |

Qwen3-1.7B is pinned in the registry at immutable revision
`44c895567ee7eca2993a98be9664adf7245aa1ee`. Its model card describes a causal
chat model, Apache-2.0 licensing and 100+ languages/dialects. These are upstream
claims, not evidence of Tantra Hindi/Hinglish quality:
https://huggingface.co/Qwen/Qwen3-1.7B

Other candidates may be considered only when a specific experiment justifies them:
- Qwen3-0.6B: smaller optional comparison.
- Qwen3-4B: larger optional comparison with higher memory requirements.
- Gemma 3-4B IT: requires separate licence review and acceptance.
- Sarvam-1 2B: Indic-focused candidate, but its model card states non-commercial licensing and completion-model behaviour; do not treat it as a default product dependency.

## Experiment rules

1. Define the experiment's purpose before downloading a checkpoint: teacher, benchmark or gateway fixture.
2. Record exact model ID, immutable revision, architecture, tokenizer identity/hash, declared licence, source URL and weight hashes.
3. Keep third-party weights separate from native Tantra checkpoints and never overwrite a native parent.
4. If using teacher outputs for training, preserve teacher provenance, review/correct the demonstrations and bind approval to the dataset hash.
5. Compare native and external models on the same family-disjoint held-out prompts, decoding settings and hardware where meaningful.
6. Report language-specific human scores for correctness, instruction adherence, naturalness, factuality and safety; report latency and memory separately.
7. Never infer native capability from an upstream model card, a smoke test, a 12-case fixture or teacher-generated data.

## Existing optional Qwen3 smoke workflow

The manually dispatched Qwen3 workflow downloads the pinned checkpoint and runs a small engineering fixture. It is a gateway/evaluation mechanics test only. It is not a statistically adequate quality benchmark, does not select Tantra's foundation and does not change the native-first roadmap.
