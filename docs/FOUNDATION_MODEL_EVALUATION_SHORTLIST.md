# Foundation-model evaluation shortlist

This shortlist is for controlled evaluation, not a selection or quality claim.
Run all candidates against the same frozen Tantra test families, prompt format,
hardware, decoding settings and output budget.

| Candidate | Size | License | Role |
|---|---:|---|---|
| HuggingFaceTB/SmolLM2-360M-Instruct | 360M | Apache-2.0 | Existing integration baseline |
| Qwen/Qwen3-1.7B | 1.7B | Apache-2.0 | Primary multilingual evaluation candidate |

Qwen3-1.7B is pinned in the registry at immutable revision
`44c895567ee7eca2993a98be9664adf7245aa1ee`. Its model card describes a causal
chat model, Apache-2.0 licensing and 100+ languages/dialects. This is an upstream
claim, not evidence of Tantra Hindi/Hinglish quality:
https://huggingface.co/Qwen/Qwen3-1.7B

Other candidates to consider only after the first controlled comparison:

- Qwen3-0.6B: smaller Apache-2.0 multilingual model for low-memory comparison.
- Qwen3-4B: stronger capacity, Apache-2.0, but requires materially more memory.
- Gemma 3-4B IT: Google reports 140+ languages at 4B and above, but Gemma terms
  require separate license acceptance/review. Gemma 3-1B is English-only.
- Sarvam-1 2B: Indic-focused across 10 Indian languages, but the model card says
  it is a text-completion model rather than chat/instruction-ready and its license
  is non-commercial. Keep out of default product candidates.

## Experiment order

1. Run SmolLM2 and Qwen3-1.7B untouched on identical held-out prompts.
2. Record per-language output, first-token/total latency, token throughput and
   blinded 0-4 human scores for correctness, instruction adherence, naturalness,
   factuality and safety.
3. Freeze the winning base checkpoint and tokenizer hashes.
4. Train one LoRA on reviewed training data only; validation may tune it, test
   families remain sealed.
5. Compare base vs adapter on identical held-out families. Require Hindi/Hinglish
   gains and English regression within the existing <=2% gate before promotion.

Never substitute upstream model-card scores for Tantra held-out evaluation.
Never train on unreviewed, synthetic-unreviewed, or held-out examples.


## Reproducible baseline run

After this PR is merged, run the GitHub Actions workflow
`Tantra Qwen3-1.7B multilingual baseline` manually. It downloads the pinned
checkpoint, runs the current 12-case engineering fixture on CPU, and uploads a
JSON artifact containing model revision, candidate provenance, outputs and
latency. This is a smoke baseline only; the 12 prompts are not a statistically
adequate quality benchmark and must not be used as the final model-selection
score. Expand the held-out suite with reviewed, family-separated Hindi,
Hinglish and English cases before making quality claims.
