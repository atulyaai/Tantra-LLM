# Foundation-model evaluation shortlist

This is a shortlist for controlled evaluation, not a model selection or quality claim.
All candidates must run against the same frozen held-out Tantra suite, tokenizer-aware
prompt format, hardware, decoding settings and output budget.

## Current registered candidates

| Candidate | Size | License | Role | Decision |
|---|---:|---|---|---|
| HuggingFaceTB/SmolLM2-360M-Instruct | 360M | Apache-2.0 | Existing integration fixture | Keep as engineering baseline |
| Qwen/Qwen3-1.7B | 1.7B | Apache-2.0 | Primary multilingual evaluation candidate | Evaluate; not promoted |

Qwen3-1.7B is registered at immutable revision
`44c895567ee7eca2993a98be9664adf7245aa1ee`. Its model card describes a causal
language model with chat template, Apache-2.0 licensing, and support for 100+
languages/dialects. Those are upstream claims, not proof of Hindi/Hinglish quality
for Tantra. See the [official model card](https://huggingface.co/Qwen/Qwen3-1.7B).

## Candidates not yet registered

- **Qwen3-0.6B**: smaller Apache-2.0 multilingual causal model; useful for low-memory
  comparison. Pin and validate its exact revision before adding it to the registry.
- **Qwen3-4B**: stronger capacity candidate, Apache-2.0, but materially heavier;
  evaluate only on suitable GPU hardware.
- **Gemma 3-4B IT**: Google reports support for 140+ languages at 4B and above. It
  requires accepting Gemma-specific terms, so do not mark it approved under the
  current registry policy without a separate license review. Gemma 3-1B is English-only
  according to Google's published family overview and is not a Hindi candidate.
- **Sarvam-1 2B**: Indic-focused and supports 10 Indian languages, but its model card
  explicitly says it is a text-completion model, not a ready chat/instruction model;
  it is under a non-commercial license. Keep it out of the default production shortlist.
  It may be considered for a separate research/continued-pretraining experiment after
  license review.

## Experiment sequence

1. Run the untouched SmolLM2 baseline and Qwen3-1.7B on exactly the same held-out
   prompts. Record output, first-token latency, total latency, generated tokens and
   per-language human scores.
2. Freeze the winning candidate's exact checkpoint/tokenizer hashes.
3. Train one LoRA adapter on reviewed training data only. Keep validation for
   hyperparameter selection and the held-out test set sealed.
4. Compare base vs LoRA on the identical held-out families. Score Hindi, Hinglish and
   English separately for correctness, instruction adherence, naturalness, factuality
   and safety (0-4 rubric, blinded review).
5. Promote only if Hindi/Hinglish improve and English regression remains within the
   existing <=2% gate, with safety and latency budgets satisfied.

Do not use model-card benchmark numbers as a substitute for Tantra's held-out test.
Do not use synthetic, unreviewed, or evaluation examples for training.
