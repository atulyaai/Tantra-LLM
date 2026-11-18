# Pretrained gateway — first integration milestone

## Candidate checkpoint

**HuggingFaceTB/SmolLM2-360M-Instruct**, pinned to revision
`c38281e01d0c0b0c36eac2f5bcb5b51fa2e803fc`.

The published model card declares Apache-2.0 and documents Transformers/chat-template inference. It is a compact decoder-only model. Its stated limitation is important: SmolLM2 primarily understands and generates English. Treat this as a gateway and LoRA-mechanics baseline, not as the final Hindi-first production foundation.

The licence decision is based on the upstream model-card declaration. Preserve the exact upstream licence and model card in the checkpoint provenance bundle before redistribution or commercial deployment.

## Runtime safety

- The registry rejects moving branch names, unapproved licences, duplicate model IDs, and remote-code execution.
- The gateway loads local snapshots only. It never downloads weights during inference.
- The upstream checkpoint remains separate and immutable; Tantra adapters/checkpoints are stored separately.
- The registry's licence approval is an intake gate, not a substitute for legal review of datasets, trademarks, or downstream distribution obligations.

## Install and inspect

```bash
pip install -e ".[pretrained,test]"
python -m Tantra.pretrained_cli info
python -m Tantra.pretrained_cli smoke --model-id HuggingFaceTB/SmolLM2-360M-Instruct --checkpoint ./checkpoints/SmolLM2-360M-Instruct
```

The smoke command measures load and generation time. It does not measure language quality. Snapshot acquisition is an explicit operator step; pin the revision above and retain file hashes.

## Acceptance gates before LoRA

Run the same held-out prompt set against Tantra's current baseline and this checkpoint on the target hardware. Record peak RAM, time-to-first-token, tokens/second, Hindi and English task scores, and exact software/hardware versions. If Hindi quality is inadequate (a known risk), keep this model as an integration fixture and choose a multilingual foundation before fine-tuning.
