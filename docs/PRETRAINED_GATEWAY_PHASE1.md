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

## Phase 2 — LoRA, inheritance, residual growth and lifecycle

The same pinned foundation is now the source for the next mechanics experiments.

### LoRA fine-tuning
Install `.[finetune]`. Supply JSONL with one `{"text":"..."}` record per line. Keep train, validation and test data in separate files. The trainer records the source checkpoint tree hash before and after training and writes a separate adapter directory. It raises if the source tree changes.

### Expert inheritance
- Same checkpoint and matching PEFT configuration: transfer the adapter after checking base hash, target modules, rank, alpha and PEFT type.
- Cross-family: generate teacher demonstrations with deterministic decoding, manually review and filter them, then fine-tune the child with LoRA. This is behavioural inheritance, not direct weight copying.
- Never use teacher-generated outputs as trusted truth without review.

### Residual-branch expansion
A Llama-style decoder block is wrapped with a parallel bottleneck MLP. The branch output projection starts at zero, so the initial network function is preserved. Original decoder parameters are frozen; only the new branch is trained. Candidate weights and structural metadata are saved separately and can be reloaded for inference. The first operator supports one branch per candidate and rejects unsupported decoder layouts.

### Promotion and rollback
`ModelLifecycle` stores immutable candidate artifact hashes, parent lineage, promotion events and the previous active candidate. Promotion fails closed unless tests, identity and safety pass; held-out score is no worse than baseline; old-capability regression is <=2%; latency is within budget; and the candidate hash is unchanged. Rollback switches the active pointer to the previous candidate.

### Commands
The CLI exposes `train-lora`, `distill`, `grow-residual`, `transfer-adapter`, `promote` and `rollback`. See README for full invocations. The real checkpoint smoke is a manual GitHub Actions workflow named **Tantra pretrained checkpoint smoke test**; it downloads the pinned 360M model and measures real local generation. It does not run on every PR.

### What this phase does not claim
The CI tests use synthetic tiny modules and mocked inference. They prove mechanics, not language quality. A real LoRA run, teacher-to-student capability gain, Hindi/Hinglish evaluation and measured 0.8B-to-2–3B scaling remain explicit experiment gates.
