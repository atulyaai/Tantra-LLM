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


## Human review and replay policy

Teacher-generated demonstrations are written with a sidecar manifest containing the teacher model ID/revision and a SHA-256 of the dataset. They are blocked from LoRA or residual training until explicitly approved:

    python -m Tantra.pretrained_cli approve-dataset --dataset Datasets/pretrained/teacher.jsonl --reviewed-by "Reviewer" --review-note "Reviewed against the prompt and corrected errors"

Approval is bound to the dataset hash. Editing the dataset after approval invalidates the approval. The trainer records both new-data and replay-data hashes and the replay ratio. Use separate held-out validation and test sets; this prototype's trainer does not automatically turn training loss into a promotion score.

## Additional inheritance primitives

- Compatible weight transfer copies only exact tensor-name and tensor-shape matches into a child model; source weights remain untouched.
- Feature-alignment loss provides a differentiable projection/MSE primitive for cross-hidden-size teacher/student alignment.
- Adapter transfer requires the parent base-checkpoint hash and matching PEFT configuration.

## Checkpoint selection evidence

The official model card reports the following zero-shot instruction-model results (reported by the publisher, not independently reproduced by Tantra):

| Benchmark | SmolLM2-360M-Instruct |
|---|---:|
| IFEval (average prompt/instruction) | 41.0 |
| MT-Bench | 3.66 |
| HellaSwag | 52.1 |
| ARC (average) | 43.7 |
| PIQA | 70.8 |
| MMLU (cloze) | 32.8 |
| BBH (3-shot) | 27.3 |
| GSM8K (5-shot) | 7.43 |

The model card lists Apache-2.0 and a 0.4B parameter size. It explicitly warns that SmolLM2 primarily understands and generates English. This supports selecting it as a low-cost integration pilot, but rules it out as the sole Hindi-English foundation without separate multilingual evaluation. Source: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct


## Growth methods — implementation status

The architecture contract accepts width, depth and residual-branch expansion. This prototype has an executable, tested residual-branch operator for Llama-style decoder blocks. Width and depth are still architecture-specific expansion work; they are not falsely represented as implemented by the current CLI. The measured first branch adds 15,360 trainable parameters to the 361,821,120-parameter SmolLM2 checkpoint. It proves a safe insertion/training/reload path, not a meaningful jump to 0.8B or 2–3B. To reach that scale, we must add multiple compatible branches or carefully expand width/depth, then distil and replay-evaluate each child against the parent.

## Verified smoke evidence

On GitHub Actions CPU runner, the pinned 360M checkpoint:
- loaded and generated a Tantra identity response through the base gateway;
- completed a one-step LoRA update with 25% replay examples;
- loaded the saved LoRA adapter and generated a Tantra identity response;
- trained a residual branch with 25% replay examples, saved/reloaded it, and generated a Tantra identity response;
- kept the original checkpoint as a separate, unchanged source artifact.

These are integration/mechanics smokes only. The reported training losses are not quality scores, and no production capability gain is claimed.
