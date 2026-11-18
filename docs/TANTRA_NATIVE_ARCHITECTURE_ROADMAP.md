# Tantra Qwen3.5 foundation and fine-tuning roadmap

**Architecture decision: CONFIRMED — 18 November 2026**  
**Foundation checkpoint:** `Qwen/Qwen3.5-0.8B-Base`  
**Training strategy:** reviewed multilingual supervised fine-tuning with LoRA, followed by controlled evaluation and promotion.  
**Status:** pipeline scaffold committed; no checkpoint has been downloaded or trained by this change.

This document supersedes the earlier native-first foundation decision and the native expandability milestone as the active product roadmap. The native Tantra model remains historical research code and may be retained for experiments, but it is no longer the foundation target.

## 1. Locked decision

Tantra will use Qwen3.5-0.8B-Base as its initial foundation checkpoint. The Qwen weights remain third-party pretrained weights under their upstream licence; Tantra's identity comes from its training data, fine-tuning recipe, evaluation, adapter lineage, product behaviour and subsequent validated improvements. Do not claim that the base weights were pretrained by Tantra.

Official checkpoint: [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)  
Model card revision used by the initial pipeline default: `5c8a1b97ddef11f79b47ab9d07bf82b9117413f6`  
Licence shown on the model card: Apache-2.0. Recheck the exact licence and model revision before redistribution or deployment.

The base-only checkpoint is intended for fine-tuning rather than direct chat. The separate post-trained `Qwen/Qwen3.5-0.8B` checkpoint may be retained as an evaluation reference, not silently substituted for the Base checkpoint.

## 2. Model and training boundary

- Use the Qwen3.5 text-only causal language model path (`Qwen3_5ForCausalLM`, loaded through `AutoModelForCausalLM`) for this Hindi/English/Hinglish text SFT pipeline.
- Preserve the upstream tokenizer, vocabulary, special tokens and official chat template. Do not replace it with Tantra's 10,277-token BPE tokenizer.
- The Qwen3.5 model uses a hybrid Gated DeltaNet / full-attention architecture. Do not try to load its weights into the legacy native Tantra Transformer.
- Keep the downloaded base immutable. Train LoRA adapters into separate candidate directories. Never overwrite the base or the existing `Model/` directory.
- The first training method is supervised fine-tuning (SFT), not pretraining from random initialization. Continued pretraining on raw text is a separate later experiment and requires its own corpus, objective, compute plan and evaluation.
- Teacher distillation, replay and residual adapter/branch growth can be evaluated after a reliable baseline SFT pipeline exists. They are not prerequisites for the first run.

## 3. Implemented pipeline scaffold

Entry point:

```bash
python -m pip install -e ".[finetune,test]"
python -m Tantra qwen35_finetune --help
```

Training inputs are separate immutable JSONL files following [DATASET_SPEC.md](DATASET_SPEC.md), with:
- `messages` containing system/user/assistant turns;
- stable `family_id`;
- `source.id`, `source.revision` and a known `source.license`;
- `review.status` equal to `reviewed` or `verified_auto`.

The pipeline checks JSON structure, provenance/review fields and train/validation family separation. It constructs one context-conditioned example per assistant turn and masks prompt tokens from the loss. It uses LoRA with rank 8 by default and `all-linear` target selection, evaluates validation loss, and writes the best adapter plus a run manifest to a new output directory.

Example command (replace paths with actual prepared split paths):

```bash
python -m Tantra qwen35_finetune \
  --train-jsonl Datasets/prepared/qwen35-v1/train.jsonl \
  --validation-jsonl Datasets/prepared/qwen35-v1/validation.jsonl \
  --output-dir runs/qwen35-08b-sft-v1
```

The default revision is a full immutable Hub commit SHA. Override it only with another verified full SHA. The first run defaults to 200 optimizer steps, max sequence length 1024, microbatch 1, gradient accumulation 16, LoRA rank 8 and learning rate 1e-4. These are conservative starting settings, not a tuned recipe.

## 4. Data readiness is the current blocker

Do not start training on the old native tokenizer datasets or blindly point the script at legacy expert files. Prepare a dedicated SFT corpus with the Qwen chat message format and the dataset provenance/review contract.

Required data work:
1. Inventory approved English, Hindi and Hinglish records. Pending Hindi/Hinglish drafts stay excluded until fluent review.
2. Create a unified instruction/chat corpus while retaining language, source and family metadata.
3. Freeze family-disjoint train, validation and test splits and record their hashes.
4. Report rows, unique families, source/licence coverage, assistant turns and token counts separately for Hindi, English and Hinglish.
5. Ensure the held-out benchmark families do not overlap with any training, replay or teacher-distillation examples.

The previous bilingual workflow failed because approved Hindi data was insufficient for the required splits. That remains a real blocker; the new model does not make missing reviewed data appear.

## 5. Controlled experiment sequence

| Stage | Experiment | Exit condition |
|---|---|---|
| 0 | Environment and checkpoint smoke | Transformers/PEFT versions support Qwen3.5; exact revision loads; one forward/generation pass succeeds |
| 1 | Baseline evaluation | Untouched Base and post-trained reference evaluated on the same frozen Hindi/English/Hinglish prompts |
| 2 | Tiny SFT smoke | Small reviewed subset trains, saves and reloads an adapter; base hash/revision unchanged |
| 3 | Main LoRA SFT | Frozen train/validation/test; best validation checkpoint; no train/test leakage |
| 4 | Quality evaluation | Per-language correctness, instruction following, naturalness, factuality and safety; English regression measured |
| 5 | Promotion decision | Adapter improves target language capability and passes English/regression/safety gates |
| 6 | Further learning | Only then test replay, reviewed distillation, full fine-tuning or residual adapter growth |

Never select a candidate by training loss alone. Compare base and adapter using the same prompts, chat template, decoding parameters, token budget and hardware. Keep the untouched base as the control.

## 6. Required benchmark report

Report at minimum:
- Exact base model ID and immutable revision; adapter hash and training code commit.
- Dataset split hashes, source/licence coverage, row/family/token counts and language distribution.
- Per-language results for Hindi, English and Hinglish separately.
- Correctness, instruction adherence, naturalness, factuality, safety and non-empty response rate.
- English baseline-to-adapter regression.
- Validation loss, but never treat it as a substitute for answer quality.
- Hardware, dtype, batch size, sequence length, optimizer settings, wall time, peak VRAM/RAM and tokens/second.
- Human-review sample and reviewer status.

Keep test data sealed until candidate selection is complete. Use the test split once for the final report, not repeated tuning.

## 7. Versioning and future growth

Every adapter must identify its exact base revision and dataset/code hashes. Keep all validated adapters and their evaluation reports. A base revision change invalidates direct adapter comparability and requires re-evaluation.

After a stable Qwen3.5 SFT baseline:
- Test replay to reduce regressions when adding capabilities.
- Test distillation only with documented teacher provenance and reviewed targets.
- Evaluate separate adapters or residual adapter branches only if a measured capability boundary justifies them.
- Consider full-parameter fine-tuning only when LoRA demonstrably limits the target capability and sufficient GPU memory/data exist.
- Consider a future native Tantra foundation only as a separate strategic project, not as an undocumented replacement for this Qwen-based foundation.

## 8. Honest status

Committed in source:
- Qwen3.5-0.8B LoRA SFT pipeline scaffold.
- Immutable revision argument/default, reviewed JSONL validation, family-leakage check, assistant-only label masking, validation selection and adapter run manifest.
- Unit tests for data validation and label masking.

Not yet done:
- No model download or actual model load test in the user's runtime.
- No end-to-end training run.
- No verified GPU/VRAM inventory.
- No approved multilingual SFT split ready for this pipeline.
- No measured quality improvement and no promoted Tantra adapter.

The next action is environment and data readiness, then a tiny end-to-end smoke run. Do not launch the full experiment until those checks pass.
