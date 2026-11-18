# Tantra foundation checkpoint and reference models

**Active foundation decision — 18 November 2026:** `Qwen/Qwen3.5-0.8B-Base` is the selected foundation checkpoint for the current Tantra training programme. This supersedes the earlier native-first decision.

| Model | Role | Status |
|---|---|---|
| `Qwen/Qwen3.5-0.8B-Base` | Active foundation for Tantra SFT | Selected; immutable revision pinned in `Tantra/qwen35_finetune.py` |
| `Qwen/Qwen3.5-0.8B` | Post-trained reference/control | Optional evaluation reference; not interchangeable with Base |
| `Qwen/Qwen3-1.7B` | Historical multilingual comparison | Retain prior experiment provenance; not current foundation |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | Historical integration smoke fixture | Optional tooling only |

## Selected foundation

- Model: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
- Pinned revision in the initial SFT pipeline: `5c8a1b97ddef11f79b47ab9d07bf82b9117413f6`.
- Model card declares Apache-2.0. Recheck upstream terms at the exact revision before redistribution.
- The Base variant is pretrained-only and intended for fine-tuning, not direct chat.
- The separate post-trained `Qwen/Qwen3.5-0.8B` is a useful reference/control but must not be substituted into the Base training run.
- Use the text-only `Qwen3_5ForCausalLM` path for this SFT experiment. Preserve the official tokenizer and chat template.
- Qwen3.5 is a hybrid Gated DeltaNet/full-attention architecture. Do not map its weights into the legacy native Tantra Transformer.

## Experiment rules

1. Pin an immutable full Hub revision; never train from a floating `main` reference.
2. Record exact model ID/revision, tokenizer identity, licence, dataset hashes, training code commit and adapter artifact hash.
3. Keep the base checkpoint immutable and store LoRA candidates separately.
4. Use reviewed, provenance-backed, family-disjoint train/validation/test splits. Never use test data for training or candidate selection.
5. Evaluate Hindi, English and Hinglish separately with the same prompts, chat template, decoding settings, token budget and hardware.
6. Report correctness, instruction adherence, naturalness, factuality and safety with human review. Report English regression explicitly.
7. Never infer Tantra capability from the upstream model card, a smoke test, validation loss or a short fixture.

## Active pipeline

```bash
python -m pip install -e ".[finetune,test]"
python -m Tantra qwen35_finetune --help
```

See [the Qwen3.5 foundation and fine-tuning roadmap](TANTRA_NATIVE_ARCHITECTURE_ROADMAP.md) for the pipeline contract, initial hyperparameters, readiness gates and honest implementation status.
