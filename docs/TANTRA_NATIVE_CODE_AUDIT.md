# Native Tantra code audit — 18 November 2026

**Scope:** source-level audit for the locked first milestone: native foundation + two compatible experts + residual-expansion proof.

**Audit result:** substantial native components already exist. Do not rebuild the Transformer, tokenizer, trainer or checkpoint utilities wholesale. The main missing piece is a native expert interface operating on the native model's hidden states, plus residual expansion, training/evaluation and version lineage integrated with that interface.

This is a source audit, not a runtime test report. No training was started, no model weights were changed, and no CI result is implied.

## 1. Repository baseline

- The default branch contains 132 tracked tree entries at the audit commit.
- `Model/` contains tokenizer artifacts only: `tokenizer.json` and `tokenizer_backup_8k.json`. The current `Model/tokenizer.json` is byte-level BPE with NFC normalization and 10,277 vocabulary entries (including six added special tokens).
- No native base checkpoint, trained native expert checkpoint, dataset corpus or run output is tracked in the repository tree. The handoff also records that local checkpoints and datasets were removed in the 10 November reset.
- `Tantra/pretrain_64k.json` points to ignored `runs/pretraining-20m-v1/...` paths. `Tantra/training_scope.json` describes 65,536 as a candidate vocabulary and records 10,277 as the current model vocabulary. The earlier 64K migration/training notes describe historical local artifacts; they are not currently tracked weights.
- `AGENTS.md` requires offline-by-default tests, preservation of existing artifacts, immutable data splits and no training run that overwrites `Model/` as a test.

## 2. Reusable native implementation

| Area | Existing code | What it provides | Decision |
|---|---|---|---|
| Model configuration | `Tantra/__init__.py` | Validated dimensions, GQA/RoPE constraints, presets and parameter estimate | Reuse; add a named prototype config rather than silently changing defaults |
| Native Transformer | `Tantra/train.py` — `RMSNorm`, `Attention`, `MLP`, `Block`, `TantraModel` | PyTorch decoder-only model, RoPE, grouped-query attention, SwiGLU-style MLP, tied embedding/output head, KV-cache generation | Reuse as native foundation |
| Tokenizer | `Tantra/tokenizer.py` | Byte-level BPE, NFC, chat special tokens and encode/decode | Reuse current tokenizer for prototype; freeze hash and vocabulary |
| Base trainer | `Tantra/train.py` — `TantraTrainer.train_base` and shared `_loop` | General/chat data encoding, base training, held-out loss, resumable training utilities | Reuse after dataset/config audit |
| Native expert baseline | `LoRALinear` and `TantraModel.add_expert` | Named low-rank adapters inserted into selected attention/MLP projections | Keep as legacy baseline; not the final two-expert interface |
| Router | `Router`, `TantraModel.route`, `TantraTrainer.train_router`; n-gram router in `Tantra/main.py` | Prompt-embedding classifier and existing routing/evaluation logic | Reuse concepts and tests; adapt routing to native expert modules |
| Checkpoints | `TantraModel.save/load`, `save_expert`, resumable checkpoint helpers in `Tantra/train.py` | Strict tensor-key/shape checks and optimizer/RNG/cursor resume | Reuse; add immutable prototype artifact manifests |
| Lifecycle | `Tantra/model_lifecycle.py` | Artifact hashing, candidate registration, promotion gates and rollback pointer | Reuse with native expert metadata and validated-version retention |
| Family metadata | `Tantra/model_family.py` | Parent lineage, hashes, tokenizer identity and proposal validation | Reuse; align growth enum with residual-branch-first decision |
| Residual growth research | `Tantra/pretrained_expansion.py` | Generic wrapper for Hugging Face/Llama-style decoder blocks; zero-output MLP branch; branch save/load | Do not directly use as native implementation; port the mathematical pattern into native model code |

## 3. Critical architectural gap

### Current native expert representation

The current `TantraModel` stores one shared Transformer stack. Each named expert is a set of LoRA A/B tensors injected into selected `LoRALinear` modules across the shared stack. The router predicts which adapter(s) to activate. This is a useful sparse adapter system, but Expert A and Expert B are not standalone native modules with a public, stable `hidden_states -> hidden_states` contract.

### Current native forward path

`TantraModel.forward(input_ids, ...)` embeds tokens, runs all Transformer blocks, applies final RMSNorm and immediately projects to vocabulary logits through the tied embedding matrix. It does not expose a public final hidden-state method or a native expert application stage between final normalization and the language-model head.

### Current residual branch limitation

`Tantra/pretrained_expansion.py:add_residual_branch` wraps a Llama-style `model.model.layers` or `model.layers` block, infers hidden width from a Hugging Face-style config, freezes the entire supplied model and attaches a zero-output-initialized bottleneck MLP. Its training helper expects a Hugging Face-style `model(input_ids=..., labels=...).loss` API and a Transformers tokenizer. It does not currently target native `TantraModel` blocks, which use a different forward signature and return logits directly.

Its tests use a synthetic HF-like `Tiny` model. They prove that generic wrapper is initially function-preserving and trainable, not that native Tantra residual expansion works.

### Consequence

Do not wire the external residual helper into native training by changing class names or adding compatibility shims. Implement a native hidden-state expert interface and native branch module explicitly, then test it against native `TantraModel`.

## 4. Proposed first native prototype architecture

Keep the foundation Transformer and tokenizer unchanged during this milestone.

1. Add a `forward_hidden(input_ids, caches=None, pos=0)` method to `TantraModel`. It returns final normalized hidden states with shape `[batch, sequence, d_model]`. Preserve existing `forward` logits and generation behaviour by implementing them through this method.
2. Add a small native expert interface: each expert consumes `[B,T,D]` hidden states and returns a same-shaped residual update. The interface must declare `d_model`, dtype/device behaviour and version.
3. Implement two separate native expert modules (initially general/reasoning and coding) using a bottleneck residual MLP. Their output projection starts at zero so a newly attached expert initially contributes exactly zero.
4. Apply selected expert updates to hidden states before the tied LM head. Keep the shared foundation frozen during the first expert training stage.
5. Reuse the existing prompt router as a starting point, but make it route to these expert modules. For the first test, deterministic routing labels are acceptable; learned routing is not an exit requirement.
6. Add a second gated residual branch to Expert A. Freeze the validated Expert A parent; train only the new branch initially.
7. Train the branch on new-capability examples plus reviewed replay and distillation targets. Teacher-generated examples require explicit provenance and human approval before training.
8. Save parent and child as separate immutable artifacts with tokenizer/config/source/data hashes and parent lineage.

This design makes both experts compatible because they operate on the same native hidden-state width and contract. It does not require each expert to be a separate full language model.

## 5. Exact implementation file plan

| Order | File | Planned change |
|---|---|---|
| 1 | `Tantra/train.py` | Add `forward_hidden`; keep `forward` output and generation compatible; introduce native hidden-state expert interface and residual MLP branch |
| 2 | `Tantra/native_experts.py` (new) | Define expert protocol, bottleneck residual expert, zero-init branch and compatibility validation |
| 3 | `Tantra/native_expert_training.py` (new) | Train one expert/branch at a time with frozen parent, replay, optional reviewed distillation and reproducible seeds |
| 4 | `Tantra/native_artifacts.py` (new) | Save/load immutable native expert and expanded-branch artifacts; validate hashes, tensor names/shapes and parent lineage |
| 5 | `Tantra/train.py` / router integration | Route selected native expert modules without changing the tokenizer or base checkpoint format unexpectedly |
| 6 | `Tests/test_native_experts.py` (new) | Shape/dtype/device contracts, zero-init no-op, two-expert compatibility, routing and gradients |
| 7 | `Tests/test_native_expansion.py` (new) | Native parent/child equivalence before training, parent immutability, branch-only gradients, serialization roundtrip and rollback |
| 8 | `configs/native_prototype_v0_1.json` (new) | Explicit small architecture/training/evaluation settings; no implicit mutation of existing presets |
| 9 | `docs/TANTRA_NATIVE_ARCHITECTURE_ROADMAP.md` | Update implementation status only after measured tests and experiments |
| 10 | `README.md` and `docs/HANDOFF.md` | Update claims and status after implementation is actually merged |

## 6. First prototype configuration

Use the existing tiny native preset as the first mechanics fixture (currently 8,192 vocabulary configuration, 256 hidden width, six layers, four query heads and two KV heads) only when paired with a tokenizer whose vocabulary size is explicitly matched. The checked-in tokenizer has 10,277 entries, so the model configuration must be created with `vocab_size=tokenizer.vocab_size`; do not instantiate a mismatched preset unchanged.

Do not select the 64K candidate tokenizer for this first prototype. It would confound architecture work with embedding migration and tokenizer quality. Preserve the current tokenizer and its hash. The 64K candidate can be reconsidered as a separately versioned migration after the native interface works.

The first mechanics run should use a tiny synthetic diagnostic dataset generated only inside tests. It validates gradients, tensor contracts and preservation, not Hindi/English/Hinglish capability. Real language capability experiments must use approved, provenance-backed, family-disjoint data and held-out evaluation.

## 7. Risks and constraints to resolve before training

- **LoRA and new expert modules are different abstractions.** Do not conflate current LoRA adapters with the planned native hidden-state expert modules.
- **Router training labels:** establish deterministic expert labels and a two-expert fixture before learning a router. Do not use 164 taxonomy IDs as evidence of 164 trained capabilities.
- **Loss path:** the current native model's loss is next-token cross entropy over logits. Expert modules must be applied before the tied output head and tested for correct gradient flow.
- **Residual no-op guarantee:** zero output projection should make the initial candidate match its parent. Test exact/equivalent logits on fixed inputs before any training.
- **Optimizer state:** current base pretraining has resumable optimizer/RNG/cursor support; the new expert branch trainer must either implement equivalent resume semantics or explicitly declare the first run non-resumable. Do not imply the existing native expert trainer has robust optimizer resume.
- **Data availability:** source datasets and run artifacts are not tracked in this clean baseline. Tests can prove mechanics; they cannot prove language quality.
- **Tokenizer compatibility:** vocabulary size, tokenizer file hash and special-token IDs must be in every native checkpoint manifest.
- **Version lifecycle:** `ModelLifecycle` already supports candidate registration and rollback but requires measured promotion reports. Do not promote synthetic mechanics fixtures.
- **External growth code:** `pretrained_expansion.py` and its HF-like tests remain optional research tools. A native residual proof must be a separate test suite.

## 8. Audit recommendation

The first code change should be the hidden-state seam in `TantraModel`, protected by a regression test proving existing logits are unchanged. Then introduce native experts behind that seam. This is the smallest safe path that reuses the existing native foundation and creates the correct extension point for residual growth.

**No model training should begin until the new native zero-branch equivalence and parent-immutability tests pass.** After those mechanics tests pass, use a separate ignored `runs/native-expandability-v0.1/` output directory for a tiny diagnostic experiment. Never write prototype outputs into `Model/`.
