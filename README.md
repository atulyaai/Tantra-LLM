<!-- Full-width hero banner -->
<div align="center">
  <img src="Assets/tantra_hero_banner_animated.gif" alt="Tantra LLM - Weaving Intelligence" width="100%"/>
</div>

<div align="center">
  <h1>
    <img src="https://readme-typing-svg.herokuapp.com?font=Cinzel&weight=700&size=45&duration=4000&pause=1000&color=F7931A&center=true&vCenter=true&width=600&height=80&lines=TANTRA+LLM;WEAVING+INTELLIGENCE;αññαñ¿αÑìαññαÑìαñ░" alt="TANTRA LLM — Weaving Intelligence αññαñ¿αÑìαññαÑìαñ░" />
  </h1>
</div>

<p align="center">
  <em><strong>αññαñ¿αÑìαññαÑìαñ░</strong> (Sanskrit) — An instrument that weaves threads of knowledge
  <strong>αññαñéαññαÑìαñ░</strong> (Hindi) — System, mechanism, governance</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.2%2B-ee4c2c.svg" alt="PyTorch 2.2+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-F7931A.svg?style=flat-square" alt="MIT License"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Architecture-Expandable--Hybrid-purple.svg" alt="Architecture: Expandable Hybrid AI"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Tokenizer-10k_Bilingual_BPE-brightgreen.svg" alt="10k Bilingual BPE"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Made_in-India_≡ƒç«≡ƒç│-FF9933.svg" alt="Made in India"/></a>
</p>

<div align="center">
  <img src="Assets/tantra_logo.jpg" alt="Tantra LLM Official Emblem" width="240"/>
</div>

---

**Tantra** is an **expandable AI platform with a pretrained-first strategy**. Tantra
owns the orchestration, identity, model registry, routing, expert interfaces,
adaptation and evaluation pipeline. It can adapt third-party pretrained models,
connect independent specialists, and build compatible neural experts through
fine-tuning, expert inheritance and controlled capacity growth. Tantra's identity
stays Tantra across every backend; underlying model provenance remains traceable
and is disclosed truthfully when asked.

The first implementation target is an architecture and expansion prototype—not
a trillion-parameter training run. We will prove model intake, licensing checks,
LoRA adaptation, behavioural distillation, replay, residual-branch growth and
versioned promotion on small models before scaling.

- **Pretrained-first model intake**: reviewed source checkpoints, explicit licence
  gates, model gateway or compatible neural-adapter integration.
- **Expert inheritance and growth**: selective adapter/weight/behaviour/feature
  transfer; width, depth and residual-branch expansion are planned and tested
  through explicit contracts before real checkpoint expansion.
- **Native backend retained**: the repository's existing tokenizer, transformer,
  sparse adapters, router and knowledge/reasoning engine remain available as a
  backend and research baseline.
- **Tantra identity**: inference enforces the Tantra product identity while
  preserving truthful disclosure of the configured underlying model.

> **State**: this branch adds the pretrained-first platform roadmap and an
> identity-policy prototype. The full repository regression suite must pass in
> CI before this work is considered ready to merge. No pretrained model has
> been selected, downloaded or fine-tuned by this change. Developers: read
> [AGENTS.md](AGENTS.md) and [docs/HANDOFF.md](docs/HANDOFF.md) first.

---

## Current Architecture Direction: Pretrained-First and Expandable

<div align="center">
  <img src="Assets/tantra_pretrained_architecture.svg" alt="Tantra pretrained-first expandable intelligence architecture" width="100%"/>
</div>

- **Pretrained-first:** start from reviewed pretrained checkpoints; adapt with LoRA/adapters or fine-tuning where justified.
- **Hybrid integration:** use a model gateway for independent models and neural adapters only for architecture-compatible blocks.
- **Expert inheritance:** transfer selected capabilities through adapter transfer, compatible weight transfer, behavioural distillation or feature alignment.
- **Three growth methods:** width, depth and residual-branch expansion. Preserve immutable source checkpoints.
- **Learning strategy:** distillation + replay + new training, with held-out evaluation before promotion.
- **Hybrid knowledge:** learned weights + retrieval/memory + tools.
- **Stable identity:** every backend presents as Tantra; direct questions about underlying models receive truthful provenance.

**Detailed plan:** [Pretrained-First Architecture & Expansion Roadmap](docs/TANTRA_PRETRAINED_EXPANSION_ROADMAP.md).

> The existing banner, logo and original architecture image below are intentionally retained. The new SVG is an additional platform-level architecture view, not a replacement.

---

## Existing Native Implementation (Compatibility Baseline)

The following diagram documents the repository's existing native transformer path. The pretrained-first platform direction above is the forward roadmap; this implementation remains available as a backend and research baseline.

## Architecture Overview

<div align="center">
  <img src="Assets/tantra_architecture.png" alt="Tantra LLM Neural Architecture Pipeline" width="100%"/>
</div>

```
   User Query (any language)
                    │
                    ▼
       ┌──────────────────────┐
       │   Fast N-Gram Router │  Unicode-aware word & char n-grams
       └──────────────┬───────┘
                     │  active = [("gk_indic", 0.8), ("greeting", 0.2)]
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    Shared Base Transformer                             │
│      28.8M-class presets ┬ RoPE ┬ GQA ┬ SwiGLU ┬ RMSNorm ┬ Tied Head │
│                                                                          │
│      Dynamic LoRA Modules on Q, V, O, and Down-projections:             │
│             y = base(x) + Σ (x @ Aₙ @ Bₙ) · scale · wₙ                   │
│                      └ only ACTIVE experts run ─┘                        │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Tantra Decoupled Knowledge Engine                        │
│      1. DNA Integer TripleStore (O(1) relational graph)               │
│      2. Hybrid Inverted BM25 + Dense Vector Index                     │
│      3. Self-Learning Verifier (auto-curates offline facts)           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Principles

1. **The platform is the product**: pretrained models are starting points; Tantra owns
   orchestration, expert contracts, inheritance, routing, evaluation and lifecycle.
   Do not assume every model can be inserted into a neural layer.
2. **Decoupled High-Density Memory**: the transformer is not expected to memorize
   static facts. Encyclopedic facts live in an integer-indexed relational triple
   store with BM25 + dense retrieval, freeing the base for language structure.
3. **Bilingual, Devanagari-true tokenization**: ByteLevel BPE (10,277 entries,
   64K extension designed in `docs/`) preserves Indic script integrity; presets
   scale from a ~6.5M smoke model (`--preset tiny`) through ~1B.
4. **Verified answers, not vibes**: the reasoning loop drafts N candidates, runs
   exact arithmetic checks and knowledge-index comparisons, then searches before
   answering; probes and benchmarks are recorded, never assumed.

---

## Directory Structure

```
Tantra-LLM/
├─ Assets/                 # Official banners, logos, and architecture diagrams
├─ Datasets/               # Local (gitignored) data: public corpus, review bundles
├─ Model/                  # Canonical model directory (currently tokenizer artifacts)
├─ Tantra/                 # Flat package: model, training, data, review, inference
│   ├─ train.py            # Trainer, checkpoints, TantraModel, PEER, workbench
│   ├─ pretrain.py         # Public sources, corpus prep, base pilot
│   ├─ tokenizer.py        # Tokenizer candidates and vocabulary migration
│   ├─ main.py             # python -m Tantra CLI (seed..chat)
│   ├─ evaluate.py         # Reply/router evaluation
│   ├─ evaluate_conversation.py  # Memory store and indexes
│   ├─ dashboard.py        # Bulk review, review sheets, live dashboard
│   ├─ conversation_train.py     # Seed writers, conversation training
│   └─ reasoning.py        # Think -> Check -> Verify -> Learn reasoning loop
├─ Tests/                  # 4 grouped test suites (126 checks)
├─ docs/                   # Specifications, expert cards, data locks and expansion roadmap
│  ├─ TANTRA_PRETRAINED_EXPANSION_ROADMAP.md
├─ clean_workspace.ps1     # Safe local-artifact cleanup
└─ pyproject.toml          # Project configuration and metadata
```

---

## Quickstart & CLI

### 1. Existing native backend (CPU research baseline)
The commands below train the repository's existing native backend. They are
not the pretrained-first adaptation workflow described above.

```bash
python -m Tantra seed
python -m Tantra stage1 --preset tiny
python -m Tantra chat
```

The pretrained-first intake, LoRA adaptation, inheritance and expansion
runtime are being implemented in phases; this PR currently provides the
roadmap, metadata contracts, identity integration and CI coverage.
`stage1`/`base`/`expert` write into `Model/`; use `--model-dir runs/pilot` to keep
your shipped weights untouched.

### 2. Interactive Chat & Questions
```bash
python -m Tantra ask "What is the capital of India?"
python -m Tantra chat
python -m Tantra chat --int8        # fast CPU inference (INT8 dynamic quantization)
```

### 3. Bigger bases & experts
```bash
python -m Tantra base --preset small --steps 1500
python -m Tantra expert greeting --steps 300
python -m Tantra router
python -m Tantra grow --layers 2                    # deepen the base, output unchanged
```

### 4. Benchmarks, review and tests
```bash
pytest -m "not slow"                          # full suite (slow end-to-end: pytest -m slow)
python -m Tantra.bench_cpu --tag my-base      # serving-path latency/RSS benchmark
python -m Tantra workbench_cli status         # dataset/expert inventory
python -m Tantra review_cli sample --queue Q --n 50   # human review workflow
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
