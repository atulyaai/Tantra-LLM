# Tantra-LLM

A 1-billion parameter Mixture-of-Adapters (MoA) language model with **164 specialized experts** across 13 domains, designed for CPU-only training and inference.

---

## Table of Contents

- [Architecture](#architecture)
- [Parameter Budget](#parameter-budget)
- [Expert Taxonomy](#expert-taxonomy)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Inference](#inference)
- [Dynamic Expert Management](#dynamic-expert-management)
- [Hardware Requirements](#hardware-requirements)

---

## Architecture

Tantra uses a **Dense-MoA** (Mixture-of-Adapters) design instead of sparse MoE. All 164 adapters are always active, weighted by a hierarchical router. This avoids branch misprediction on CPUs and gives consistent throughput.

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT TOKENS [B, T]                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              FROZEN BASE TRANSFORMER (540M)                     │
│  ┌──────────┐  ┌────────────┐  ┌───────────┐  ┌─────────────┐  │
│  │ Embed    │→ │ 26 Layers  │→ │Final Norm │→ │ Latent Head │  │
│  │ 64K×1024 │  │ Gated Attn │  │  RMSNorm  │  │ (reasoning) │  │
│  │          │  │ + SwiGLU   │  │           │  │             │  │
│  └──────────┘  └────────────┘  └───────────┘  └─────────────┘  │
│                                                                 │
│  • Never trained (frozen forever)                               │
│  • 26 transformer layers, d_model=1024                          │
│  • Gated attention (w_q, w_k, w_v, w_o, w_gate)                │
│  • SwiGLU MLP (w_up, w_gate, w_down)                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ hidden states [B, T, 1024]
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FUSED LoRA ADAPTERS (13.5M)                    │
│                                                                 │
│  For each expert i:                                             │
│    h_i = x + scale_i × (x @ A_i @ B_i)                         │
│                                                                 │
│  Fused output = Σ (router_weight_i × h_i)                       │
│                                                                 │
│  • 164 adapters, always active (dense, no sparse branching)     │
│  • Rank 16-128 depending on domain                              │
│  • Only adapters + router are trainable                         │
│  • Single fused matmul for CPU efficiency                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              HIERARCHICAL ROUTER (2M)                           │
│                                                                 │
│  Level 1: Domain Router (13 domains)                            │
│    "Which domain?" → conversation, knowledge, reasoning, ...    │
│                                                                 │
│  Level 2: Specialist Router (164 experts)                       │
│    "Which expert within domain?" → greeting, empathy, ...       │
│                                                                 │
│  • Interpretability: see which expert handled each token        │
│  • Trained briefly after each expert commits                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 OUTPUT PROJECTION (frozen)                      │
│              1024 → 64000 vocab logits                          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Dense-MoA over sparse MoE** | CPU has no branch predictor benefit from sparsity; dense matmul is faster and more predictable |
| **Frozen base forever** | Prevents catastrophic forgetting; each expert trains independently |
| **Hierarchical router** | Scalable to 164+ experts; provides expert attribution for interpretability |
| **Fused adapter matmul** | Single `x @ [A₁B₁, A₂B₂, ...]` instead of 164 separate ops |
| **Sequential training** | Train one expert at a time (Colibri method): verify → commit → next |
| **INT8 base + FP16 adapters** | Quantized base for speed; adapters stay precise for learning |

---

## Parameter Budget

```
┌──────────────────────────────────────────────────────────┐
│  GRAND TOTAL: 1,000,000,000 (1.0B)                       │
├──────────────────────────────────────────────────────────┤
│  ██████████████████████████████░░░░░░░░░░  Base (frozen) │
│  540,000,000  (540M)     54%                             │
│                                                          │
│  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Reserve       │
│  444,475,000  (444M)     44.4%  (for growth)             │
│                                                          │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Adapters      │
│  13,525,000  (13.5M)     1.35%  (164 experts)            │
│                                                          │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Router        │
│  2,000,000  (2M)         0.2%   (hierarchical)           │
└──────────────────────────────────────────────────────────┘
```

| Component | Parameters | % of Total | Trainable? |
|-----------|-----------|-----------|------------|
| Frozen base transformer | 540,000,000 | 54.0% | ❌ No |
| Reserve (future growth) | 444,475,000 | 44.4% | Reserved |
| LoRA adapters (164 experts) | 13,525,000 | 1.35% | ✅ Yes |
| Hierarchical router | 2,000,000 | 0.2% | ✅ Yes (briefly) |
| **Total** | **1,000,000,000** | **100%** | |

---

## Expert Taxonomy

164 experts across 13 domains. Knowledge experts get **1:100 ratio** (rank 128, 200K params) vs others (rank 16-64, 25-100K params).

| Domain | Experts | Rank | Params/Expert | Domain Total | Purpose |
|--------|---------|------|---------------|--------------|---------|
| **conversation** | 15 | 32 | 50K | 750K | Greetings, empathy, identity, tone, refusal |
| **knowledge** | 25 | 128 | 200K | 5.0M | Science, history, math, programming, law, etc. |
| **reasoning** | 12 | 64 | 100K | 1.2M | Logic, debugging, optimization, causal inference |
| **safety** | 10 | 48 | 75K | 750K | PII, toxicity, jailbreak, bias, medical safety |
| **tasks** | 12 | 48 | 75K | 900K | Summarization, translation, QA, classification |
| **agentic** | 6 | 64 | 100K | 600K | Planning, decomposition, state tracking |
| **tool_calling** | 6 | 64 | 100K | 600K | API schemas, parallel calls, error handling |
| **structured_output** | 5 | 48 | 75K | 375K | JSON, XML, Markdown, CSV, regex |
| **rag** | 5 | 64 | 100K | 500K | Query rewrite, ranking, citation, hallucination |
| **code_agent** | 6 | 64 | 100K | 600K | Repo navigation, test gen, debug loop |
| **multilingual** | 50 | 16 | 25K | 1.25M | 50 languages (en, zh, es, hi, ar, ...) |
| **translation** | 8 | 48 | 75K | 600K | Pivot, quality, terminology, formality |
| **long_context** | 4 | 64 | 100K | 400K | Retrieval, hierarchy, compression, memory |
| **Total** | **164** | | | **13.5M** | |

### Knowledge vs Others (1:100 Ratio)

```
Knowledge expert:   rank 128 → 200,000 params  ████████████████████
Reasoning expert:   rank  64 → 100,000 params  ██████████
Conversation expert: rank 32 →  50,000 params  █████
Multilingual expert: rank 16 →  25,000 params  ██
```

---

## How It Works

### 1. Sequential Colibri Training

Experts train **one at a time** to prevent catastrophic forgetting:

```
Expert 1 (greeting)     Expert 2 (farewell)     Expert 3 (smalltalk)
    │                       │                       │
    ▼                       ▼                       ▼
[Load Data]             [Load Data]             [Load Data]
    │                       │                       │
    ▼                       ▼                       ▼
[Pre-Verify] ──────►   [Pre-Verify] ──────►   [Pre-Verify]
    │                       │                       │
    ▼                       ▼                       ▼
[Train Adapter]         [Train Adapter]         [Train Adapter]
    │                       │                       │
    ▼                       ▼                       ▼
[Post-Verify]           [Post-Verify]           [Post-Verify]
    │                       │                       │
    ▼                       ▼                       ▼
[Regression Check]      [Regression Check]      [Regression Check]
    │                       │                       │
    ▼                       ▼                       ▼
[Commit ✓]              [Commit ✓]              [Commit ✓]
    │                       │                       │
    └───────────────────────┴───────────────────────┘
                            │
                     Frozen base never changes
```

Each expert goes through:
1. **Pre-training verification** — see baseline outputs
2. **Training loop** — only this expert's adapter weights update
3. **Post-training verification** — confirm improvement
4. **Regression check** — confirm no forgetting of previous experts
5. **Commit** — save adapter weights, update router

### 2. Dense-MoA Forward Pass

```python
# All adapters active simultaneously (no sparse selection)
hidden = frozen_base(input_ids)          # [B, T, 1024]

# Router produces weights for all 164 experts
domain_weights = domain_router(hidden)   # [B, T, 13]
expert_weights = specialist_router(hidden)  # [B, T, 164]

# Fused adapter computation (single matmul)
output = Σ(expert_weight_i × adapter_i(hidden))

# Router gives interpretability: "80% greeting, 15% empathy, 5% smalltalk"
```

### 3. Wrong-Data Detection

Per-expert monitoring catches bad training data automatically:
- **Loss spike** detection
- **Gradient explosion** detection
- **Distribution shift** detection
- **Auto-quarantine** of suspicious samples

---

## Project Structure

```
Tantra-LLM/
├── Tantra/                          # Core MoA architecture
│   ├── __init__.py                  # Package exports
│   ├── main.py                      # CLI entry point
│   ├── test_model.py                # Quick validation
│   │
│   ├── config/
│   │   ├── __init__.py              # Taxonomy loading + legacy shims
│   │   └── taxonomy.yaml            # 164 expert definitions
│   │
│   ├── model/
│   │   ├── __init__.py              # Model exports
│   │   └── tantra_model.py          # Dense-MoA architecture
│   │                                #   ├── LoRAAdapter
│   │                                #   ├── FusedLoRALayer
│   │                                #   ├── HierarchicalRouter
│   │                                #   ├── _BaseModuleWrapper (frozen base)
│   │                                #   └── TantraModel (main model)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── tantra_trainer.py        # Sequential training pipeline
│   │                                #   ├── ColibriTrainer
│   │                                #   ├── ExpertTrainConfig
│   │                                #   └── run_tantra_training
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── expert_dataset.py        # Dataset builders
│   │                                #   ├── ExpertDatasetBuilder
│   │                                #   ├── ExpertDataset
│   │                                #   └── StreamingExpertDataset
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── inference_engine.py      # Optimized CPU inference
│   │                                #   ├── OptimizedTantraModel
│   │                                #   ├── TantraInferenceEngine
│   │                                #   └── InferenceConfig
│   │
│   └── utils/
│       ├── __init__.py
│       ├── health_monitor.py        # Per-expert monitoring
│       │                            #   ├── ExpertHealthMonitor
│       │                            #   ├── ExpertHealth
│       │                            #   └── HealthStatus
│       └── lifecycle.py             # Dynamic expert management
│                                    #   ├── ExpertLifecycleManager
│                                    #   ├── ExpertStats
│                                    #   └── ReserveCapacityManager
│
├── Datasets/
│   └── experts/                     # 164 experts × 10K samples = 1.5M
│       ├── conversation/            # 15 files (greeting.jsonl, ...)
│       ├── knowledge/               # 24 files (gk_science.jsonl, ...)
│       ├── reasoning/               # 12 files
│       ├── safety/                  # 10 files
│       ├── tasks/                   # 12 files
│       ├── agentic/                 # 6 files
│       ├── tool_calling/            # 6 files
│       ├── structured_output/       # 5 files
│       ├── rag/                     # 5 files
│       ├── code_agent/              # 6 files
│       ├── multilingual/            # 42 files
│       ├── translation/             # 8 files
│       └── long_context/            # 4 files
│
├── Model/
│   └── best.pt                      # Frozen base model (540M params)
│
├── setup_expert_datasets_large.py   # Dataset generator (10K/expert)
├── README.md                        # This file
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Validate

```bash
python Tantra/main.py --mode validate --taxonomy Tantra/config/taxonomy.yaml
```

### 3. Train One Expert

```bash
python Tantra/main.py --mode train \
  --taxonomy Tantra/config/taxonomy.yaml \
  --base-model Model/best.pt \
  --data-dir Datasets/experts \
  --output checkpoints \
  --device cpu \
  --dtype bfloat16 \
  --max-experts 1 \
  --epochs 1 \
  --batch-size 2
```

### 4. Train All 164 Experts

```bash
python Tantra/main.py --mode train \
  --taxonomy Tantra/config/taxonomy.yaml \
  --base-model Model/best.pt \
  --data-dir Datasets/experts \
  --output checkpoints \
  --device cpu \
  --dtype bfloat16
```

### 5. Run Inference

```bash
python Tantra/main.py --mode inference \
  --model checkpoints/tantra_model.pt \
  --prompt "Hello, how are you today?" \
  --max-tokens 100 \
  --temperature 0.7
```

### 6. Health Check

```bash
python Tantra/main.py --mode health \
  --model checkpoints/tantra_model.pt
```

### 7. Export for Inference

```bash
python Tantra/main.py --mode export \
  --model checkpoints/tantra_model.pt \
  --export-format int8
```

---

## Training Pipeline

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | required | `train`, `inference`, `validate`, `export`, `benchmark`, `health` |
| `--taxonomy` | `Tantra/config/taxonomy.yaml` | Expert taxonomy definition |
| `--base-model` | `Model/best.pt` | Frozen base model checkpoint |
| `--data-dir` | `Datasets/experts` | Expert dataset directory |
| `--output` | `checkpoints` | Checkpoint output directory |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--dtype` | `bfloat16` | `float32`, `bfloat16`, `float16` |
| `--max-experts` | all | Limit number of experts to train |
| `--epochs` | 3 | Training epochs per expert |
| `--batch-size` | 4 | Batch size |
| `--lr` | 1e-4 | Learning rate for adapters |
| `--seq-len` | 512 | Sequence length |

### Training Flow

```
┌─────────────────────────────────────────────────┐
│  1. Load taxonomy (164 experts, 13 domains)      │
│  2. Load frozen base model (540M)                │
│  3. Build adapter layers (164 experts)           │
│  4. Initialize hierarchical router               │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  FOR EACH expert in training_order:             │
│                                                 │
│    a. Load expert dataset (10K samples)         │
│    b. Pre-training verification                 │
│    c. Freeze all adapters except current        │
│    d. Train current adapter (N epochs)          │
│    e. Post-training verification                │
│    f. Regression check (no forgetting)          │
│    g. Commit adapter weights                    │
│    h. Brief router training (100 steps)         │
│    i. Health check & lifecycle assessment       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  5. Save final checkpoint                       │
│  6. Generate training report                    │
└─────────────────────────────────────────────────┘
```

---

## Inference

### Optimized CPU Inference

```python
from Tantra.inference import TantraInferenceEngine, InferenceConfig

config = InferenceConfig(
    device="cpu",
    dtype="bfloat16",
    quantization="int8",      # INT8 base + FP16 adapters
    max_batch_size=1,
    target_tok_per_sec=30,    # ~25-40 tok/s target
)

engine = TantraInferenceEngine(
    model_path="checkpoints/tantra_model.pt",
    config=config,
)

# Generate with expert attribution
result = engine.generate(
    prompt="Explain quantum computing",
    max_tokens=200,
    temperature=0.7,
    top_p=0.9,
    interpret=True,  # Show which experts were used
)

print(result.text)
print(result.expert_attribution)
# Output: {"reasoning": 0.6, "knowledge": 0.3, "tasks": 0.1}
```

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Tokens/sec | 25-40 | CPU-only, INT8 quantized |
| Latency (first token) | < 500ms | With prompt caching |
| Memory (inference) | < 4GB | INT8 base + FP16 adapters |
| Memory (training) | < 10GB | Single expert at a time |

---

## Dynamic Expert Management

### Lifecycle Operations

```python
from Tantra.utils import ExpertLifecycleManager

manager = ExpertLifecycleManager(taxonomy, reserve_manager)

# Analyze expert usage
stats = manager.analyze_usage(training_logs)

# Automatic decisions
actions = manager.auto_manage(stats)
# Possible actions:
#   EXPAND  - expert underperforming → increase rank
#   SHRINK  - expert overperforming with redundancy → decrease rank
#   SPLIT   - expert overloaded with diverse data → create child experts
#   COMBINE - two experts highly correlated → merge them
#   QUARANTINE - wrong data detected → isolate for review
```

### Health Monitoring

```python
from Tantra.utils import ExpertHealthMonitor, HealthStatus

monitor = ExpertHealthMonitor()

# Per-batch monitoring
monitor.update(
    expert_name="greeting",
    loss=current_loss,
    grad_norm=gradient_norm,
    activation=hidden_activation,
)

# Check health
status = monitor.check("greeting")
# HealthStatus: HEALTHY | WARNING | CRITICAL | QUARANTINED

# Auto-quarantine on critical issues
if status == HealthStatus.CRITICAL:
    monitor.quarantine("greeting", reason="loss_spike")
```

### Reserve Capacity

```
Current: 13.5M adapters used / 458M available = 3% utilized
Reserve: 444M reserved for:
  - Expanding underperforming experts (+50M budget)
  - Splitting overloaded experts (+100M budget)
  - Adding new domain experts (+200M budget)
  - Router improvements (+50M budget)
  - Contingency (+144M budget)
```

---

## Hardware Requirements

### Minimum (Training)

| Component | Requirement |
|-----------|-------------|
| CPU | 4 cores / 8 threads (AMD Ryzen 5 7520U+) |
| RAM | 14GB (10GB for model + 4GB for data) |
| Storage | 5GB free (2GB datasets + 3GB checkpoints) |
| GPU | Not required (CPU-only design) |

### Recommended (Inference)

| Component | Requirement |
|-----------|-------------|
| CPU | 4+ cores with AVX2 support |
| RAM | 8GB |
| Storage | 2GB (INT8 quantized model) |

### Expected Performance

| Task | Speed | Notes |
|------|-------|-------|
| Training (per expert) | ~5 min | 10K samples, 1 epoch, batch=2 |
| Full training (164 experts) | ~14 hours | Sequential, 1 epoch each |
| Inference (generation) | 25-40 tok/s | INT8 base, FP16 adapters |
| Inference (prompt eval) | 200-500 tok/s | Batch prompt processing |

---

## Data Format

### Expert Dataset (JSONL)

Each expert has a JSONL file with one JSON object per line:

```json
{"text": "<|user|>\nHello! How can I help you today?\n<|assistant|>\nHi! I'm doing well, thanks for asking."}
{"text": "<|user|>\nGoodbye!\n<|assistant|>\nGoodbye! Have a great day!"}
```

### Generating Datasets

```bash
# Generate 10K samples per expert (1.5M total)
python setup_expert_datasets_large.py
```

### Mixed Data Sources

1. **Manual curated** (highest quality) — hand-written expert samples
2. **Auto-classified** — general corpus classified by keywords/quality
3. **Synthetic** — generated from templates with variations

---

## CLI Reference

```bash
# Validate taxonomy and datasets
python Tantra/main.py --mode validate

# Train experts
python Tantra/main.py --mode train --max-experts 5 --epochs 2

# Run inference
python Tantra/main.py --mode inference --prompt "What is AI?"

# Export for deployment
python Tantra/main.py --mode export --export-format int8

# Benchmark performance
python Tantra/main.py --mode benchmark

# Health check
python Tantra/main.py --mode health

# Full training with custom settings
python Tantra/main.py --mode train \
  --taxonomy Tantra/config/taxonomy.yaml \
  --base-model Model/best.pt \
  --data-dir Datasets/experts \
  --output checkpoints \
  --device cpu \
  --dtype bfloat16 \
  --epochs 3 \
  --batch-size 4 \
  --lr 0.0001 \
  --grad-accum 2 \
  --seq-len 512
```

---

## License

See [LICENSE](LICENSE) for details.
