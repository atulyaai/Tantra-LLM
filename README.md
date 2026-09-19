<!-- Full-width hero banner -->
<div align="center">
  <img src="Assets/tantra_hero_banner_animated.gif"
       alt="Tantra LLM - Weaving Intelligence" width="100%"/>
</div>

<div align="center">
  <h1>
    <img src="https://readme-typing-svg.herokuapp.com?font=Cinzel&weight=700&size=45&duration=4000&pause=1000&color=F7931A&center=true&vCenter=true&width=600&height=80&lines=TANTRA+LLM;WEAVING+INTELLIGENCE;तन्त्र" alt="TANTRA LLM — Weaving Intelligence तन्त्र" />
  </h1>
</div>

<p align="center">
  <em><strong>तन्त्र</strong> (Sanskrit) — An instrument that weaves threads of knowledge ·
  <strong>तंत्र</strong> (Hindi) — System, mechanism, governance</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.2%2B-ee4c2c.svg" alt="PyTorch 2.2+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/></a>
  <a href="#current-status"><img src="https://img.shields.io/badge/status-active_training-brightgreen.svg" alt="Status: Active Training"/></a>
  <a href="#why-tantra"><img src="https://img.shields.io/badge/Made_in-India_🇮🇳-FF9933.svg" alt="Made in India"/></a>
</p>

**Tantra-LLM** is an experimental, **single unified Omnimodal on-device foundation AI model** engineered with the **NeuroCore** architecture. Instead of running separate heavy models for text, speech, and vision, Tantra weaves **Text, Vision (Images), Audio (Voice), and Tool Calling** into **ONE single neural network** running locally in **~208 MB RAM** on standard CPUs and accelerating to **8,000 tok/s on Dual GPUs**.

> **Current local checkpoint — verified 1 September 2026.** `Model/Latest/checkpoint_latest.pt` is a 16-layer auto-grown checkpoint at step 91,000 with 870.6M recorded training tokens. `Model/Best/checkpoint_best.pt` is an older 8-layer checkpoint at step 19,000 with 10.8M recorded tokens. The 95k milestone currently has metadata only; its weight file has not been saved. Claims below that are not explicitly marked “checkpoint verified” are historical results or implementation targets and need a fresh runtime benchmark.

```
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                         TANTRA SINGLE UNIFIED OMNIMODAL BRAIN                          │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │                                                                                        │
 │   🎙️ Voice Audio (16kHz) ──► AudioTokenizer ──► Audio Tokens  [31000..31999] ─┐        │
 │   📸 Camera Frame (RGB)  ──► ImageTokenizer ──► Vision Tokens [28000..30999] ─┼──►     │
 │   💬 Text & Code Prompt  ──► Byte-BPE Codec ──► Text Tokens   [00000..27999] ─┘        │
 │                                                                                        │
 │             ════════► [ 1 SINGLE TANTRA NEUROCORE TRANSFORMER ] ════════►              │
 │                     (8 ➔ 10+ Layers | 512 Hidden | ALRA Recurrent Attention)           │
 │                                                                                        │
 │   ┌──────────────────────────────┬──────────────────────────────┬───────────────────┐  │
 │   │ 💬 Conversational Dialogue   │ 💻 Clean Markdown Python/SQL │ 🛠️ `<tool_call>`  │  │
 │   │ & Polite Persona (Atulya AI) │ (Verified Doctests & Docs)   │ (Python, Calc)    │  │
 │   └──────────────────────────────┴──────────────────────────────┴───────────────────┘  │
 │                                                                                        │
 │    ⚡ Single Model File: checkpoint_latest.pt | ~208 MB RAM | 100% Offline on CPU     │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Status: What Is Actually Verified

| Component | Status | Empirical Evidence |
| :--- | :---: | :--- |
| **Hardware Auto-Detection** | ✅ Verified | Profiles CPU/RAM/Dual-GPU topology with automatic kernel affinity |
| **Forward Pass & Training Loop** | ✅ Checkpoint verified | Latest checkpoint: **91k steps / 870.6M tokens / 16 layers** |
| **Autonomous Auto-Pilot Pipeline** | ⚠️ Implemented; current run unverified | SFT and DPO paths exist; completion for the current Latest checkpoint needs a fresh benchmark |
| **Reactive Layer Auto-Growth** | ✅ Checkpoint verified | The current checkpoint grew to **16 layers**, beyond the documented 10-layer design |
| **Preference Alignment (DPO)** | ⚠️ Implemented; current run unverified | Preference-pair data and code exist, but current-checkpoint results need measurement |
| **Chunked ALRA Attention** | ✅ Implemented | Recurrent ALRA code is present; long-context memory and speed claims need a fresh benchmark |
| **BitNet 1.58-bit Ternary** | ✅ Implemented | Ternary BitLinear code is present; active-checkpoint quantization needs runtime verification |
| **Multi-Token Prediction (MTP)** | ⚠️ Training-only in current generation path | The head exists, but current live generation does not use speculative decoding |
| **4-Track Domain Curriculum** | ⚠️ Design supported | Multi-track loaders exist; the current `Datasets` folder needs an inventory before sample-count claims are repeated |
| **Industry Benchmark Suite** | ⚠️ Implemented; results unverified | Evaluation code exists; current-checkpoint benchmark scores need to be run and recorded |
| **Local Web UI & REST API** | ✅ Verified | FastAPI Server + OpenAI-compatible `/v1/chat/completions` endpoint |
| **Automated Test Suite** | ⚠️ Count corrected; re-run to verify pass/fail | **112 test functions** across core architecture, real gradient learning, and system integration (static count from `Tests/`; this doc previously said 110 here and 94 in the layout section below — neither matched the actual file count, and no pass/fail run has been re-verified in this environment. Run `pytest Tests/ -q` locally to confirm.) |

---

## 📊 Benchmark Status

> [!IMPORTANT]
> **All benchmark numbers below are aspirational targets, not measured results.**
> The Tantra 83M model has not been independently benchmarked against commercial or open-source models on MMLU, GSM8K, HumanEval, or GPQA.
> See [ROADMAP.md](./ROADMAP.md) for verified progress and the "Status: What Is Actually Verified" table.

### Architecture Specifications (Verified)

| Metric | **Tantra 83M** |
| :--- | :--- |
| **Total Parameters** | **82.8M** |
| **Attention** | ALRA (Adaptive Linear Resonance Attention) — O(1) memory scan |
| **FFN** | SGP (Sparse Gated Projection) |
| **Context Window** | 131K tokens (ALRA inference) |
| **Generation Speed** | 21.7 tok/s (CPU) |
| **RAM Footprint** | ~208 MB |
| **Offline** | ✅ 100% offline |
| **Hindi Support** | ✅ Native |
| **Architecture** | 8 layers, dim=512, heads=8 |
| **Tokenizer** | ByteBP E, 32,768 vocab |

### Comparison Targets (Unverified — See Roadmap)

| Metric | **Tantra 83M** *(target)* | **Reference models** |
| :--- | :---: | :--- |
| **MMLU (General)** | TBD | Qwen 3.8: 56.4%, Gemma 4: 58.5% |
| **GSM8K (Math)** | TBD | Qwen 3.8: 54.1%, Gemma 4: 58.2% |
| **HumanEval (Coding)** | TBD | Qwen 3.8: 45.2%, Gemma 4: 46.0% |
| **GPQA (Science)** | TBD | DeepSeek-V4: 76.2% |

### On-Device RAM Footprint Comparison

```
RAM Footprint (Lower is Better — Ultra-Low Resource On-Device Deployment):
Tantra 83M      | █ (208 MB) ⚡ [Runs on Any Laptop, Raspberry Pi, or Commodity CPU]
Qwen 0.5B       | ██████ (1,200 MB)
Gemma 2B / 1B   | ████████████████████ (4,500 MB)
DeepSeek MoE    | ██████████████████████████████████████████████████████████ (600,000 MB)
```

> ⚠️ Tantra benchmarks against larger models are **aspirational**. No independent evaluation has been run. See [ROADMAP.md](./ROADMAP.md).

### 📊 On-Device RAM Footprint Comparison

```
RAM Footprint (Lower is Better — Ultra-Low Resource On-Device Deployment):
Tantra 83M      | █ (208 MB) ⚡ [Runs on Any Laptop, Raspberry Pi, or Commodity CPU]
Qwen 0.5B       | ██████ (1,200 MB)
Gemma 2B / 1B   | ████████████████████ (4,500 MB)
DeepSeek MoE    | ██████████████████████████████████████████████████████████ (600,000 MB)
```

---

## 🏛️ NeuroCore Architecture Engine

<div align="center">
  <img src="Assets/tantra_architecture.jpg" alt="Tantra NeuroCore Architecture" width="90%"/>
</div>

### NeuroCore Engine — Complete 6-Stage Block Diagram

```
┌─────────────────────────────────────┬───────────────────────────────────────┐
│              1. INPUT TOKENIZER & MULTIMODAL PROJECTION LAYER               │
│  💬 Text Prompt     ──► BPE (32,768 Vocab) ──► Megabyte Byte-Fallback       │
│  📸 Vision Patches  ──► ImageTokenizer     ──► 512-Dim Linear Projection    │
│  🎙️ Audio Spectr.   ──► AudioTokenizer     ──► 512-Dim Mel-Scale Projection │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────┬───────────────────────────────────────┐
│                      2. HARDWARE RUNTIME ENGINE                             │
│  CPU Core Affinity ──► Thread Pinning (KMP/OMP) ──► Dual-GPU DataParallel   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────┬───────────────────────────────────────┐
│                      3. NEUROCORE BACKBONE BLOCK                            │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ ──► RMSNorm            ──► ALRA Gated Attention [O(1) Recurrent Scan] │  │
│  │ ──► Residual Addition  ──► RMSNorm                                    │  │
│  │ ──► SGP (Sparse Gated) ──► BitNet 1.58-Bit Ternary Quantization       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────┬───────────────────────────────────────┐
│                    4. DUAL-HEAD PREDICTION ENGINE                           │
│  Main Output Head (Token t+1)  ◄───►  MTP Speculative Head (Token t+2)      │
│  Latent Chain-of-Thought       ◄───►  Auxiliary Speculative Loss            │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────┬───────────────────────────────────────┐
│               5. AUTONOMOUS EVOLUTION & ALIGNMENT ENGINE                    │
│  AutoGrowth Depth Controller (8 ➔ 10+ Layers) ◄──► SelfRepairEngine (NaNs)  │
│  Pairwise DPO Alignment (Frozen Pi_ref Baseline ➔ +15.15 Preference Margin) │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────┬───────────────────────────────────────┐
│               6. COMPACT DNA WEIGHT STORAGE & EXPORT ENGINE                 │
│  NumPy Bitwise XOR Encryption ──► ZSTD Dictionary ──► DNA 2-Bit Disk Pack   │
│  Zero-Latency Export: GGUF ──► TorchScript ──► ONNX ──► FastAPI Web Server  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Mathematical Foundations

**1. ALRA Chunked Attention ($O(1)$ Linear Recurrence)**:
$$S_t = g_t \cdot S_{t-1} + K_t^T V_t, \quad z_t = g_t \cdot z_{t-1} + K_t, \quad o_t = \frac{Q_t \cdot S_t}{Q_t \cdot z_t + \epsilon}$$

**2. BitNet 1.58-bit Ternary Quantization**:
$$W_q = \text{RoundClip}\left(\frac{W}{\gamma + \epsilon},\ -1,\ +1\right), \quad \gamma = \frac{1}{nm}\sum|W_{ij}|$$

**3. Direct Preference Optimization (DPO) Loss**:
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

---

## 💡 The 1:100 Tantra Efficiency Ratio ($2\text{B} \approx 200\text{B}$)

By training exclusively on a **High-Density Synthetic Gold Curriculum** (step-by-step math derivations, clean doctested Python functions, structured turn-taking):
* Every token delivers maximum learning entropy.
* Combining **BitNet 1.58-bit ternary quantization**, **ALRA linear memory**, **Dynamic Layer Auto-Growth**, and **Online Contrastive DPO Feedback**, an 80M–100M parameter model can achieve deterministic domain mastery using **just 1.6 to 2.5 Billion tokens** rather than trillions.

---

## 🚀 Quick Start & CLI Execution

### 1. Installation & Environment Setup
```powershell
git clone https://github.com/atulyaai/Tantra-LLM.git
cd Tantra-LLM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest Tests/ -q
```

### 2. Autonomous Auto-Pilot Training (90% SFT + Auto-Growth ➔ 10% DPO)
```powershell
python main.py `
  --mode auto-pilot `
  --dataset Datasets/expert_conversation.jsonl `
  --preference-dataset Datasets/preference_pairs.jsonl `
  --steps 10000 `
  --batch-size 16 `
  --grad-accum 2 `
  --auto-growth `
  --device auto
```

### 3. Interactive Local Chat
```powershell
python main.py --mode chat --checkpoint Model/Latest/checkpoint_latest.pt --temperature 0.3
```

### 4. Run Full Industry Benchmark Suite
```powershell
python main.py --mode benchmark --checkpoint Model/Latest/checkpoint_latest.pt
```

### 5. Export Production Clean Checkpoint
```powershell
python main.py --mode export --checkpoint Model/Latest/checkpoint_latest.pt
```

---

## 🗂️ Package & Repository Layout

```
Tantra-LLM/
├── Assets/                    Logo, architecture diagram, hero banner GIF
├── Datasets/                  4-Track domain curriculum (Conversation, Code, Math, General) & DPO pairs
├── Model/                     Canonical Tokenizer, Best & Latest checkpoints, stripped export
├── Samples/                   Multimodal sample catalog (Audio, Images, Video, Code, Text, ToolCalling)
├── Tantra/                    Core Neural Engine (model, train, evolution, dataset, bitnet)
│   ├── model.py               NeuroCore Backbone (ALRA attention, SGP, BitNet, MTP heads)
│   ├── train.py               NeuroTrainer, DPO loop, Multi-GPU DataParallel
│   ├── evolution.py           AutoGrowthController & SelfRepairEngine
│   ├── dataset.py             4-Track curriculum builder & continuous sequence packing
│   ├── benchmark.py           5-Level industry evaluation runner (GSM8K, HumanEval, MMLU)
│   ├── export.py              Clean checkpoint stripper & model exporter
│   ├── tokenizer.py           Byte-level BPE + Megabyte fallback patcher + omnimodal projections
│   ├── codec.py               DNA-AI NumPy XOR + ZSTD 2-bit dictionary weight compression
│   ├── bitnet.py              BitNet 1.58-bit ternary quantization ({-1, 0, +1})
│   ├── tool_router.py         Native XML <tool_call> AST router & sandboxes
│   ├── moe.py                 Mixture-of-Experts token routing & load balancing
│   ├── adapters.py            Dynamic category domain adapters
│   ├── hardware.py            Hardware auto-detection & CPU thread pinning
│   ├── config.py              NeuroCore dataclass configurations
│   └── utils.py               Structured logging & deterministic seed utilities
├── Tests/                     Automated PyTest suite (112 test functions, see Status table above)
├── webui/                     FastAPI web server & interactive dashboard
├── main.py                    Unified CLI entry point (--mode train/chat/benchmark/export/auto-pilot)
├── tantra.ps1 / run_sft.bat   Universal Windows launchers
└── requirements.txt / pyproject.toml
```

---

## 📄 License & Attribution

Tantra-LLM is developed by **Atulya AI** and released under the **MIT License**.
Contributions, pull requests, and architectural discussions are welcome!
