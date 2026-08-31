# 📦 Tantra-LLM Native Multimodal & Tool Calling Manifest

This directory contains **100% Native Tantra Samples** created exclusively by the Tantra-LLM neural architecture and codebase (developed by **Atulya AI**).

---

## 🏛️ What is Tantra's Own System vs. Others?

| System Component | Engine Used | Status |
| :--- | :--- | :---: |
| **🧠 Language Model & Reasoning** | **Tantra NeuroCore** (ALRA Attention + SGP BitNet) | **100% Native Tantra** |
| **🛠️ Tool Calling & Sandboxes** | **Tantra ToolRouter** (AST Math + Python Sandbox) | **100% Native Tantra** |
| **🔊 Audio Tokenizer & Codec** | **Tantra AudioTokenizer** (1D-Conv Discrete VQ) | **100% Native Tantra** |
| **🖼️ Image Tokenizer (Images)** | **Tantra ImageTokenizer** (2D VQ-VAE) | **100% Native Tantra** |
| **🎬 Video Tokenizer (Video)** | **Tantra VideoTokenizer** (3D-Conv Spatio-Temporal VQ)| **100% Native Tantra** |

---

## 📂 Samples Catalog & Source Checkpoints

* **🔊 [`Samples/Audio/`](Audio/)**: Native acoustic chimes and formant waveforms (`.wav`).
* **🖼️ [`Samples/Images/`](Images/)**: Reconstructed neural graphics across formats (`.png`, `.jpg`, `.bmp`, `.svg`).
* **🎬 [`Samples/Video/`](Video/)**: 3D spatio-temporal animated video loops (`.gif`).
* **📝 [`Samples/Text/`](Text/)**: Real text generation evaluations with exact checkpoint sources:
  * `Greeting`: `checkpoint_step_44000.pt` (Step 44K)
  * `Identity & Science`: `checkpoint_step_58600.pt` (Step 58.6K, 9-Layer AutoGrowth)
  * `Math & Coding`: `checkpoint_step_59100.pt` (Step 59.1K, 10-Layer AutoGrowth)
* **💻 [`Samples/Code/`](Code/)**: Multi-language code generated at `Step 59,100` (`.py`, `.java`, `.js`, `.cpp`).
* **🛠️ [`Samples/ToolCalling/`](ToolCalling/)**: Real JSON tool call execution traces and AST logs.
