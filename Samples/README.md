# 📦 Tantra-LLM Native Multimodal & Evaluation Manifest

This directory contains **100% Native Tantra Artifacts, Multimodal Codecs, and Evaluation Suites** created for the Tantra-LLM foundation model (developed by **Atulya AI**).

---

## 🏛️ System Components & Architecture

| System Component | Engine Used | Status |
| :--- | :--- | :---: |
| **🧠 Language Model & Reasoning** | **Tantra NeuroCore** (ALRA Attention + SGP BitNet) | **100% Native Tantra** |
| **🛠️ Tool Calling & Sandboxes** | **Tantra ToolRouter** (AST Math + Python Sandbox) | **100% Native Tantra** |
| **🔊 Audio Tokenizer & Codec** | **Tantra AudioTokenizer** (1D-Conv Discrete VQ) | **100% Native Tantra** |
| **🖼️ Image Tokenizer (Images)** | **Tantra ImageTokenizer** (2D VQ-VAE) | **100% Native Tantra** |
| **🎬 Video Tokenizer (Video)** | **Tantra VideoTokenizer** (3D-Conv Spatio-Temporal VQ)| **100% Native Tantra** |

---

## 📂 Samples Catalog & Evaluation Suites

* **🔊 [`Samples/Audio/`](Audio/)**: Native acoustic chimes and formant waveforms (`.wav`).
* **🖼️ [`Samples/Images/`](Images/)**: Reconstructed neural graphics across formats (`.png`, `.jpg`, `.bmp`, `.svg`).
* **🎬 [`Samples/Video/`](Video/)**: 3D spatio-temporal animated video loops (`.gif`).
* **📝 [`Samples/Text/`](Text/)**:
  * **[`tantra_checkpoint_raw_evals.md`](Text/tantra_checkpoint_raw_evals.md)**: Real verbatim raw inference outputs from `Model/Latest/checkpoint_latest.pt` (Step 61,000 Pre-Training Baseline).
  * **[`tantra_eval_responses.md`](Text/tantra_eval_responses.md)**: Ground-truth target gold standards for MMLU and conversational reasoning.
* **💻 [`Samples/Code/`](Code/)**: Multi-language code benchmark standards (`.py`, `.java`, `.js`, `.cpp`).
* **🛠️ [`Samples/ToolCalling/`](ToolCalling/)**: Real JSON tool call execution traces and AST logs.
