<!-- Full-width hero banner -->
<div align="center">
  <img src="Assets/tantra_hero_banner_animated.gif" alt="Tantra LLM - Weaving Intelligence" width="100%"/>
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
  <a href="#-honest-status"><img src="https://img.shields.io/badge/status-v2_rebuild-orange.svg" alt="Status: v2 rebuild"/></a>
  <a href="#-tests"><img src="https://img.shields.io/badge/tests-18_passing-brightgreen.svg" alt="18 tests passing"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Made_in-India_🇮🇳-FF9933.svg" alt="Made in India"/></a>
</p>

**Tantra** is the brain of **Atulya** — a **Hindi-first, CPU-first language model built from scratch**.
**Atulya** is the assistant ecosystem (app, memory, tools, voice); **Tantra** is the model that thinks.
Tokenizer, architecture, training loop and WebUI are all our own code — no pretrained language model from anyone else.

> **Current state (26 Sep 2026):** v2 code is complete and tested. **No trained model yet** — the first long training run is next. No benchmark numbers are claimed until a command in this repo measures them.

```
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                               TANTRA  v2                                     │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │  💬 Hindi · Hinglish · English  ──►  64k Byte-level BPE tokenizer            │
 │                                              │                               │
 │            ═══════►  [ ALRA, ALRA, ALRA, Window-Attention ] × N  ═══════►    │
 │                       linear-time memory   exact recall of last 512 tokens   │
 │                                              │                               │
 │            + optional category layer (greetings / math / code …)             │
 │                                              ▼                               │
 │   🖥️ WebUI (chat · training · model)   🔌 OpenAI-compatible API   🎙️ voice   │
 │                                                                              │
 │        ⚡ ~70M "small" preset · grows toward ~1B · 100% offline on CPU        │
 └──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Honest status

| Part | State |
| :--- | :---: |
| **Tokenizer** — 64k Hindi+English BPE, fixed for all model sizes | ✅ |
| **Architecture** — hybrid 3 ALRA : 1 sliding-window attention (cached generation = full recompute, tested) | ✅ |
| **Training** — streams any-size JSONL, packs windows, resumes exactly (fp32 weights + optimizer + LR schedule) | ✅ |
| **Growth** — new layers start as an exact identity, so growing never makes the model forget | ✅ |
| **Category layers** — one specialist layer per topic, trained while the base is frozen | ✅ |
| **Evaluation** — validation loss + fixed 50-question recall probe every eval | ✅ |
| **WebUI** — streaming chat, live training dashboard, checkpoint manager, test & export | ✅ |
| **Trained model** | ⏳ next |
| **Speech** — Whisper (STT) / Kokoro-82M (TTS) as optional plug-ins | 🔌 optional |
| **Knowledge store (Smriti / "DNA memory")** | 🗺️ planned |

---

## 🏛️ How it works

```
text ─► 64k tokenizer ─► embedding ─► [ ALRA, ALRA, ALRA, Window-Attention ] × N ─► next token
                                         │ linear-time memory     │ exact recall of the last 512 tokens
                                         └─ constant memory per generated token on CPU
                                    + optional: category layer (e.g. "math") after the base stack
```

* **Why hybrid:** linear/recurrent layers are fast with constant memory but weak at exact recall; a few exact-attention layers fix that.
* **Why 64k vocab for every size:** the tokenizer never changes, so a model grows from ~70M toward ~1B (more layers) without retraining from zero.
* **Why ~70M first:** model size follows data (~20 tokens per parameter). `master_train.jsonl` ≈ 0.4B tokens → tens of millions of parameters.
* **Speed on CPU:** chunked recurrent attention, one-pass prompt prefill, int8 inference (`--int8`).

### 🔬 Math

**ALRA gated linear attention** (exact chunkwise-parallel form of this recurrence):
$$S_t = g_t \cdot S_{t-1} + K_t^T V_t, \quad z_t = g_t \cdot z_{t-1} + K_t, \quad o_t = \frac{Q_t \cdot S_t}{Q_t \cdot z_t + \epsilon}$$

**DPO preference loss** (`--mode dpo`):
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

---

## 🚀 Start (Windows)

Double-click **`tantra.bat`**:

```
1 Train   2 Chat   3 WebUI   4 Test the model   5 Export   6 Build tokenizer   7 Install   8 Code tests
```

Same from a terminal:

```bash
pip install -r requirements.txt
python main.py --mode train          # continues automatically; Ctrl+C saves
python main.py --mode chat --int8    # int8 = ~2x faster on CPU
python main.py --mode serve --int8   # WebUI on http://127.0.0.1:8000
python main.py --mode eval           # val loss + 50 questions + speed
python -m pytest Tests -q            # code tests
```

**Faster on a free GPU (Kaggle/Colab):** same code, same files.

```bash
git clone https://github.com/atulyaai/Tantra-LLM && cd Tantra-LLM
# put your .jsonl in Datasets/ (and latest.pt in Model/ to continue a run)
python main.py --mode train --device cuda --batch-size 32 --grad-accum 1
# download Model/latest.pt afterwards and put it back in your Model/ folder
```

---

## 🖥️ WebUI

`python main.py --mode serve` → **http://127.0.0.1:8000**

| Tab | What you get |
| :--- | :--- |
| **Chat** | Streaming replies with **Stop**, markdown/code, copy · regenerate · edit · read aloud, saved & searchable chats, settings (temperature, top-p, repetition penalty, max tokens, system prompt), voice input, "still training" notice |
| **Training** | Live tiles (status, step, loss, val loss, speed, ETA, tokens), progress bar, **loss chart**, **remembered X/50 chart**, start/stop with options, live log |
| **Model** | Loaded model + hardware, checkpoint table (step, val loss, size, date) with Load / INT8, **Run test**, **Export**, datasets |

API: OpenAI-compatible `POST /v1/chat/completions` (stream or not). Set `TANTRA_API_KEY` to protect training/checkpoint actions. Binds to 127.0.0.1 only.

---

## 🗂️ Folders

```
Tantra-LLM/
├── main.py              every command (--mode train|chat|generate|serve|eval|export|tokenizer|dpo|adapter|hardware)
├── tantra.bat           menu for the above
├── requirements.txt
├── Tantra/              the engine
│   ├── config.py        all settings + presets (tiny, small ≈70M, billion ≈1B)
│   ├── tokenizer.py     64k byte-level BPE, build once
│   ├── model.py         the network + generation + load_model()
│   ├── dataset.py       JSONL → training windows (chat masking, packing)
│   ├── train.py         training loop, checkpoints, DPO, live status for the WebUI
│   ├── eval_suite.py    validation metrics, speed, 50-question probe
│   ├── evolution.py     growing layers / category layers
│   ├── adapters.py      category registry + request router
│   ├── bitnet.py        ternary weights (for later, after training)
│   ├── export.py        small fp16 file for use (Model/tantra.pt)
│   ├── hardware.py      CPU/RAM/GPU detection
│   └── utils.py         logging, seeds, safe checkpoint loading
├── WebUI/               server.py + index.html + app.js + app.css
├── Tests/               test_model.py · test_data_train.py · test_webui.py
├── Datasets/            your .jsonl files (+ probe_50.jsonl)
├── Model/               tokenizer.json · latest.pt · best.pt · tantra.pt
└── Assets/              images
```

## 📝 Data format

One JSON object per line, any of:

```json
{"messages": [{"role": "user", "content": "भारत की राजधानी?"}, {"role": "assistant", "content": "नई दिल्ली।"}]}
{"user": "...", "assistant": "..."}
{"instruction": "...", "input": "...", "output": "..."}
{"text": "plain text / articles"}
```

In the default `sft` stage only the answers are learned. Rows whose question is generic filler ("इसके बारे में विस्तृत जानकारी दें:" + an article) are learned as plain text instead.

---

## 🧪 Tests

```bash
python -m pytest Tests -q      # 18 passed
```

## 🗺️ Roadmap (in order)

1. Rebuild tokenizer once with Hindi + English + Sanskrit samples → then freeze forever.
2. Long training run (laptop or Kaggle) until the 50-question probe climbs.
3. Knowledge store (Smriti): compressed facts + retrieval before answering.
4. Delta-rule upgrade of ALRA (better recall), then grow depth.
5. Category layers: greetings, math, code, sentiment/emotion.
6. Our own speech models (Hindi STT/TTS) replacing the plug-ins.

## 🤝 Rules for contributors

* One place for each thing. No duplicate scripts, no one-time files in the repo.
* No number goes in this README unless a command in this repo measured it.
* `python -m pytest Tests -q` must pass before every commit.

---

<p align="center">📄 MIT License · Developed by <b>Atulya AI</b> · Made in India 🇮🇳</p>
