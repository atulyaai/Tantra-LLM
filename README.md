# Tantra — CPU‑first Voice+Text LLM Agent (Flat, Fast, Friendly)

Tantra is a minimal, GPU‑optional assistant that runs anywhere (Windows/Linux/macOS). It speaks, listens, plans, uses tools, and remembers—without complex setup.

## ✨ Features
- Human‑like chat (voice + text), barge‑in friendly
- LLM‑first planning with safe tool use (web, files, calc, shell)
- Memory: short‑term context + long‑term vector memory (FAISS)
- RAG‑lite: build local indexes and cite sources
- CPU‑first runtime; GPU optional
- Flat repo, config‑driven, no hardcoded prompts

## 🚀 Quickstart
1) Python 3.10+
2) Install deps:
```bash
pip install -r requirements.txt
```
3) Start REST API:
```bash
python Training/serve_api.py
```
4) Start realtime WS (voice):
```bash
uvicorn Training.serve_realtime:app --host 0.0.0.0 --port 8001
```

## 📦 Model Assets
- Tokenizer: `Model/tokenizer.json`
- Vocabulary: `Model/vocab.json` (optional)
- Weights: `Model/tantra_weights.safetensors` (+ `Model/tantra_weights.bak`)

### Build tokenizer (example)
```bash
python Training/tokenizer_train.py --input_glob "Dataset/*.jsonl" --out Model/tokenizer.json
```
A tiny sample is provided at `Dataset/sample.jsonl`. Use your real data for meaningful tokenization.

### Weights
- Place your checkpoint at `Model/tantra_weights.safetensors`
- Update `Config/serve.yaml` paths if needed

## ⚙️ Configs
- `Config/serve.yaml` — CPU, quant, telemetry
- `Config/agent.yaml` — persona, planning, tools, memory, safety
- `Config/realtime.yaml` — VAD mode, barge‑in, wake‑word

## 🧠 Architecture (high‑level)
- `Training/model_runtime.py` — CPU text gen via transformers
- `Training/agent.py` — ReAct planner + tool registry + safety
- `Training/memory.py` — convo buffer + summarizer hooks
- `Training/embedding_build.py` — MiniLM embeddings + FAISS HNSW
- `Training/rag_index.py` — index I/O
- `Training/serve_api.py` — FastAPI REST (`/infer`, `/agent/stream`)
- `Training/serve_realtime.py` — WebSocket voice loop (`/voice`)

## 🔍 RAG Build
```bash
python Training/embedding_build.py --input_files docs.txt notes.md --out Model/
```
This creates `Model/rag_texts.json` and `Model/rag_index.faiss`.

## 🔊 Speech
- STT: Whisper (optional; CPU works, GPU optional)
- TTS: pyttsx3 (no network) or Piper (optional)

## 🗺️ Roadmap
- v1: CPU‑first voice+text, tools, memory, RAG‑lite, safety, eval
- v1.1: wake‑word, metrics dashboard, richer tools
- v2: enable vision head (`Training/model_multimodal.py`) and `/vision/infer`

## 📚 Docs
See `Docs/DOCS.md` for endpoints and examples.

## 📝 License
MIT © 2025 Atulya AI and Contributors

## 🔗 Links
- Repo: `https://github.com/atulyaai/Tantra-LLM`
