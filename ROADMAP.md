# Roadmap — Tantra-LLM

## ✅ Completed
- [x] Custom BPE tokenizer (64k vocab, Hindi + English)
- [x] Hybrid ALRA attention architecture
- [x] CPU-first training loop with checkpoint resume
- [x] WebUI chat interface
- [x] Pytest test suite

## 🔄 In Progress
- [ ] INT8 quantization for CPU inference
- [ ] Hindi instruction dataset (supervised fine-tuning)
- [ ] Model size presets: tiny, small (~70M), moe, billion (first long training run next)

## 🔮 Planned
- [ ] INT4 quantization for Raspberry Pi / mobile
- [x] OpenAI-compatible API server (`--mode serve`) — integration with Atulya-Tantra still to do
- [ ] GGUF export for llama.cpp compatibility
- [ ] Devanagari-aware tokenizer improvements
- [ ] Benchmark vs IndicBERT and MuRIL on Hindi NLU tasks
- [ ] Model card and HuggingFace Hub upload
- [ ] Voice-to-text pipeline integration (Whisper → Tantra-LLM)
