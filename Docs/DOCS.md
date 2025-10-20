# Tantra Docs (v1)

## Run
- Start API: `python Training/serve_api.py`
- Start Realtime: `uvicorn Training.serve_realtime:app --host 0.0.0.0 --port 8001`

## Endpoints
- POST `/infer` {"prompt": "..."}
- POST `/agent/stream` {"prompt": "..."}
- WS `/voice` — send text transcripts (scaffold) and receive partial/final

## Configs
- `Config/serve.yaml` CPU-only, quantization, telemetry
- `Config/agent.yaml` persona, tools, memory, safety
- `Config/realtime.yaml` VAD, barge-in, wake-word

## Next steps
- Replace `dummy_llm` with real Mamba inference
- Plug Whisper (STT) and Piper (TTS) into `/voice`
- Implement FAISS embedding builder and retrieval in memory

## Roadmap (concise)
- v1: CPU-first voice+text, tools, memory, RAG-lite, safety, eval
- v1.1: offline wake-word, better tools, metrics dashboard
- v2: enable vision head (`Training/model_multimodal.py`), `/vision/infer`
