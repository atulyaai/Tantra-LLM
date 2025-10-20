# Tantra (flat repo)

Cross-platform CPU-first voice+text LLM agent. Simple to run on Windows/Linux/macOS.

## Quickstart
1. Python 3.10+
2. Install deps: `pip install -r requirements.txt`
3. Start REST: `python Training/serve_api.py`
4. Start WS: `uvicorn Training.serve_realtime:app --host 0.0.0.0 --port 8001`

## Model assets
- Tokenizer: `Model/tokenizer.json`
- Vocabulary: `Model/vocab.json` (optional)
- Weights: `Model/tantra_weights.safetensors` (main), backup `Model/tantra_weights.bak`

Place your files with these names, or train via:
- Tokenizer: `python Training/tokenizer_train.py --input_glob "Dataset/*.jsonl" --out Model/tokenizer.json`
- Pretrain: `python Training/training_pretrain.py --config Config/pretrain.yaml`

## Configs
- `Config/serve.yaml` (CPU, quant, telemetry)
- `Config/agent.yaml` (persona, tools, memory)
- `Config/realtime.yaml` (VAD, barge-in)

## Notes
- Config-driven, no hardcoded prompts/tools.
- GPU optional; CPU-only works.
