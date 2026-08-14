# Tantra-LLM

<div align="center">
  <img src="Assets/tantra_hero_logo.jpg"
       alt="Tantra Logo" width="240"/>
</div>

Tantra-LLM is an experimental, local-first language-model project built with
PyTorch. The maintained training path is a CPU-first 32K-token dense model;
it is designed to be understandable, resumable, and usable on a normal
Windows computer.

## Current status

The project is an engineering prototype, not a production assistant. The
active CPU profile is an 8-layer, 512-hidden, 8-head causal model with tied
input/output embeddings and a 32K BPE tokenizer. Base pretraining improves
next-token prediction; useful chat behaviour also requires an instruction
fine-tuning pass and evaluation.

Implemented and tested:

- JSONL dataset training with resume, gradient accumulation, checkpoints,
  metrics, ETA, and recovery.
- CPU profiles: compact dense, 10M baseline, and an experimental real top-1
  MoE comparison profile.
- Byte-level BPE tokenizer with byte fallback.
- ALRA attention, BitNet quantization, DNA codec, MoE routing, category
  adapters, self-repair/growth controls, and TokenJuice data processing.
- A local FastAPI WebUI and a CLI chat path.

These are retained system components. Do not interpret an experimental setting
or implementation as a proven speed or quality claim without measurement.
The repository does not ship model checkpoints or private datasets.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest Tests -q
```

Train the maintained CPU profile. `Latest` is the only checkpoint retained by
this command, so a resume is inexpensive in disk space.

```powershell
python -m Tantra.cpu_cli train --profile dense --attention causal `
  --vocab-size 32768 --model-dir Model\CPU_Dense32K `
  --tokenizer Model\tokenizer.json --dataset Datasets --steps 50000 `
  --batch-size 8 --grad-accum 1 --seq-len 128 --data-workers 2 `
  --checkpoint-every 500 --eval-every 1000 --resume
```

After stopping training, chat with that exact profile/checkpoint pair:

```powershell
python -m Tantra.cpu_cli chat --model-dir Model\CPU_Dense32K `
  --tokenizer Model\tokenizer.json
```

Run the WebUI with `webui\start_webui.ps1`, then open the printed local URL.

## Repository layout

<div align="center">
  <img src="Assets/tantra_architecture.jpg"
       alt="Tantra architecture" width="640"/>
</div>

The maintained source surface is intentionally compact: `Tantra/` contains the
LLM system, `Tests/` contains focused test suites, and `webui/` contains the
local browser interface. Git history, datasets, local checkpoints, assets, and
Python caches are not application-source complexity.

```text
Tantra/       Reusable model, data, training, tokenizer, adapters, and offline data utilities
webui/        Local FastAPI backend, page, CSS/JS assets, and launchers
Tests/        Pytest coverage
Datasets/     Local training data (ignored by Git)
Model/        Local tokenizer, checkpoints, and chat state (mostly ignored)
```

Supported CPU training, chat, and benchmark commands live in `Tantra.cpu_cli`.
The two offline dataset utilities are also in `Tantra/`, so there is no
separate tools folder.

## Checkpoint policy

`Model/CPU_Dense32K/Latest/checkpoint_latest.pt` is the active recovery
checkpoint. It includes model, optimizer, scheduler, and training state.
It is local-only and ignored by Git because it is large and changes during
training. The CPU training command disables `Best` and per-step archive copies.

## Development

```powershell
python -m pytest Tests -q
python -m py_compile main.py Tantra\*.py tools\*.py
```

Read [ARCHITECTURE.md](ARCHITECTURE.md) for design boundaries,
[ROADMAP.md](ROADMAP.md) for planned work, and
[CONTRIBUTING.md](CONTRIBUTING.md) before contributing.

## License

[MIT](LICENSE)
