# Tantra LLM / NP-DNA

<p align="center">
  <img src="assets/tantra-banner.svg" alt="Tantra Cognitive OS banner" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/CPU--first-local-2E8B57" alt="CPU first" />
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT" />
  <img src="https://img.shields.io/badge/tokenizer-dynamic-orange" alt="Dynamic tokenizer" />
  <a href=".github/workflows/ci.yml"><img src="https://img.shields.io/badge/CI-pytest-informational" alt="CI" /></a>
</p>

Tantra LLM is a local-first research stack for building a small CPU-friendly language model and a memory-centric cognitive layer around it. The model package is `npdna`: a NeuroPlastic DNA Network with dynamic tokenization, sparse strand routing, low-rank weight generation, checkpointing, generation, training, and local memory tools.

The project is intentionally practical: it can load a seed checkpoint, run local generation, resume training from `latest`, mix supervised chat examples into raw data, and keep private datasets outside the repository.

![Tantra V4 architecture](assets/tantra-architecture.svg)

## System Snapshot

| Layer | What exists now | Notes |
| --- | --- | --- |
| Tokenizer | `AtulyaTokenizer` | Dynamic capacity, BPE-style merges, byte fallback, seed artifacts. |
| Model core | `NpDnaModel`, `NpDnaCore` | Embeddings, mesh layers, cortex, save/load, generation API. |
| Routing | Genome + sparse mesh | Low-rank strand parameter generation, top-k routing, balance loss. |
| Attention | Local attention strand | `USE_ATTENTION=True` is backed by an in-repo implementation. |
| Memory | `MemoryCortex` | Dense local memory store with save/load and retrieval hooks. |
| Training | `npdna/train_npdna_v3.py` | Resume-safe curriculum trainer with MTP, seed chat, ETA, batch/sequence controls. |
| Chat behavior | Seed chat mix | Supervised `System/User/Assistant` examples train assistant answer tokens only. |
| CLI | `npdna-*` commands | Info, chat, train, benchmark, and release helpers. |
| Agent | `NpDnaAgent` | Local tool wrapper with conservative defaults. |
| Multimodal bridge | Metadata-to-text | Image/audio/structured inputs are converted to prompt context; learned multimodal encoders are not complete yet. |

![Memory learning loop](assets/tantra-memory-learning.svg)

## Repository Layout

```text
Tantra LLM/
|-- npdna/                    # Model, tokenizer, generation, memory, trainer
|-- tests/                    # Pytest suite
|-- assets/                   # README diagrams and banner
|-- data/
|   `-- seed_chat.jsonl       # Sanitized supervised chat seed examples
|-- model/
|   |-- npdna_v3/best/        # Tiny public smoke checkpoint
|   `-- tokenizer/            # Seed tokenizer artifacts
|-- Download/                 # Local/private training data, ignored
|-- .github/workflows/ci.yml
|-- pyproject.toml
|-- SECURITY.md
|-- LICENSE
`-- README.md
```

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For a minimal runtime install:

```powershell
pip install -r requirements.txt
```

## Quick Use

Load the bundled seed checkpoint:

```python
from npdna import NpDnaCore

core = NpDnaCore.load("model/npdna_v3/best")
print(core.generate("What is gravity?", max_tokens=40))
```

Create a fresh seed model:

```python
import torch
from npdna import NpDnaCore

core = NpDnaCore.from_config("seed")
ids = core.encode("Hello from NP-DNA")
logits, balance_loss = core.model(torch.tensor([ids]))
print(logits.shape, float(balance_loss))
```

Use the CLI:

```powershell
npdna-info
npdna-chat "What is gravity?"
npdna-chat --interactive
```

`npdna-chat` prefers `model/npdna_v3/latest` when present, then falls back to `model/npdna_v3/best`.

## Training

The trainer resumes from `model/npdna_v3/latest` before `best`. Generated checkpoints, logs, and raw datasets are local artifacts by default.

Recommended correction run after adding seed chat:

```powershell
python npdna\train_npdna_v3.py --target-steps 35000 --mtp-depth 2 --threads 12 --seed-chat-ratio 0.50 --seed-ratio-min 0.35 --batch-size 4 --seq-len 256
```

Higher-throughput run if RAM allows:

```powershell
python npdna\train_npdna_v3.py --target-steps 35000 --mtp-depth 2 --threads 12 --seed-chat-ratio 0.50 --seed-ratio-min 0.35 --batch-size 8 --seq-len 256
```

Smoke run:

```powershell
python npdna\train_npdna_v3.py --steps 50 --mtp-depth 2 --threads 8
```

Important options:

| Option | Use |
| --- | --- |
| `--target-steps N` | Full training target. Omit `--steps` for normal training. |
| `--steps N` | Short smoke run only. |
| `--mtp-depth N` | Multi-token prediction depth. |
| `--seed-chat-ratio R` | Peak fraction of seed chat samples. |
| `--seed-ratio-min R` | Floor after seed-ratio decay. Use higher values for chat correction. |
| `--batch-size N` | More examples per step; uses more RAM. |
| `--seq-len N` | Longer context per example; slower but richer. |
| `--threads N` | PyTorch CPU thread count. |
| `--compile` | Try `torch.compile` for long runs. |
| `--freeze-backbone` | Train lighter adaptation parameters. |

Stage 7 datasets can contain multi-GB JSONL chunks. The trainer samples bounded windows from large files instead of loading whole chunks into memory.

## Seed Chat Data

`data/seed_chat.jsonl` contains sanitized supervised examples:

```json
{"system":"You are Atulya. Answer clearly and briefly.","user":"What is gravity?","assistant":"Gravity is the force that pulls objects with mass toward each other."}
```

During training, seed chat examples are formatted as:

```text
System: ...
User: ...
Assistant: ...
```

Only assistant answer tokens contribute to the supervised seed-chat loss. This avoids teaching the model to emit `System`, `User`, or `Assistant` headers as the answer.

Structured instruction/Q&A/chat datasets should be converted into this format. Raw articles, code, math text, and general documents should remain raw text unless they are naturally instruction-answer records.

## Checkpoints and Artifacts

| Path | Purpose |
| --- | --- |
| `model/npdna_v3/best/` | Tiny public smoke checkpoint used by tests and CI. |
| `model/npdna_v3/latest/` | Local resume checkpoint, ignored by Git. |
| `model/npdna_v3/step_*/` | Local milestone snapshots, ignored by Git. |
| `model/tokenizer/tokenizer_seed.json` | Public tokenizer seed. |
| `model/tokenizer/tokenizer_seed.pt` | Compact tokenizer metadata. |
| `model/tokenizer/vocab_samples.txt` | Sanitized vocabulary seed samples. |
| `Download/` | Private local training data, ignored by Git. |

Commit seed tokenizer/checkpoint artifacts only when they are intentionally sanitized and useful for smoke tests. Do not commit raw datasets, private logs, or ad hoc training dumps.

## Benchmark and Release

```powershell
npdna-benchmark --checkpoint model/npdna_v3/best --output model/npdna_v3/best/benchmark.json
npdna-release npdna-seed-v0.1 --checkpoint model/npdna_v3/best
```

Release folders are written under `model/releases/` and are ignored by default.

## Multimodal Context Bridge

The current multimodal layer is a prompt bridge, not an end-to-end learned vision/audio model.

```python
from npdna.multimodal_context import build_multimodal_prompt

prompt = build_multimodal_prompt(
    "Summarize this input.",
    image="assets/tantra-architecture.svg",
    structured={"project": "NP-DNA", "mode": "local"},
)
print(prompt)
```

## Agent Use

```python
from npdna import NpDnaAgent, NpDnaCore

core = NpDnaCore.from_config("seed")
agent = NpDnaAgent(core)

agent.register_tool("echo", lambda text: text)
print(agent.run("Use math_eval to calculate sqrt(81)."))
```

For shared deployments, review every tool before enabling it. Network access, file writes, shell execution, and persistent memory writes should remain behind explicit policy controls.

## Verification

```powershell
pytest
python -c "from npdna import NpDnaCore; core = NpDnaCore.load('model/npdna_v3/best'); print(core.generate('Hello.', max_tokens=3))"
```

The local suite currently covers checkpoint loading, tokenizer/config behavior, CLI routing, agent defaults, benchmark helpers, training utilities, seed-chat formatting, and large dataset window sampling.

## Security

| Area | Policy |
| --- | --- |
| Secrets | Never commit API keys, private tokens, credentials, or `.env` files. |
| Raw data | Keep local datasets in `Download/`; do not publish dataset inventories by accident. |
| Checkpoints | Treat training outputs as private unless deliberately sanitized. |
| Logs | Avoid publishing training logs that reveal folder names, private examples, or dataset composition. |
| Tools | Keep risky agent tools disabled unless the runtime has clear permissions and sandboxing. |

## License

MIT License. See [LICENSE](LICENSE).
