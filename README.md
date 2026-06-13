# NP-DNA: NeuroPlastic DNA Network

<p align="center">
  <img src="assets/tantra-banner.svg" alt="Tantra / NP-DNA banner" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/CPU--first-local-2E8B57" alt="CPU first" />
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT" />
  <img src="https://img.shields.io/badge/vocab-dynamic%204K--256K-orange" alt="Dynamic vocab" />
</p>

NP-DNA is a CPU-first, local language-model prototype built around dynamic vocabulary growth, sparse mixture-of-strands routing, low-rank DNA-style weight generation, and a persistent retrieval cortex.

```text
NP-DNA = AtulyaTokenizer + Genome + Sparse Mesh Layers + Memory Cortex + Generation / Agent Helpers
```

The current checkout is centered on the `npdna` package. It includes model architecture, tokenizer, generation, checkpointing, training, memory, topic classification, a ReAct-style agent wrapper, CPU optimization helpers, and practical multimodal prompt builders.

![NP-DNA architecture](assets/tantra-architecture.svg)

## Current Features

| Area | Status | Notes |
| --- | --- | --- |
| Dynamic tokenizer | Built | BPE-style tokenizer with byte fallback, Devanagari/Vedic base characters, save/load, and capacity growth. |
| Model core | Built | `NpDnaModel` and `NpDnaCore` wrap embeddings, mesh layers, cortex, checkpointing, and generation. |
| Genome | Built | Shared low-rank strand parameter generation via `Genome`. |
| Sparse mesh | Built | `NeuralMesh`, `CategoryMesh`, top-k routing, load-balancing loss, and strand usage tracking. |
| Strands | Built | SSM-style strands plus attention strand support. |
| Dynamic growth | Built | Runtime strand growth, layer addition, and embedding resize paths. |
| Memory cortex | Built | Dense in-process memory store with retrieval and augmentation. |
| Generation | Built | Standard and streaming generation with temperature, top-k, top-p, repetition penalty, token suppression, and prompt formatting. |
| Checkpointing | Built | Saves/loads model, tokenizer, metadata, cortex, component format, and sharded format. |
| Training | Built | `npdna/train_npdna_v3.py` local trainer with configurable staged data loading. |
| Agent wrapper | Built | `NpDnaAgent` with cortex search/store and safe expression tools. Network-backed tools should stay disabled or reviewed in production. |
| Classification | Built | `NpDnaTopicClassifier` and `tag_text` helpers. |
| Multimodal context | Practical bridge | Converts image/audio/JSON metadata into text context; not a learned multimodal embedding path yet. |
| CPU optimization | Experimental | Quantization, torch compile, thread tuning, parameter counting, and partial-freezing helpers. |

![Memory learning](assets/tantra-memory-learning.svg)

## Repository Structure

```text
Tantra LLM/
|-- npdna/
|   |-- __init__.py              # Public exports
|   |-- autonomy.py              # ReAct-style agent and safe math evaluator
|   |-- classifier.py            # Topic classifier helpers
|   |-- codecs.py                # Frozen codec registry
|   |-- config.py                # One seed config with dynamic expansion
|   |-- cortex.py                # Memory cortex and auto-store helpers
|   |-- generation.py            # Text generation and streaming generation
|   |-- genome.py                # Low-rank genome module
|   |-- mesh.py                  # Sparse mesh, category mesh, strands
|   |-- model.py                 # NpDnaModel, NpDnaCore, checkpointing
|   |-- multimodal_context.py    # Image/audio/structured prompt context bridge
|   |-- plasticity.py            # Growth metrics and autoscaling engine
|   |-- quant_turbo.py           # CPU quantization/optimization helpers
|   |-- tokenizer.py             # AtulyaTokenizer plus audio/vision feature shells
|   `-- train_npdna_v3.py        # Curriculum training script
|-- assets/
|   |-- tantra-architecture.svg
|   |-- tantra-banner.svg
|   `-- tantra-memory-learning.svg
|-- model/                       # Local generated model artifacts
|   |-- .gitkeep
|   `-- tokenizer/
|       |-- tokenizer_seed.json
|       |-- tokenizer_seed.pt
|       `-- vocab_samples.txt
|-- Download/                    # Local/private training data
|-- pyproject.toml
|-- requirements.txt
|-- SECURITY.md
|-- LICENSE
`-- README.md
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a fresh core:

```python
import torch
from npdna import NpDnaCore

core = NpDnaCore.from_config("seed")
ids = core.encode("Hello from NP-DNA")
logits, balance_loss = core.model(torch.tensor([ids]))
print(logits.shape, balance_loss.item())
```

Load a local checkpoint when available:

```python
from npdna import NpDnaCore

core = NpDnaCore.load("model/npdna_v3/best")
text = core.generate(
    "What is gravity?",
    max_tokens=100,
    temperature=0.35,
    top_k=30,
    top_p=0.85,
    repetition_penalty=1.2,
)
print(text)
```

## Training

The training script is intended for local/private training data. Keep datasets, checkpoints, logs, and generated archives out of Git unless they are explicitly sanitized and meant for release.

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python npdna\train_npdna_v3.py
```

Default script settings:

| Setting | Value |
| --- | --- |
| Config | `seed`, then automatic growth |
| Attention | Enabled |
| Batch size | 4 |
| Sequence length | 128 |
| Learning rate | `5e-3` |
| Checkpoints | Local `model/` output |
| Tokenizer assets | Local `model/tokenizer/` output |

## Configuration

NP-DNA exposes one public named configuration: `seed`. It starts small and expands automatically as vocabulary, strands, and layers need more capacity.

| Field | Seed start |
| --- | ---: |
| Hidden size | 64 |
| Layers | 5 |
| Initial vocab capacity | 4096 |
| Max vocab capacity | 256000 |

Use `NpDnaCore.from_config("seed")` for normal runs. Numeric complexity values are still supported internally for experiments, but the release path is seed-first auto-expansion.

## Seed Artifacts

The repository includes a small tokenizer/vocab seed pack under `model/tokenizer/`:

| File | Purpose |
| --- | --- |
| `model/tokenizer/tokenizer_seed.json` | Loadable tokenizer seed. |
| `model/tokenizer/tokenizer_seed.pt` | Compact torch metadata for vocab size, capacity, and merges. |
| `model/tokenizer/vocab_samples.txt` | Sanitized sample text used to shape the seed vocabulary. |

## Generation

```python
core.generate(
    prompt="Explain sparse routing in simple terms.",
    max_tokens=100,
    temperature=0.35,
    top_k=12,
    top_p=1.0,
    repetition_penalty=1.12,
)
```

Streaming is also available:

```python
for token in core.generate_stream("Write one sentence about memory.", max_tokens=40):
    print(token, end="", flush=True)
```

## Agent Use

```python
from npdna import NpDnaAgent, NpDnaCore

core = NpDnaCore.from_config("seed")
agent = NpDnaAgent(core)

agent.register_tool("echo", lambda text: text)
print(agent.run("Use math_eval to calculate sqrt(81)."))
```

For public or shared deployments, review every enabled agent tool before use. Keep network access, file writes, and code execution behind explicit policy controls.

## Multimodal Context Bridge

The multimodal helper is intentionally explicit: it turns file metadata or structured JSON into plain text context that the model can consume.

```python
from npdna.multimodal_context import build_multimodal_prompt

prompt = build_multimodal_prompt(
    "Summarize this input.",
    image="assets/tantra-architecture.svg",
    structured={"project": "NP-DNA", "mode": "local"},
)
print(prompt)
```

## Assets

Visual assets live in `assets/` and are referenced by this README:

| File | Purpose |
| --- | --- |
| `assets/tantra-banner.svg` | Header/banner image. |
| `assets/tantra-architecture.svg` | Architecture diagram. |
| `assets/tantra-memory-learning.svg` | Memory and learning diagram. |

## Security Notes

Before publishing changes, check for:

| Check | Current guidance |
| --- | --- |
| Secrets | Do not commit API keys, private tokens, credentials, or `.env` files. |
| Training data | Do not publish raw local datasets or dataset inventories unless they are intentionally public. |
| Model artifacts | Treat checkpoints, tokenizer dumps, logs, and generated archives as local artifacts by default. |
| Agent tools | Review any tool that can access the network, execute code, or write memory before enabling it outside local development. |

## Verification

Basic import check:

```powershell
python -c "from npdna import NpDnaCore; core = NpDnaCore.from_config('seed'); print(core.model.parameter_count())"
```

Run training:

```powershell
python npdna\train_npdna_v3.py
```

## License

MIT License. See [LICENSE](LICENSE).
