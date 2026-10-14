# Tantra-LLM

A CPU-first sparse Mixture-of-Adapters language model. One small shared base transformer, many tiny LoRA experts, and a cheap router that picks which experts each request needs.

**Current model (stage 1)**: 4.7M base + 11 conversation experts, ~29 tokens/sec on a laptop CPU.

---

## How it works

```
   user message
        │
        ▼
┌─────────────────┐   bag-of-embeddings classifier, one tiny matmul
│     Router      │   → picks 1-4 experts (adaptive: "hi" = 1, "hi, who are you?" = 2)
└────────┬────────┘
         │  active = [("greeting", 0.7), ("identity_self", 0.3)]
         ▼
┌─────────────────────────────────────────────────────┐
│           shared base transformer                   │
│   RoPE · grouped-query attention · SwiGLU · RMSNorm │
│   tied embeddings · KV-cached generation            │
│                                                     │
│   every Linear layer can carry named LoRA experts:  │
│                                                     │
│   y = base(x) + Σ  (x @ Aₙ @ Bₙ) · scale · wₙ     │
│                     └ only ACTIVE experts run ──┘   │
└─────────────────────────────────────────────────────┘
```

Design points:

- **Sparse, not dense.** Only the selected experts run, so adding expert #1000 costs disk space, not CPU time.
- **One file per expert.** `Model/experts/greeting.pt` can be trained, replaced or deleted without touching anything else.
- **Adaptive routing.** The router decides *how many* experts a message needs (up to 4), based on cumulative probability.
- **Function-preserving growth.** `grow_depth()` inserts layers with zeroed output projections — output is bit-identical, training continues from there: tiny → small → base → 1B on the same laptop.
- **CPU-first.** INT8 dynamic quantization of base matmuls (~4× smaller); experts stay fp32 because they're tiny.

---

## Training pipeline

| Stage | What | Command |
|-------|------|---------|
| 0 | Seed curated data + general corpus | `python -m Tantra seed` |
| 1 | Train BPE tokenizer (8k vocab) | `python -m Tantra tokenizer` |
| 2 | Train base on **general text** | `python -m Tantra base --preset tiny --steps 1500` |
| 3 | Train each expert alone (frozen base) | `python -m Tantra expert greeting identity_self` |
| 4 | Train router (message → experts) | `python -m Tantra router` |
| 5 | Grow, then continue | `python -m Tantra grow --layers 2` |

Everything at once:

```bash
python -m Tantra stage1 --preset tiny
```

**Base vs expert data.** The base trains on the general corpus
(`Datasets/general/`) **plus all the chat data**, which is what keeps
conversation fluent (a plain-text-only base writes gibberish in chat —
measured). Each expert then refines on its own full dataset while everything
else is frozen, sharpening its domain style when the router activates it.

> Note on the held-out metric: the stage-1 data is template-generated and very
> homogeneous, so the base already predicts any held-out slice at low loss.
> Held-out loss therefore stays roughly flat across expert training — it is
> not a meaningful quality signal at this scale. What matters is the generated
> replies and routing, both of which are verified end-to-end.

**Router learns mixed messages.** During router training, user texts from two
different experts are concatenated and labeled with both, so `"hi, who made
you?"` activates *greeting* and *identity_* experts together instead of
picking one.

Each expert trains in isolation — training `greeting` never touches
`identity_self`, so experts can be retrained one at a time.

**Checkpoints.** `stage1` snapshots the full model state (tokenizer + base +
router + all experts) to `checkpoints/stage1_latest/` when it finishes. Save a
named snapshot anytime with:

```bash
python -m Tantra checkpoint --tag before_growth
```

---

## Inference

```bash
python -m Tantra ask "who are you?"        # single question
python -m Tantra chat                      # interactive, shows router weights
python -m Tantra chat --int8               # INT8 quantized (less RAM, faster)
python -m Tantra bench                     # real tokens/sec on your CPU
```

Example:

```
You: hi, who made you?
Tantra: I'm Tantra, an AI assistant made by Atulya AI.
   [greeting:0.45, identity_self:0.55]
```

---

## Model sizes

| Preset | Base params | d_model | Layers | Use |
|--------|------------|---------|--------|-----|
| `tiny` | ~6.5M | 256 | 6 | smoke tests, current stage |
| `small` | ~32M | 512 | 8 | first real stage |
| `base` | ~213M | 1024 | 16 | mid scale |
| `1b` | ~1012M | 2048 | 21 | target size |
| `7b` | ~6624M | 4096 | 36 | beyond |

Scaling path: train at `tiny`, finish the stage, `grow_depth()`, continue at the next size. The same checkpoints and expert files keep working.

---

## Stage-1 experts (11)

| Expert | Data |
|--------|------|
| greeting | hellos, first contact |
| farewell | goodbyes, closing |
| smalltalk | casual chat |
| empathy | emotional support |
| apology | sorry / recovery |
| clarification | "what do you mean?" |
| identity_self | who am I |
| identity_capabilities | what can I do |
| identity_limitations | what I can't do |
| humor | jokes, lightening the mood |
| safety_refusal | refusing harmful requests |

Taxonomy for the full 164-expert / 13-domain design lives in `Tantra/config/taxonomy.yaml`.

---

## Data

```
Datasets/curated/<domain>/<expert>.jsonl   # specialist chat data (experts only)
Datasets/general/general.jsonl             # plain English text (base only)
```

Curated rows use the chat format:

```json
{"messages": [{"role": "user", "content": "hello"},
              {"role": "assistant", "content": "Hi! What can I do for you?"}]}
```

General rows are plain text (no roles):

```json
{"text": "Water expands when it freezes, which is why ice floats on a lake."}
```

- 1,500 curated rows per stage-1 expert; ~6,000 general rows for base training
- Junk filter drops template filler (`"Sample 12 of 10000"`) and exact duplicates at load time
- Legacy `{"text": "<|user|>...<|assistant|>..."}` rows are converted automatically

---

## Project layout

```
Tantra/
├── main.py                  # CLI (python -m Tantra ...)
├── tokenizer.py             # byte-level BPE, 8k vocab, no <unk>
├── config/                  # size presets, taxonomy
│   └── taxonomy.yaml        # 164 experts / 13 domains (full design)
├── model/tantra_model.py    # base transformer, LoRA experts, router, growth
├── training/tantra_trainer.py  # 5-stage pipeline
├── data/
│   ├── expert_dataset.py    # jsonl loading, junk filter, encoding
│   ├── seed_conversation.py # curated chat data generator (experts)
│   └── seed_general.py      # plain-text corpus generator (base)
├── inference/               # KV-cached engine, INT8, benchmark
└── utils/                   # health monitor, expert lifecycle
Datasets/curated/            # specialist chat data (11 experts)
Datasets/general/            # plain English text for base training
Model/
├── tokenizer.json
├── base.pt                  # base + router
└── experts/<name>.pt        # one file per expert
checkpoints/                 # saved snapshots (stage1_latest, custom tags)
Model_v1/                    # backup of first trained model
tests/test_tantra.py         # 14 tests (pytest)
```

---

## Tests

```bash
python -m pytest tests/ -v
```

14 tests: forward/loss shapes, expert isolation, KV-cache equivalence, function-preserving growth, router add/remove, save/load roundtrip, INT8, adaptive routing count, junk filtering, end-to-end tiny pipeline.

---

## Known limitations (tiny stage)

- **Template data caps quality.** Stage-1 data is template-composed English
  (recombined intents/phrasings). The base reaches loss ~0.3 and replies are
  fluent, but knowledge and open-ended ability are limited by the data, not
  the architecture. Real scraped text is needed before the `base` preset.
- **Mixed prompts**: the router mean-pools token embeddings, so in a mixed
  message like `"hi, who made you?"` the second intent is diluted — it now
  activates two experts (progress over one), but the pair isn't always the
  right one. Attention pooling or a deeper router fixes this at higher presets.
- **INT8 is slower at tiny size** (30.8 vs 55.1 tok/s measured): dynamic
  quantization overhead outweighs the smaller matmuls when the base is only
  4.7M params. INT8 pays off from the `small` preset up.

---

## Requirements

- Python 3.12, PyTorch (CPU), `tokenizers`, `pyyaml`
- ~450 MB RAM to train the tiny stage; no GPU needed

```bash
pip install -r requirements.txt
```
