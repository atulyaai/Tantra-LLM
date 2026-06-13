# NP-DNA — NeuroPlastic DNA Network

CPU-first language model with auto-scaling vocabulary, sparse mixture-of-strands routing, and curriculum learning.

## Structure

```
NP-DNA/
├── npdna/                  # Model code (14 files)
│   ├── config.py           # Architecture configs (seed/nano/micro)
│   ├── model.py            # NpDnaModel + NpDnaCore + checkpoint
│   ├── tokenizer.py        # BPE tokenizer + audio/vision encoders
│   ├── mesh.py             # NeuralMesh + CategoryMesh + Strand
│   ├── genome.py           # Low-rank DNA weight generation
│   ├── generation.py       # Text generation mixin
│   ├── cortex.py           # Memory cortex
│   ├── plasticity.py       # Auto-scaling engine
│   ├── autonomy.py         # Agent + safe expression evaluator
│   ├── classifier.py       # Topic classifier
│   ├── codecs.py           # Frozen codec registry
│   ├── multimodal_context.py
│   ├── quant_turbo.py      # Quantization
│   ├── assets/             # Trained tokenizer files
│   └── train_npdna_v3.py   # Training script
├── model/                  # Checkpoints (created during training)
├── Download/               # 20 GB training data (8 curriculum folders)
├── pyproject.toml
└── requirements.txt
```

## Train

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python npdna\train_npdna_v3.py
```

Curriculum: samples → agentic → factual → code → reasoning → translation → general → math (~9100 steps, ~2h on CPU)

## Configs

| Name | d_model | Params | Vocab |
|------|---------|--------|-------|
| seed | 64 | 3.2M | 4K |
| nano | 128 | 7.3M | 8K+ |
| micro | 256 | 28.5M | 16K+ |

## License

MIT
