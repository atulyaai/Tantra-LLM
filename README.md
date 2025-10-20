# 🧘 Tantra LLM - Your Own Progressive Mamba Model

A **CPU-first, dynamically-growing Mamba LLM** trained from scratch on **579K high-quality samples**. Auto-expands from 256×4 → 512×8 → 768×16 → 1024×24 layers with progressive training.

## 🚀 Quick Start (Server Recommended)

### 1. Upload to Server
```bash
scp -r "D:\Atulya Tantra LLM" user@your-server:/home/user/
```

### 2. Run Setup Script
```bash
cd Tantra-LLM
chmod +x setup_server.sh
./setup_server.sh
```

**Training Time**: ~5 hours on server (vs 67 hours on single CPU)

## 📊 What You Get

- **579,547 high-quality samples** (Alpaca, OpenAssistant, UltraChat, Dolly, WizardLM, SQuAD)
- **Progressive architecture**: Starts small, grows automatically
- **Your own model**: Trained from scratch, no pre-trained weights
- **Dynamic tokenizer**: Expands vocabulary with data
- **CPU-optimized**: Works on any server, GPU optional

## 🏗️ Architecture Growth

| Stage | Parameters | Layers | Seq Length | Time |
|-------|------------|--------|------------|------|
| 1 | 17M | 256×4 | 128 | ~2h |
| 2 | 68M | 512×8 | 256 | ~1.5h |
| 3 | 200M | 768×16 | 512 | ~1h |
| 4 | 500M | 1024×24 | 1024 | ~0.5h |

## 📁 Clean Project Structure

```
Tantra-LLM/
├── Training/
│   ├── training_main.py          # 🎯 Main progressive training
│   ├── download_datasets.py      # 📥 Download all datasets  
│   ├── combine_datasets.py       # 🔄 Combine datasets
│   ├── tokenizer_train.py        # 🔤 Train tokenizer
│   ├── model_mamba.py            # 🧠 Mamba model definition
│   └── [core files]
├── Dataset/
│   ├── combined_full_training.jsonl  # 📚 579K samples
│   └── [individual datasets]
├── Model/
│   ├── tokenizer.json            # 🔤 Trained tokenizer
│   └── tantra_weights.safetensors # 🧠 Final model
└── Config/                       # ⚙️ Configuration files
```

## 🎯 Key Features

- ✅ **Your own model** - No pre-trained weights, trained from scratch
- ✅ **Auto-expanding** - Architecture grows during training
- ✅ **High-quality data** - 579K curated samples (no C4/Pile)
- ✅ **Server-optimized** - Multi-core, large batch sizes
- ✅ **Progressive training** - 4 stages with automatic transitions
- ✅ **Clean codebase** - Removed duplicates, organized files

## 🔧 Manual Setup

```bash
# 1. Install dependencies
pip install torch transformers tokenizers datasets safetensors tqdm

# 2. Download datasets
python Training/download_datasets.py

# 3. Combine datasets  
python Training/combine_datasets.py

# 4. Train tokenizer
python Training/tokenizer_train.py --input_glob "Dataset/combined_full_training.jsonl" --out Model/tokenizer.json

# 5. Start training
python Training/training_main.py
```

## 📈 Expected Results

- **Stage 1**: Basic language understanding
- **Stage 2**: Better conversation flow  
- **Stage 3**: Complex reasoning
- **Stage 4**: Advanced capabilities

## 🖥️ Server Requirements

- **Minimum**: 8 cores, 32GB RAM
- **Recommended**: 16 cores, 64GB RAM  
- **Optimal**: 32 cores, 128GB RAM
- **GPU**: Optional (100x speedup)

## 📚 Datasets Included

- **Alpaca** (52K): Instruction-following
- **OpenAssistant** (82K): Conversations
- **UltraChat** (200K): High-quality chats
- **Dolly** (15K): Instructions
- **WizardLM** (143K): Complex instructions
- **SQuAD** (87K): Reading comprehension

**Total**: 579,547 samples, ~110MB

## 🎉 After Training

```bash
# Test your model
python Training/mamba_runtime.py

# Start API server
python Training/serve_api.py
```

## 📖 Documentation

- `SERVER_DEPLOYMENT.md` - Detailed server setup
- `Docs/DOCS.md` - Complete documentation
- `Config/` - All configuration files

---

**Ready to train your own Mamba LLM? Upload to server and run `./setup_server.sh`!** 🚀