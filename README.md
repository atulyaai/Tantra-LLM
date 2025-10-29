# Tantra-LLM

**Proprietary Multimodal Cognitive Architecture with SpikingBrain-7B**

A complete multimodal AI system using SpikingBrain-7B + Long-VITA + Whisper with dynamic context, fusion layers, hybrid memory, and adaptive personality.

## 🧠 Architecture Overview

### Core Components

- **SpikingBrain-7B**: Reasoning core
- **Long-VITA/EVA-ViT**: Vision encoder
- **Whisper Large-v3**: Audio encoder
- **Coqui TTS**: Speech generation
- **Proprietary Fusion Layers**: Vision+audio → language alignment (~5M params)
- **Hybrid Memory**: Working memory + episodic vector + semantic graph
- **Adaptive Personality**: Mode-aware behavior with user overrides

### Version Roadmap

Versions reflect **capability milestones**, not calendar dates:

| Version | Codename                 | Capability                                         |
|---------|--------------------------|----------------------------------------------------|
| v0.1    | origins                  | Brain boots, core IO routing operational ✅        |
| v0.2    | eyes_open                | Dynamic architecture + flattened structure ✅      |
| v0.3    | remembers                | Episodic memory influences output ✅               |
| v0.4    | understands_relations    | Semantic graph influences responses ✅              |
| v0.5    | self_shaping             | Adaptive personality routing online ✅             |
| v0.6    | fusion_wiring            | Fusion gates + projector shapes validated ✅        |
| v0.7    | fusion_production        | Real encoders integrated (Whisper, CLIP) ✅         |
| v0.8    | inference_live           | SpikingBrain forward pass + embeddings injection ✅  |
| v1.0    | stable_identity          | **Production system + one-click API ✅** 🎯        |

See CHANGELOG for detailed file changes.

## 📁 Project Structure

```
config/                # System identity, model, memory configs
core/
  fusion/             # Vision/audio projection layers
  memory/             # Working/episodic/semantic memory
  control/            # Perception, decision, response, orchestrator
  models/             # Dynamic context, token streaming
encoders/              # Vision/audio/text encoders
personality/           # Adaptive personality layer
training/              # Fusion layer training pipeline
utils/                 # Model loader, device manager
demos/                 # Example scripts

core/                  # Legacy OCR-native system (reference)
```

## 🚀 Quick Start

### One-Click Start (Windows)

```cmd
scripts\start_api.bat
```

Server starts on `http://localhost:8000`.

### Manual Start

```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Set env
$env:PYTHONPATH="D:\Atulya\Tantra-LLM"
$env:TANTRA_LV_DIR="D:\models\longvita-16k"  # Optional: Long-VITA path
$env:TANTRA_SPB="microsoft/DialoGPT-medium"   # Your model name

# Start API
uvicorn demos.api_server:app --host 0.0.0.0 --port 8000
```

### Python API Usage

```python
import requests

# Text-only
response = requests.post("http://localhost:8000/generate", files={"text": "Hello"})
print(response.json())

# With image
with open("image.jpg", "rb") as f:
    response = requests.post("http://localhost:8000/generate", 
        files={"text": "Describe this", "image": f})
```

### Model Status (4 Required Models)

| Model | Status | Type | Memory | Notes |
|-------|--------|------|--------|-------|
| SpikingBrain-7B | ✅ Required | Custom LLM (~7B) | ~14GB | Full 7B model, reasoning engine |
| Long-VITA | ✅ Required | Vision & Text | ~1-2GB | Image/video understanding |
| Whisper | ✅ Required | Audio STT (base) | ~150MB | Speech-to-text |
| Coqui TTS | ✅ Required | Audio TTS | ~500MB | Speech generation |

**💡 No Fallbacks**: Only these 4 models are used. All are required.

## 🎯 Key Features

### Completed (v0.1–v1.0)

- v0.1: System identity
- v0.2: Dynamic context + flattened structure
- v0.3: Episodic memory influence
- v0.4: Semantic graph influence
- v0.5: Adaptive personality routing
- v0.6: Fusion gates + projector shape validation
- v0.7: Real encoder integration
- v0.8: SpikingBrain forward pass with embeddings injection
- **v1.0**: Complete production system with one-click start

### Future Enhancements

- Long-VITA full local forward (requires GPU)
- Coqui TTS when MSVC tools available
- Advanced KV-cache and streaming optimizations
- Database backends (ChromaDB/FAISS/Neo4j)

## 🔬 Design Philosophy

### Proprietary Assets

1. **Fusion projection weights** (~5M params) - Owned, trainable, small
2. **Adapter parameters** - Personality/style injection
3. **Memory routing logic** - Control loop decisions
4. **Identity configuration** - Behavioral rules
5. **Training prompts** - Fusion layer curricula

All stored separately from base models, easily versioned and protected.

### Semantic Commits

All commits follow `<type>(<scope>): <message>` format:

- `feat(core):` - New capabilities
- `improve(control):` - Performance/reasoning improvements
- `fix(memory):` - Bug fixes
- `refactor(utils):` - Structure/cleanup
- `config(system):` - Build/system changes

## 📖 Documentation

- CHANGELOG - versioned changes and file lists
- Phase 1 Notes - `config/identity.py`
- Phase 2 Stubs - `core/models/dynamic_context.py`

## 🔒 License

Proprietary - See [LICENSE](LICENSE) for details.

Copyright (c) 2024 Tantra-LLM Project.
All rights reserved.

## 🤝 Contributing

This is a private project. Development follows strict semantic commits, branch protection, and capability-based versioning.

---

## 🧪 Testing

See [TEST_INSTRUCTIONS.md](TEST_INSTRUCTIONS.md) for detailed testing instructions.

Quick test:
```powershell
# Create fresh venv if needed
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Test model
$env:TANTRA_LOW_RESOURCE="0"
python -c "from utils.model_loader import ModelLoader; from config import model_config; m = ModelLoader(model_config.MODEL_CONFIG).load_spikingbrain(); print('Loaded:', 'model' in m)"
```

## 📝 Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

**Current Version**: v1.4-original-specs  
**Status**: Using original model specifications  
**Changes**: SpikingBrain-7B (full), Whisper base, Long-VITA, Coqui TTS  
**Next**: Test SpikingBrain-7B → Add Long-VITA → Vision conversations

