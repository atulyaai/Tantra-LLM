# Tantra-LLM v0.6-fusion_wiring

**Proprietary Multimodal Cognitive Architecture**

A complete rewrite transforming into a proprietary brain system using SpikingBrain-7B + Long-VITA + Whisper with dynamic context, fusion layers, hybrid memory, and adaptive personality.

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
| v0.7    | fusion_training          | Training loop operational                          |
| v0.8    | dynamic_compute          | Performance optimizations live                     |
| v0.9    | safety_personality       | Safety + personality modules operational           |
| v1.0    | stable_identity          | Stable reasoning identity with session persistence 🎯|

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

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from config import model_config, identity
from utils.model_loader import ModelLoader
from core.control.brain_orchestrator import BrainOrchestrator

# Initialize model loader
loader = ModelLoader(model_config.MODEL_CONFIG)

# Load encoders
spikingbrain = loader.load_spikingbrain()
whisper = loader.load_whisper()

# Build orchestrator (see demos/demo_minimal.py for full setup)
# brain = BrainOrchestrator(...)
```

### Demo

```bash
python demos/demo_minimal.py
```

## 🎯 Key Features

### Completed (v0.1–v0.6)

- v0.1: System identity
- v0.2: Dynamic context + flattened structure
- v0.3: Episodic memory influence
- v0.4: Semantic graph influence
- v0.5: Adaptive personality routing
- v0.6: Fusion gates + projector shape validation

### Next (→ v1.0)

- v0.7: Fusion training loop
- v0.8: Compute routing + KV reuse + streaming
- v0.9: Safety + values + style modules
- v1.0: E2E stability, persona consistency, final docs

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

**Current Version**: v0.6-fusion_wiring  
**Status**: Fusion stream wired and validated; next: fusion training (v0.7)  
**Capability**: Core architecture + hybrid memory + adaptive personality + validated fusion

