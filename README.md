# Tantra-LLM v0.3-remembers

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
| v0.5    | self_shaping             | Adaptive personality fully online                  |
| v0.6    | fusion_wiring            | Fusion layer pipeline complete                     |
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

### Phase 1 (v0.1-origins)

- ✅ System identity configuration with behavioral profiles
- ✅ Mixed model loading (local SpikingBrain/Whisper + remote Long-VITA)
- ✅ Project scaffolding with proper imports
- ✅ Git workflow with semantic commits

### Phase 2 (v0.2-eyes_open)

- ✅ Dynamic context window (short vs long)
- ✅ Flattened repo structure and updated docs

### Phase 3 (v0.3-remembers)

- ✅ Episodic memory retrieval influences response generation
- ✅ Smoke tests for demo wiring

### Planned

- Phases 4-6: Fusion training, semantic memory, control loop, performance optimization

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

- CHANGELOG - See versioned changes and file lists
- Phase 1 Notes - See `config/identity.py`
- Phase 2 Stubs - See `core/models/dynamic_context.py`

## 🔒 License

Proprietary - See [LICENSE](LICENSE) for details.

Copyright (c) 2024 Tantra-LLM Project.
All rights reserved.

## 🤝 Contributing

This is a private project. Development follows strict semantic commits, branch protection, and capability-based versioning.

---

**Current Version**: v0.4-understands_relations  
**Status**: Semantic + episodic memory integrated; next: personality routing (v0.5)  
**Capability**: Core architecture + dynamic context + hybrid memory active

