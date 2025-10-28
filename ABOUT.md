# About Tantra-LLM

**🧠 Proprietary Multimodal Cognitive Architecture**

Tantra-LLM is a complete brain system integrating:

- **SpikingBrain-7B** - Reasoning core
- **Long-VITA/EVA-ViT** - Vision encoding
- **Whisper Large-v3** - Audio encoding
- **Coqui TTS** - Speech generation

## 🎯 Core Innovation

Unlike simple chaining of models, Tantra uses:

1. **Proprietary Fusion Layers** (~5M params) - Vision/Audio → Language alignment
2. **Hybrid Memory System** - Working memory + episodic vector + semantic graph
3. **Adaptive Personality** - Mode-aware behavior with automatic/user overrides
4. **Dynamic Context** - Short (fast) vs Long (deep reasoning) routing

## 🧬 Architecture Philosophy

**Not a chatbot. A cognitive mind.**

- Proprietary fusion projections define the "soul"
- Hot-swappable personality modules (<50M params total)
- Memory that grows and consolidates over time
- Reasoning style that adapts to task complexity

## 📊 Version Roadmap

Versions reflect **capability milestones**, not calendar dates:

| Version | Codename | Capability |
|---------|----------|------------|
| v0.10 / v0.1-origins | ✅ Current | Core architecture operational |
| v0.2 / v0.2-eyes_open | ✅ Current | Dynamic architecture + flattened structure |
| v0.3-remembers | 🚧 Planning | Episodic memory influences output |
| v0.4-understands_relations | 🚧 Planning | Semantic graph reasoning live |
| v0.5-self_shaping | 🚧 Planning | Adaptive personality fully online |
| v1.0-stable_identity | 🎯 Target | Stable reasoning identity with persistence |

## 🔒 Proprietary Assets

The following components are 100% owned and defined:

- Fusion projection weights (~5M params)
- Adapter parameters (personality/style)
- Memory routing logic (control loop)
- Identity configuration (behavioral rules)
- Training prompts for fusion layers

Base models (SpikingBrain, Long-VITA, Whisper) remain under their original licenses.

## 📁 Project Structure

```
config/          # System identity, model, memory configs
core/
  fusion/        # Vision/audio projection layers
  memory/        # Working/episodic/semantic memory
  control/       # Perception, decision, response, orchestrator
  models/        # Dynamic context, token streaming
encoders/        # Vision/audio/text encoders
personality/     # Adaptive personality layer
training/        # Fusion layer training pipeline
utils/           # Model loader, device manager
demos/           # Example scripts
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python demos/demo_minimal.py
```

```python
from config import model_config, identity
from utils.model_loader import ModelLoader

# Initialize model loader
loader = ModelLoader(model_config.MODEL_CONFIG)

# Load encoders
spikingbrain = loader.load_spikingbrain()
whisper = loader.load_whisper()
```

## 🎓 Design Principles

- **Small proprietary surface** - Only fusion layers are trainable (<50M params)
- **Dynamic context** - Adapt context length based on urgency/complexity
- **Memory that evolves** - Consolidates similar memories over time
- **Personality that switches** - Auto-detects mode or accepts user override
- **Git workflow** - Semantic commits + capability-based versioning

## 🤝 Contributing

This is a private project. Development follows:

- Semantic commit format: `<type>(<scope>): <message>`
- Branch strategy: `main`, `develop`, `feature/*`, `research/*`
- Version tags tied to capability milestones

## 📄 License

See [LICENSE](LICENSE) for details. Proprietary components are separately licensed.

---

**Current Version**: v0.2-eyes_open  
**Status**: Dynamic architecture complete, fusion training next  
**Repository**: [github.com/atulyaai/Tantra-LLM](https://github.com/atulyaai/Tantra-LLM)

