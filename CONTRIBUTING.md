# 🤝 Contributing to Tantra-LLM

Thank you for your interest in contributing to **Tantra-LLM**! We welcome contributions from AI researchers, systems engineers, and open-source developers to advance open, efficient, local-compute foundation models.

---

## 🛠️ 1. Development Setup

```powershell
# Clone the repository
git clone https://github.com/atulyaai/Tantra-LLM.git
cd Tantra-LLM

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies in editable mode
pip install -r requirements.txt

# Run the automated test suite to verify setup
pytest Tests/ -q
```

---

## 🏛️ 2. Architectural Guidelines & Code Organization

* **`Tantra/`**: All reusable core neural network code, training engines, tokenizers, and evolution controllers live here.
* **`Tests/`**: Automated unit and regression tests categorized into 4 suites:
  * `Tests/test_core_architecture.py` (Architecture, Attention, BitNet 1.58b, MTP)
  * `Tests/test_training_alignment.py` (Continuous Packing, DataLoaders, DPO Preference)
  * `Tests/test_omnimodal_tools.py` (Vision, Audio, Video, Tool Calling)
  * `Tests/test_system_integration.py` (Robustness, Red-Teaming, WebUI Server API)
* **`Datasets/`**: 4-Track Domain Curriculum (`expert_conversation.jsonl`, `expert_code.jsonl`, `expert_math_science.jsonl`, `expert_general.jsonl`).
* **`main.py`**: Unified entry point for all modes (`train`, `chat`, `benchmark`, `export`, `auto-pilot`).

---

## 📋 3. Pre-Flight PR Checklist

Before submitting a Pull Request, ensure:

1. **All Tests Pass**: Run `pytest Tests/ -q` and ensure **100% of tests pass**.
2. **Deterministic & Safe**: Code should execute cleanly without NaN/Inf leaks.
3. **No Large Binary Checkpoints**: Checkpoint `.pt` weights and temporary data are gitignored.
4. **Documentation**: Update docstrings and documentation if modifying public APIs or CLI arguments.

---

## 📜 4. License & Contributor Agreement

By contributing to Tantra-LLM, you agree that your contributions will be licensed under the project's **[MIT License](LICENSE)**.
