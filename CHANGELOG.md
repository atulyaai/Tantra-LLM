# Changelog — Tantra-LLM

All notable changes to Tantra-LLM are documented here.
Follows [Semantic Versioning](https://semver.org/) and [Keep a Changelog](https://keepachangelog.com/).

---

## [Unreleased]
### Fixed
- `main.py` no longer moves `ROADMAP.md`, `CHANGELOG.md` and `pyproject.toml` into `_old_code/` on the first run
- README test badge now shows 59 passing tests

### Added
- `--mode export --int4`: 4-bit group-scaled (64) weight storage, about 2.4x smaller than fp16 on the moe preset; embeddings, norms, router and MTP head stay fp16; `load_model` reads it transparently
- MoE: per-layer expert usage (dead experts, busiest-expert load) logged during training and written to `training_status.json`
- MoE: optional router z-loss (`moe.router_z_coeff`, 1e-3 in the moe preset)

### Notes
- OpenAI-compatible API + WebUI is served by `python main.py --mode serve`
- Config lives in `Tantra/config.py` (presets: tiny, small, moe, billion)

### Planned
- Native INT4 compute kernels for Raspberry Pi
- Hindi instruction-tuning dataset (10k examples)

---

## [v0.3.0] — 2026-09-15
### Added
- Hybrid ALRA (Adaptive Local-Relative Attention) architecture
- Sliding-window attention for long sequences (4096 context)
- Byte-level BPE tokenizer with 64k vocabulary (Hindi + English)
- CPU-first training loop with gradient checkpointing
- WebUI chat interface (`python webui.py`)
- `--mode train` and `--mode infer` CLI flags

### Changed
- Switched from character-level to BPE tokenization (3× faster training)
- Model config moved to `config.yaml` (was hardcoded)

### Fixed
- OOM crash on Windows during long training runs
- Hindi tokenization edge cases with conjunct consonants

---

## [v0.2.0] — 2026-08-01
### Added
- Initial Hindi + English dual-language training pipeline
- Custom PyTorch `Trainer` class with checkpoint resume
- `Tests/` suite with pytest (`python -m pytest Tests -q`)

---

## [v0.1.0] — 2026-07-01
### Added
- Initial commit: from-scratch transformer architecture
- Basic tokenizer and dataset loader
- README and project structure
