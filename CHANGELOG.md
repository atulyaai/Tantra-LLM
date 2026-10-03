# Changelog — Tantra-LLM

All notable changes to Tantra-LLM are documented here.
Follows [Semantic Versioning](https://semver.org/) and [Keep a Changelog](https://keepachangelog.com/).

---

## [Unreleased]
- Quantized INT4 inference for Raspberry Pi
- Hindi instruction-tuning dataset (10k examples)
- REST API server (`python serve.py`)

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
