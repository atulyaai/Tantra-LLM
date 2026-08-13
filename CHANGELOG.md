# Changelog

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Changed

- Replaced overstated architecture and roadmap documentation with the current
  CPU-first implementation and explicit experimental boundaries.
- Consolidated planning into `ROADMAP.md`; removed the duplicate `TASKS.md`.
- Documented `Tantra/` as reusable package code with integrated offline
  utilities.
- Moved the FastAPI backend into `webui/`; `Tantra/` no longer contains WebUI
  implementation code.

### Fixed

- CPU checkpoints now store their architecture configuration, allowing loaders
  to rebuild the compatible model instead of answering with randomly initialised
  weights after a shape mismatch.
- The maintained CPU trainer saves only `Latest`, avoiding redundant best and
  per-step checkpoint archives.

### Added

- `Tantra.cpu_cli` for loading and training the active CPU profile explicitly.

## 1.0.0 - 2026-08-07

Initial public project structure, NeuroCore research components, local server,
dataset pipeline, and automated tests.
