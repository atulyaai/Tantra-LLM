# Changelog

All notable changes are tracked here. The repository uses a flat 12-file layout
under `npdna/` (no sub-packages). Large binaries and datasets are gitignored;
checkpoints live in `model/` and datasets in `Download/` (runtime-local only).

## v1.0.0 — Flat NP-DNA (NeuroPlastic DNA) baseline
- **Core**: from-scratch neuroplastic transformer `NpDnaModel` (embedding → Mesh
  strands via shared `Genome` hypernetwork → final norm → LM head). Strands are
  small learnable seeds expanded into real weight matrices, so each checkpoint
  stores a compact genome rather than dense per-strand matrices.
- **Dynamic growth**: vocab, strand count, and layer count auto-scale during
  training; LoRA adapters (`nn.Linear` low-rank) for fine-tuning.
- **Multimodal fusion (activation-level)**: built-in single `vision_projector` /
  `audio_projector` as `nn.Linear(4096, H)` in `model.py`; projectors are the
  only multimodal weights (no separate standalone MLP trainer).
- **Layout**: single package `npdna/` (12 .py + `personality_config.json`),
  `tests/` (pytest), `tools/` (standalone one-shot scripts).
- **Entry points** registered in `pyproject.toml`: `npdna-chat`, `npdna-train`,
  `npdna-info`, `npdna-benchmark`, `npdna-cpu-benchmark`, `npdna-release`,
  `npdna-serve`, `npdna-studio`.
- **Checkpoint slots** (flat): `model/best`, `model/latest` (with rolling
  `latest.1`/`.2`/`.3` backups), `model/final`, `model/step_N`.

### Fixes since baseline
- Checkpoint slot layout flattened: slots now live at `model/<name>` rather than
  nested under `model/latest/<name>`.
- Stripped legacy 512/128-dim `vision_projector`/`audio_projector` weights from
  the bundled checkpoint so they no longer trigger size-mismatch warnings on
  resume (projectors are re-initialized to the correct 4096-dim shape).
- Hardened `NpDnaCore.load`: shape-mismatched checkpoint keys now raise a clear
  `RuntimeError` (set `NPDNA_REPAIR=1` to restore the old strip-and-warn recovery
  path for rebuilding a checkpoint).
- `--distill` (GPT-2 teacher) no longer crashes training on network/Hub download
  failures — it warns and continues without distillation.
- Rolling-latest backup rotation is best-effort on Windows (retries + non-fatal
  fallback) instead of crashing the run on a transient file lock.
- Repaired UTF-8 mojibake (cp1252 round-trip) in `serving.py` and
  `architecture.py` section dividers and emoji.

## Notable removals
- `FusionTrainer` / `FusionProjector` / `FusionTrainingConfig` classes —
  projectors are built-in to the text model and trained via `--fusion`.
- `train_fusion.py` / `generate_training_data.py` standalone scripts — folded
  into the single `npdna/train.py` entrypoint.
- `final_projectors.pt` orphaned artifact from the old standalone-projector era.
