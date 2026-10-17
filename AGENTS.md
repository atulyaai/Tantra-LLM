# Repository instructions

- Read `docs/HANDOFF.md`, `docs/DATASET_SPEC.md`, and the relevant expert card before dataset/training changes.
- Preserve existing model, tokenizer, expert weights, datasets and snapshots. Export candidates to a separate directory.
- All 164 taxonomy IDs are specifications, not proof of trained or validated capabilities.
- Reject filler data and unreviewed provenance. Do not manufacture native-language examples, citations, review or evaluation scores.
- Keep expert scopes distinct. Reuse the shared workbench/validators; do not copy pipelines or the same dataset into overlapping experts.
- Use frozen train/validation/test splits. Group paraphrases and translations by source/problem family. Never train on test data.
- Start at rank 8 and train one candidate adapter at a time. Require independent held-out gains before deployment.
- Keep claims aligned with the current code: legacy inference eagerly loads adapters; expert optimizer resume and automatic repair/growth are absent. The separate public base pilot has explicit optimizer/RNG/cursor resume.
- Work offline by default for tests; no paid compute or paid generation is authorized.
- Run `python -m pytest tests -m "not slow" -q` and relevant new tests. Do not run training scripts that overwrite `Model/` as a test.
- Update README/status/handoff when behavior or measured capability changes. Do not label bootstrap examples production-ready.
- Read Tantra/config/training_scope.json and docs/CONVERSATION_UPDATES.md before language/tokenizer changes. Preserve balanced Hindi/English budgets; report Hinglish independently. The isolated 64K model has been expanded and started training; read docs/VOCABULARY_MIGRATION.md. Deployment and adapter compatibility still require evaluation.
