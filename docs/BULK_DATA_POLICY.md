# Bulk own-data policy (owner decision, 2026-10-19)

Tantra's conversation data is authored by this project, not imported from outside
datasets. To reach 10K–100K English conversations, large generated batches use a
sample-based review instead of row-by-row reading. This is an explicit owner decision;
it does not change the rules for Hindi/Hinglish or for persona data.

## What counts as reviewed for a bulk batch

1. **Automatic filters** (`python -m Tantra.data.bulk_review filter`): schema validation,
   persona rules (`Tantra/data/persona_rules.py`: no unasked "I'm an AI", no unasked
   creator/company mention, no service wording), English rows contain no non-Latin script,
   length limits (≤ 8 turns, ≤ 400 characters per turn), exact-duplicate removal.
   Every row must pass; failures are listed in `filter-report.json`.
2. **Owner spot-check** (`sample`, then `approve`): a seeded random sample (default 50)
   of the filtered rows is written to `spot-check.csv`. The owner marks every sampled row
   Y or N. The batch is approved only if at least 95% are Y; rows marked N are dropped.
   Below the threshold the generator must be fixed and a new batch drawn.
3. **Provenance on every row**: `review_method: owner_sample_spot_check`, sample size,
   approval rate, batch size, filter list and the SHA-256 of the marked sample. The reviewer
   string says "spot-checked N of M", so it is never mistaken for row-by-row review.

## What it does not cover

- Hindi and Hinglish: still need a named fluent reviewer per row (`fluent_language_review`).
  `bulk_review approve` refuses non-English rows.
- Persona and identity data (who Tantra is, creator Atul Vij, parent company Atulya AI):
  stays small and fully reviewed row by row.
- Family counts: generated conversations from one topic pack share a family, so 10K rows
  from 200 topics are 200 independent families, not 10K. Report both numbers.
- Release claims: a spot-checked bulk batch is training data for pilots; it is not an
  evaluation set and does not make a model production-ready.
