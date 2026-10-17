# English greeting pilot: 17 October 2026

Trained `runs/greeting-pilot-v1/greeting.pt` for 200 optimizer updates, rank 8,
on the preserved 28.86M base. No deployment or router update was performed.
The 303,104-parameter candidate remains paired with exact source base/tokenizer
hashes. Original weights were checked unchanged after training.

The original seed is tracked in `docs/data/greeting_seed.en.json`: 30 English
examples across ten scenario families, authored and content-reviewed by Codex AI.
This is not independent human review or fluent Hindi/Hinglish certification.
Frozen family splits contain 24 train, three validation and three test examples
(eight/one/one families). The manifest correctly remains `needs_data`; the release
minimum is still 1000/100/100 independent families.

An explicitly requested small training pilot uses `--small-pilot`; normal training
still rejects undersized data, and small pilots cannot request release validation.
This flag does not modify readiness, quality thresholds, routing or deployment.

```powershell
python -m Tantra.data.expert_workbench --out Datasets/workbench-greeting-pilot-v1 train --expert greeting --model-dir Model --output runs/greeting-pilot-v1 --steps 200 --rank 8 --threads 4 --small-pilot
```

To reconstruct the incoming seed in a NEW bundle, export the `records` array from
`docs/data/greeting_seed.en.json` to JSONL at `incoming/conversation/greeting.jsonl`,
then run the normal immutable workbench prepare command. Do not overwrite the
existing bundle or candidate.

Measured held-out assistant-token loss: base 5.59565, candidate 5.47050.
Training plus final validation took 39.38 seconds by the trainer's timer;
sampled peak RSS was 493,817,856 bytes. The timer excludes loading and baseline
validation; sampled RSS is not an OS-enforced peak limit.

Response inspection is saved in `runs/greeting-pilot-v1/responses.json`:

- `Hi!` -> `Hi! What can I help you with?`
- `Hi, just checking in.` -> `Hi!`
- An untimed hello incorrectly produced `Good morning!`.
- Two of three test prompts requesting no question still received a question.
- Hindi `नमस्ते` produced `on while!`; Hindi remains unsupported by this pilot.

No percentage release score is inferred from this tiny set. Test answers were
inspected once after training; do not tune this candidate against those test
examples. Expand reviewed conversation coverage in a new bundle and reserve a
new independent final test suite. Current candidate is not release-ready.

Validation after the change: 46 offline tests passed, one slow test deselected.
