# Handoff for developers and coding models

## Read first

1. `README.md`: honest architecture, current limitations and commands.
2. `docs/EXPERT_STATUS.md`: generated inventory of all 164 taxonomy IDs.
3. `docs/DATASET_SPEC.md`: strict dataset and non-overlap contract.
4. `docs/expert_catalog.json` and the relevant `docs/experts/<domain>/<expert>.md`.
5. Local `Datasets/workbench/audit.json`, `prepared/manifest.json`, and
   `training_plan.json` when present. Large local datasets are not in Git.

## Current state

The restored code started at upstream commit `7daf618`. Local `Model/` contains
20 adapter files: 19 taxonomy experts and supplementary `hindi_grammar`, which is
not one of the 164 IDs. `gk_indic` is also supplementary local data. Preserve them;
do not silently rename or delete assets to make counts look tidy.

The base is 28,863,488 parameters with a 10,277-token byte-level tokenizer.
The tested local neural outputs are often incorrect or garbled. Calculator and
knowledge-store successes must not be reported as neural model accuracy.
Most old raw expert files are filler, and some use unsupported bracket-language
schemas. Read the audit before choosing a dataset. Never train the old corpus
merely because there is a file for an expert.

## Work queue

1. Start with greetings, identity, clarification, empathy, capabilities and refusal
   in Hindi/Hinglish/English. Existing examples need provenance and careful review.
2. Use generated arithmetic/JSON/CSV records as reproducible pipeline bootstrap
   tests. Expand real task patterns, edge cases and language coverage; do not
   inflate the row count by template noise.
3. Add reviewed datasets for one distinct expert at a time, using its card.
4. Prepare a NEW immutable dataset bundle after changing incoming data.
5. Train candidates against a frozen base; evaluate answers and regression, then
   integrate the adapter and retrain/evaluate routing in a separate release directory.
6. Expand beyond conversation only when the specialist beats existing experts
   on its own held-out task. Keep unsupported or redundant specialists disabled.

## Commands (from repo root)

```powershell
python -m pip install -r requirements.txt
python -m Tantra.data.expert_workbench init
python -m Tantra.data.expert_workbench seed --count 5000
python -m Tantra.data.expert_workbench prepare
python -m Tantra.data.expert_workbench audit
python -m Tantra.data.expert_workbench plan
python -m Tantra.data.expert_workbench status
python -m Tantra.data.expert_workbench report
python -m Tantra.data.expert_workbench train --expert math --model-dir Model --output runs/math-pilot --steps 100 --rank 8 --threads 4
python -m pytest tests -m "not slow" -q
```

For another dataset version, put `--out Datasets/workbench-v2` BEFORE the command.
The default prepared bundle cannot be regenerated in place. Seed generation is
repeatable, but deliberately does not overwrite hand-authored files.

Use `--release-check` only after a pilot improves validation. It evaluates held-out
answers; keep test data out of training and avoid tuning repeatedly on test results.
The expert pilot starts a fresh optimizer and has no resume command. Candidate
weights and metrics are saved on successful completion; interrupted pilots must
restart. The separate overnight recovery campaign from the lost `F:\Tantra`
checkout is NOT in this restored repository.

## Architecture and resource constraints

This is a shared dense base plus LoRA adapters (Q/V/O/down), usually routed once
per request. It is not Colibri-style disk-streamed per-token MoE. The legacy
inference loader loads all deployed adapters into RAM. The new pilot trainer
loads only the base and one selected adapter. Do not claim lazy inference loading,
expert optimizer resume, automatic self-repair or automatic quality-gated growth. The separate public base pilot now supports explicit optimizer/RNG/cursor resume; see PUBLIC_DATA_PIPELINE.md.

Use CPU, no paid APIs/compute, and an 8 GB process / 2 GB free-RAM check for pilots.
Default pilot rank is 8; configured target ranks can be much larger. Count actual
parameters using `rank * layers * sum(in_features + out_features)` for target
projections. Adding expert files does not make the base a 1B model.

## Validation and release

Tests cover catalog coverage, exact verifier behavior, filler rejection, duplicate
removal, frozen splits, output isolation and existing model/math/router behavior.
Add a meaningful failing test for new behavior. Preserve `Model/`, `Model_v1/`,
`checkpoints/`, and original datasets. Do not commit generated large data or weights.
A fresh clone lacks local weights; supply assets or train before running chat.
Record actual commands, metrics, limitations and unresolved work in the status doc.
See TRAINING_PRIORITIES.md for the initial language scope and bounded one-candidate expansion command.

Read PUBLIC_DATA_PIPELINE.md and data_sources.lock.json for the new pinned acquisition, review export and mmap base-pilot path. Live job state is runs/public-campaign-v1/status.json; do not claim pending expert review is complete.

Greeting training has now run as an explicitly undersized English pilot; see GREETING_PILOT.md. Its bundle remains needs_data, and the candidate is not deployed.

See CONVERSATION_UPDATES.md for the audited conversations, new samples, measured expansion and remaining review/training gates.

The expanded model has now been initialized and started training; see VOCABULARY_MIGRATION.md. The older statement that embedding migration is outstanding is superseded for this isolated candidate, not for deployed Model/.

The new 19M-token continuation is configured and launched; read CONTINUED_PRETRAINING.md and its live status before starting another worker.

Read TRAINING_COMMAND.md for python -m Tantra.train, spread-window validation on new runs, review gap counts and reserved conversation evaluation.

Read RESTORED_TRAINING_CONTROLS.md for restored token metrics, evaluation protocol resets, early stopping and cooperative stop behavior.

The colored animated read-only viewer is python -m Tantra.dashboard. See TRAINING_COMMAND.md. It never starts a second trainer.

Read CPU_BATCH_TUNING.md before changing microbatch size. Benchmark reports must match exact checkpoint hashes; larger RAM use is not assumed faster.

Training-time development probes are now implemented; read TRAINING_QUESTION_CHECKS.md. Existing running workers need a clean resume to load hooks; do not train the reserved probes.

Read RESPONSE_REVIEW_1950.md: latest actual responses still fail basic conversation; English/Hinglish probe loss regressed despite lower encyclopedia validation. Do not promote the base as chat-ready.

CONVERSATION_TRAINING_STAGE.md documents the implemented supervised trainer and actual blocked multilingual data gate. Do not claim conversation training started when only base training is running.

Read CONVERSATION_DATA_ACQUISITION.md: automatic approval rejected self-attested multilingual review; a safer licensed human-source candidate acquisition is running with all rows pending. Do not bypass review gates.

GPU training (2026-10-17): `TANTRA_DEVICE=cuda` moves the resumable base trainer to a GPU (CPU remains default; checkpoints/resume unchanged). `python -m Tantra.pretrain --early-stopping-patience N` overrides patience while keeping the saved bad-eval count. Kaggle helpers live in `scripts/kaggle/` (dataset upload + kernel, live notebook dashboard). The Kaggle run resumed at update 1951 and reached 2310 with Hindi/English validation 2.832/5.929; probe answers were still incoherent. Download its output to a new folder; do not overwrite `runs/pretraining-20m-v1`.

Conversation drafts: `scripts/conversation_drafts_v1.py` writes 129 PENDING English drafts (80 situations, 9 conversation experts; identity = Tantra by Atulya AI) and a blank Hindi/Hinglish writing sheet to `runs/conversation-draft-queue-v1/`. `python -m Tantra.data.review_sheets` imports only rows a named human approved (and Hindi/Hinglish only with a named fluent reviewer) through the shared `approve` gate. These drafts alone do not meet the 1000/100/100 family gate.

19M-token stage complete (Kaggle GPU, 2026-10-17): state pilot_complete, 4,639 updates, 18,999,808 target tokens, ~3,560 tok/s, bad_evals 0. Final validation Hindi 2.642 (top-1 43.0%), English 5.699 (top-1 22.0%). Final base.pt equals best-base.pt (SHA-256 673C701D...). Identities and tokenizer hash match the local run. Stored locally at runs/kaggle-output/base-training/ (not in Git). Probe answers at 4,639 are still incoherent, with greedy-decoding repetition; English/Hinglish probe answer loss did not improve. This is the source base for the conversation stage once reviewed data passes the gate.

Greeting pilot on the 64K base (2026-10-17): `runs/greeting-pilot-64k-v1/greeting.pt`, rank 8, 200 steps, CPU 31 s, undersized_pilot_unvalidated (15/2/4 English rows from workbench-conversation-v2). Val loss 9.99 -> 3.66 on only 2 rows. Spot checks on new prompts: Hello!/Good evening/hey there now answered with matching greetings; "Who are you?" still answered "Hello!" (identity needs its own reviewed data). Not deployed; no Hindi/Hinglish.

Identity pilots (2026-10-17): the owner approved all 40 v1 identity drafts in chat (recorded as reviewer, Datasets/conversation-reviewed-identity-v1). Merged with v2 incoming into Datasets/workbench-identity-v1 (82 rows). Frozen splits gave identity_self and identity_capabilities zero validation families, so only identity_limitations could be piloted: runs/identity-limitations-pilot-64k-v1, val loss 7.19 -> 7.35 on 1 row (no measured gain); replies partly right ("No. I can make mistakes.") and partly jumbled. scripts/conversation_drafts_identity_v2.py adds 62 new distinct English identity situations (pending; one Hinglish draft is held back for fluent review) to grow family counts.

identity_self pilot on the 64K base (2026-10-17): owner approved all 62 v2 identity drafts in chat (Datasets/conversation-reviewed-identity-v2); rebuilt Datasets/workbench-identity-v2 (identity_self families 46/3/4). runs/identity-self-pilot-64k-v1: rank 8, 300 steps, 107 s CPU, val loss 6.95 -> 3.33 on 3 rows, undersized_pilot_unvalidated, not deployed. Spot checks on new prompts: "Who are you?" -> "I'm Tantra, an AI assistant."; "Are you ChatGPT?" -> "No. I am Tantra, developed by Atulya AI."; "Who developed you?" wrongly starts with "No,"; longer answers still drift into garbled text. English only.

Identity facts and persona (owner, 2026-10-17): creator **Atul Vij**; parent company **Atulya AI**. Style: warm and human-like, no "assistant"/service wording, but always honest that Tantra is an AI when sincerely asked (never claim to be human). `scripts/conversation_drafts_identity_v3.py` writes 50 PENDING identity_self drafts in this style to `runs/conversation-draft-queue-identity-v3/`. v1/v2 identity_self rows ("developed by Atulya AI", "AI assistant") are superseded; build the next identity bundle from v3 once approved, not by mixing conflicting facts.

identity_self v3 pilot (2026-10-17): owner approved all 50 v3 drafts in chat (Datasets/conversation-reviewed-identity-v3); v3-only bundle Datasets/workbench-identity-v3 (families 45/3/2). runs/identity-self-pilot-64k-v3: rank 8, 300 steps, 82 s CPU, val loss 7.31 -> 3.15 on 3 rows, undersized_pilot_unvalidated, not deployed. Spot checks: correct for "Who are you?", "Who created you?", "Who is your founder?", "Are you a real human?" (honest AI answer), Atul Vij / Atulya AI questions (some are near-copies of training rows, i.e. memorised). Wrong: "What is your name?" -> family answer; "Are you ChatGPT?" -> "alive" answer. 50 rows is far below the gate; answers confuse neighbouring situations.

identity_self v4 pilot (2026-10-17): owner approved all 100 v4 drafts in chat (Datasets/conversation-reviewed-identity-v4). Bundle Datasets/workbench-identity-v4 = v3 + v4 only, 150 rows, families 133/10/7. runs/identity-self-pilot-64k-v4: rank 8, 400 steps, 135 s CPU, val loss 6.81 -> 3.92 on 10 rows (not comparable with v3's 3-row score), undersized_pilot_unvalidated, not deployed. Same 8 unseen prompts vs v3: v4 fixes "What is your name?" and "Are you ChatGPT?" and keeps "Who are you?"/creator correct; compound or unusual questions ("Which firm is behind you?", "Are you from Amazon?") still produce garbled AI-disclaimer text. More distinct situations and the full chat stage on GPU are the next levers.

v5 pilots (2026-10-17): owner approved all 88 v5 drafts in chat (Datasets/conversation-reviewed-v5). identity_self on v3+v4+v5 (Datasets/workbench-identity-v5, 182 rows, families 162/10/10): runs/identity-self-pilot-64k-v5, 400 steps, val 6.81 -> 3.47. On 10 unseen prompts v5 beats v4 on two-part and other-company questions ("Tell me who you are and who built you." -> "I'm Tantra, an AI. Atul Vij created me.") but regressed on "What is your name?" (garbled) and "Who is the person behind Tantra?". greeting on v5 only (Datasets/workbench-greeting-v5, families 26/1/1): runs/greeting-pilot-64k-v5, 200 steps, val 7.99 -> 7.33 on 1 row; friendly style learned but errors ("Good evening" -> "Good morning!", garbled festival names). Both undersized_pilot_unvalidated, not deployed. Adapter-only pilots plateau; more families and the full chat stage are needed.

identity_self v6 pilot (2026-10-17): owner approved 41 v6 drafts (extra wordings/replies joined to existing families, Datasets/conversation-reviewed-identity-v6). Datasets/workbench-identity-v6: 223 rows, families unchanged 162/10/10. runs/identity-self-pilot-64k-v6: 500 steps, 158 s, val 6.81 -> 3.62 (v5: 3.47). On 12 fresh prompts absent from all data, v5 and v6 each answer about 5-6 acceptably; v6 fixed "Who gave you life?" and "Do I speak with a machine?" but broke "Are you a Google product?" and stays garbled on "Which company do you represent?" / "What should I know about you?". Conclusion: rank-8 adapter pilots on this base have plateaued; further gains need full-model supervised chat training (GPU) and more families, not more paraphrases.
