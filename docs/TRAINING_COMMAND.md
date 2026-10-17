# Unified training command and remaining quality work

From F:\Tantra-LLM:

```powershell
.venv\Scripts\python -u -B -m Tantra.train
.venv\Scripts\python -B -m Tantra.train status
.venv\Scripts\python -u -B -m Tantra.train watch
```

The default command attaches to an existing verified worker. Otherwise it starts
or resumes the configured 19M-token continuation. It refuses a stale worker lock
until the PID is inspected and the lock archived. The viewer shows updates,
training loss, latest language validation, speed/ETA measured after attachment,
checkpoint path/age and worker liveness. Ctrl+C in an attached viewer closes only
the viewer; Ctrl+C while this command itself owns training interrupts training.
Validation appears after the first evaluation, not on every update.

New trainer invocations now sample validation windows evenly across each shard
instead of only its first windows. This broadens token coverage, but does not
replace more independent documents. The currently running process keeps its
loaded evaluation implementation. Do not directly compare old first-window
scores with new spread-window scores as if their evaluation sets were identical.

`docs/data/conversation_review_gaps.json` reports reviewed/pending language counts
for greetings and all three identity experts in the authored seed. Current
English-heavy reviewed coverage is not a balanced multilingual release dataset.
The 19 original Hindi/Hinglish drafts remain outside training until fluent review.
No manufactured approvals or repeated variants have been added to inflate counts.

A reserved 18-case conversation suite now specifies correctness/style rubrics in
Hindi, English and Hinglish. It is a small AI-authored evaluation seed; multilingual
prompts need fluent review and much broader final coverage. Keep it out of training.

```powershell
.venv\Scripts\python -B -m Tantra.evaluate_conversation --model-dir runs/pretraining-20m-v1/base-training --output runs/conversation-evaluation-new.json
```

Run after the candidate base.pt and tokenizer.json have been exported. This
records actual generated answers, routes, expected expert and source hashes.
It leaves correctness review pending instead of calling a route hit an accurate
answer. A base without adapters has no validated expert routing. Router macro-F1,
independent rubric scoring, safety coverage and a 90% conversation release gate
remain outstanding; this small suite does not certify them.

No running training weights or frozen datasets were replaced by these changes.
## Configured training script

The original repo exposed `python -m Tantra base`, `expert`, `router`, `stage1`
and `stage2`, using TrainArgs for step count, LR, batch size and sequence length.
That legacy path loads data in RAM, re-splits data and saves into its model
folder. It is preserved, but should not be used on frozen new bundles or Model/
without understanding those differences.

The new base command reuses the shared disk-backed resumable loop:

```powershell
.venv\Scripts\python -u -B -m Tantra.pretrain --config Tantra/config/pretrain_64k.json --info
.venv\Scripts\python -u -B -m Tantra.pretrain --config Tantra/config/pretrain_64k.json --resume
```

Config specifies source model, corpus, output, CPU threads, maximum updates,
validation coverage and checkpoint interval. It displays AdamW LR 1e-4, decay
0.01, FP32, 256 target positions per window, 4096 targets per update, benchmarked
microbatch 1/2/4. These optimizer/token-budget settings remain fixed in this
trainer and changing them in JSON is explicitly rejected, rather than silently
ignored. Model width/layers/heads/vocabulary are read from the source checkpoint,
not overridden by an incompatible preset. No parameter count is inferred from
expert taxonomy estimates.

On new runs the shared loop prints model dimensions/actual base parameter count,
training loss, tokens/sec, ETA, memory, language validation and checkpoint events.
It evaluates every 50 updates and at completion; configured validation samples
128 spread windows per language. The running worker keeps its loaded code and
settings until a clean explicit resume; no hot patch is claimed. Resume preserves
weights/optimizer/cursor/RNG; changing evaluation coverage changes the evaluation
protocol and should be recorded when comparing validation scores.

Resume now archives locks only when their recorded PID is confirmed absent. Live or unreadable lock ownership is refused; checkpoint files are preserved. This handling has dedicated tests.

## Colored animated dashboard

```powershell
Set-Location 'F:\Tantra-LLM'
& '.\.venv\Scripts\python.exe' -u -B -m Tantra.dashboard
```

The Rich terminal dashboard redraws live with colored panels, a progress bar,
activity animation and a loss-history trend. It reports actual worker liveness,
updates/tokens, measured timestamp-based speed/ETA, loss/LR/gradient, CPU/system
memory, sampled peak training RAM, per-language validation/perplexity/top-1/top-5,
and checkpoint age. Validation from a previous completed session is labeled
explicitly until a new evaluation is logged. Resumed/rolled-back metric segments
are separated for speed calculations. The animation is viewer activity, not
proof of completed updates. Ctrl+C closes the viewer only. No new training or
checkpoint mutation is performed. `--once` prints a static snapshot;
`--export-html runs/preview.html` exports a static colored preview.

The model/settings panel currently describes the configured 64K continuation
(57.16M parameters); this is not a generic arbitrary-model inspector. CPU load is
system-wide, and RSS is a sampled training high-water observation, not an OS
hard limit. Keep the console reasonably wide for the multi-column details.

To open the colored dashboard alongside training use: python -m Tantra.pretrain --resume --microbatch 16 --dashboard. It opens a separate console and does not duplicate a live training worker. Plain logs stay in the training console.
