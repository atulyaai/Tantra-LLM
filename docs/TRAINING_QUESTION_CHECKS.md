# Training-time question checks restored

Inspected historical bc8b8d1 main.py, Tantra/train.py and Tantra/eval_suite.py.
The old on_eval callback computed reference answer loss at evaluations, generated
replies every fourth evaluation, logged rough keyword hits and saved checkpoints.
It also supported STOP requests, early stopping, optional growth and max-hours.
Those mechanisms were distinct from real tool actions or source-code repair.

The current compatible hook runs a separate development monitoring suite:
- Every 50 updates: reference-answer loss, token-weighted per language.
- Every 200 updates and final corpus completion: actual greedy question/answers.
- Saved probe_history.jsonl and dashboard recorded-question panel.
- No tool execution, no code editing, no automatic release based on word matches.
- Existing cooperative stop, checkpointing, nonfinite failure diagnostics and
  configurable early stopping remain in effect. Automatic growth/repair is not
  silently enabled or claimed equivalent to the historical implementation.

The nine diagnostic questions are AI-authored; fluent Hindi/Hinglish review is
pending. Exact reference matches are explicitly not semantic answer accuracy.
This repeatedly monitored development suite is separate from the reserved final
conversation evaluation seed, and must not be copied into training data. Neither
suite is large enough for release certification. Do not claim passing routing,
tools or language quality based on these monitoring questions.

New trainer starts/resumes read probe_suite in pretrain_64k.json. The running
process keeps its already loaded code; a read-only evaluation of its saved latest
checkpoint can run without changing weights or optimizer. It competes for CPU,
so periodic synchronous hooks are preferred over continuous parallel evaluation.

```powershell
python -m Tantra.pretrain --resume --microbatch 16 --dashboard
python -m Tantra.training_probes --checkpoint runs/pretraining-20m-v1/base-training/latest.pt --tokenizer runs/pretraining-20m-v1/source-model/tokenizer.json --suite docs/data/training_probes.v1.json --output runs/pretraining-20m-v1/probe-check-now
```

The dashboard also reads the latest one-off probe-check-now history. Existing
viewers need reopening to load new panel code. Tests use a real tiny transformer
to verify probes do not change weights, RNG or training/evaluation mode.
