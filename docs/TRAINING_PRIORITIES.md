# Sequential training priorities

Initial scope: English and Hindi, including Hinglish code switching. Retain all
164 taxonomy IDs for compatibility; defer other language experts. This is the
expansion queue policy, not a claim that legacy inference disables their weights.

1. Audit and review conversation data: greeting, identity, clarification, empathy,
   capabilities, limitations, farewell, smalltalk and refusal. Require independent
   train/validation/test families; never copy one dataset into multiple experts.
2. Train one isolated candidate at a time. Arithmetic, JSON and CSV currently have
   verified bootstrap datasets; the other 161 experts still need prepared data.
3. Add tool schema selection, arguments, error handling and multi-step reasoning
   with executable checks. Tool execution belongs to the application runtime.
4. Build a licensed, revision-pinned Hindi/English foundation corpus and reviewed
   factual evaluations. Languages alone do not provide world knowledge. Separate
   learned factual accuracy from answers supplied by retrieval or calculators.
5. Evaluate a useful small-model release before benchmarking larger bases. More
   adapters do not turn the 28.86M base into a 1B base. Larger widths need explicit
   transfer or initialization and measured hardware feasibility.

## One-at-a-time expert expansion

```powershell
python -m Tantra.data.auto_growth --steps 100
```

Each invocation selects one unattempted data-ready expert within the initial
language scope, trains a rank-8 candidate using four CPU threads, and saves an
event under `runs/expert-growth/<expert>/`. A lock rejects concurrent invocations.
Failed attempts remain recorded and are not retried automatically. After a hard
process kill, inspect the recorded PID before manually removing a stale lock.
Use a new output directory only after reviewing previous attempts.

Candidates are never automatically deployed. Require independent answer tests,
per-language conversation success >=90%, router macro-F1 >=0.90 and regression
checks before release. Bootstrap loss improvement alone is insufficient.

Depth growth (`grow_depth`) and adding adapters are different operations. This
queue adds candidate adapters only. Automatic depth trials, expert optimizer resume,
rollback and automatic promotion are not implemented in the restored checkout.

The separate public base pilot supports explicit resume; see [public-data pipeline](PUBLIC_DATA_PIPELINE.md).
