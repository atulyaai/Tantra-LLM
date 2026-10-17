# CPU memory and microbatch tuning

More available RAM permits larger microbatches, but does not guarantee faster
matrix multiplication on this four-core CPU. Benchmark candidates use 4096
supervised targets each, one warmup and two timed accumulated backward passes.
They do not change weights or include optimizer/validation overhead. The 8 GiB
process / 2 GiB available-system-memory gates remain; transient allocation peaks
are not a hard OS limit. Larger batches can be slower.

```powershell
python -m Tantra.benchmark_training --stage runs/pretraining-20m-v1 --output runs/pretraining-20m-v1/new-benchmark.json
python -m Tantra.pretrain --resume --benchmark-report runs/pretraining-20m-v1/new-benchmark.json
```

Do not benchmark concurrently with training. Reports must match the exact latest
checkpoint and CPU thread count. Resume rejects mismatched reports. The chosen
microbatch can change FP32 accumulation rounding but leaves the effective update
target fixed at 4096. No optimizer or learning-rate change is implied. Subsequent
normal resumes keep the saved chosen microbatch. Exact-resume identity tests
continue to use unchanged configurations; cross-batch bitwise equivalence is not
claimed. The original model, tokenizer and frozen corpus remain preserved.

Measured 17 October 2026 on the current checkpoint: microbatches 1/2/4/8/16
produced approximately 316/303/325/362/356 targets per second. Batch 8 was selected;
batch 16 used more RAM without improving throughput. The report confirms the
source checkpoint was unchanged. These are accumulated backward benchmarks,
not a guaranteed whole-run speedup; desktop contention and benchmark noise apply.
The resumed real trainer completed update 41 using the selected setting, with a
sampled process RSS of 2,674,077,696 bytes (about 2.49 GiB). The effective target
budget and learning rate remain unchanged. Offline tests: 56 passed, one slow
test deselected. Full benchmark report is local at
runs/pretraining-20m-v1/microbatch-benchmark-v1.json.

Manual override is now supported: python -m Tantra.pretrain --resume --microbatch 16. It is mutually exclusive with --benchmark-report. This changes microbatch only, retains the 4096-target update and memory gates, and may be slower than the measured batch-8 choice. Stopped dashboards now show paused ETA and no live throughput. Five focused tests passed.
