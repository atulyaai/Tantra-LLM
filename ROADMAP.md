# Tantra-LLM roadmap

This is the single maintained planning document. Completed work is recorded in
the changelog; `TASKS.md` was removed to avoid a second, conflicting roadmap.

## Now: make the CPU path reliable

- [x] Maintain a compact 32K dense CPU profile with tied embeddings.
- [x] Save a single resumable `Latest` checkpoint for CPU training.
- [x] Restore checkpoint architecture/config instead of loading mismatched
      weights into the legacy model shape.
- [x] Provide a CPU-profile chat launcher that loads the correct checkpoint.
- [ ] Finish the current base-pretraining run and record held-out metrics.
- [ ] Build a fixed evaluation set for general chat, math, code, safety, Hindi,
      and Sanskrit before judging response quality.
- [ ] Run a small instruction fine-tune after base pretraining; do not present
      raw base-model completion as an assistant response benchmark.

## Next: measured quality and speed

- [ ] Benchmark dense causal, dense ALRA, `micro10`, and `moe2` under identical
      CPU settings. Publish parameters, tokens/sec, memory, validation loss,
      and a small generation evaluation.
- [ ] Decide whether ALRA is retained based on measured CPU speed and quality.
- [ ] Keep MTP and latent reasoning off for base pretraining; evaluate them only
      in an instruction/reasoning fine-tune.
- [ ] Add automatic test coverage for checkpoint-config restoration and the
      CPU chat launcher.

## Adapters and growth: experimental

- [ ] Train one category adapter at a time against an untouched held-out split.
- [ ] Track adapter improvement, parameter count, and routing accuracy.
- [ ] Add capacity only after sustained evidence of a plateau.
- [ ] Investigate structured pruning/distillation separately; it is not a safe
      in-place reduction of an existing checkpoint.
- [ ] Support vocabulary conversion only as an explicit conversion + fine-tune
      workflow, never as an automatic checkpoint mutation.

## Longer term

- [ ] Package supported CLI entry points as installable commands.
- [ ] Make dataset preparation reproducible with manifests and data cards.
- [ ] Add export/inference benchmarking only after model quality is measured.
- [ ] Evaluate GPU support on actual available hardware rather than advertising
      untested acceleration.

## Not committed

The project does not currently commit to a trillion-parameter model, 500-expert
bank, multimodal production pipeline, or a fixed performance multiplier. Those
require independent benchmarks, storage plans, and training resources.
