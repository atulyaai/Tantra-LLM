# Tantra-LLM Issue Tracker

## Current Status
- **66/66 tests passing**
- **Code compiles without errors**
- **Active development** - experimental local-first LLM training engine

## Recently Fixed (Local Commits)
1. **Fix: enable archive_checkpoints by default in training mode**
   - Changed `archive_checkpoints=False` to `archive_checkpoints=True` in `Tantra/cpu_cli.py`
   - Ensures checkpoint archiving is enabled by default for training runs

2. **Fix: add error handling to interactive chat REPL**
   - Added `except Exception` handler to `run_interactive_chat()` in `main.py`
   - Prevents REPL crashes on tokenizer or generation errors
   - Provides user-friendly error messages via console or print

## Roadmap Items (From ROADMAP.md)
### Now: Make the CPU path reliable
- [ ] Finish the current base-pretraining run and record held-out metrics
- [ ] Build a fixed evaluation set for general chat, math, code, safety, Hindi, and Sanskrit
- [ ] Run a small instruction fine-tune after base pretraining

### Next: Measured quality and speed
- [ ] Benchmark dense causal, dense ALRA, micro10, and moe2 under identical CPU settings
- [ ] Add automatic test coverage for checkpoint-config restoration and the CPU chat launcher

### Adapters and growth: experimental
- [ ] Train one category adapter at a time against an untouched held-out split
- [ ] Track adapter improvement, parameter count, and routing accuracy

### Longer term
- [ ] Package supported CLI entry points as installable commands
- [ ] Make dataset preparation reproducible with manifests and data cards
- [ ] Evaluate GPU support on actual available hardware

## New Tech Opportunities
1. **torch.compile inductor backend** - Enable PyTorch 2.0 compiler for CPU kernel fusion
2. **MTP speculative decoding** - Improve generation throughput with multi-token prediction
3. **Category adapter vitrualization** - Allow dynamic category loading/unloading without restart
4. **Distributed CPU training** - Coordinate multiple CPU instances for faster training

## Bug Reports
- No critical bugs found - all 66 tests pass
- Code quality improvements tracked in local commits above

## Contributing
See CONTRIBUTING.md for guidelines on contributing code, benchmarks, and measurements.