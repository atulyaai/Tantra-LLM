# 64K model migration and CPU pilot

The isolated model now has 65,536 vocabulary entries and 57,156,096 base parameters. Initialized model: runs/model-64k-initial-v1/. Trained candidate/checkpoint directory: runs/vocabulary-64k-v1/base-pilot/. The original Model/ assets are preserved and remain the deployed configuration.

## Initialization

All original token IDs and embedding rows were retained. Each new embedding is initialized as the mean of its old-vocabulary BPE component embeddings. Transformer weights are copied unchanged. No expert adapters or router state were transferred. New output logits change probabilities, so preserved old rows do not imply identical generated replies.

Migration provenance and hashes are in runs/model-64k-initial-v1/migration.json. Existing experts need re-evaluation or retraining against the exact trained 64K base before any release.

## Measured first updates

The first two optimizer updates completed with 8,192 supervised next-token targets. Each update uses 4,096 targets; the selected microbatch is two with four CPU threads. The original base and migrated initialization are unchanged by candidate training.

| Validation language | Before training | After two updates |
| --- | ---: | ---: |
| hi | 13.8899 | 9.7806 |
| en | 10.7212 | 9.9272 |

Sampled peak RSS was 1,931,198,464 bytes (about 1.8 GiB), below the 8 GiB process threshold. Microbatch benchmarks measured approximately 378/403/396 targets per second for sizes 1/2/4. These backward-pass benchmarks are not end-to-end completion estimates. The first two updates plus final validation took 48.81 seconds by the training timer, excluding setup and baseline validation.

The frozen corpus has 500,000 Hindi and 500,000 English training tokens, including document-boundary EOS tokens. Text-token counts are separately recorded. Validation/test documents remain separate. This pilot is a narrow source-shard sample with 53 Hindi and 61 English training documents and only a few held-out documents per language; lower loss does not establish broad world knowledge or conversational accuracy. Hinglish conversation training still awaits suitable reviewed data.

## Continuing job

The hidden vocabulary_training_job worker resumes from the two-update optimizer/RNG/cursor checkpoint and runs one pass over the remaining corpus. Checkpoints are now saved every ten optimizer updates or ten minutes, whichever comes first. Status is runs/vocabulary-64k-v1/status.json; metrics are base-pilot/metrics.jsonl; stdout/stderr are worker.stdout.log and worker.stderr.log. Check process liveness as well as saved status: an earlier 10K worker stopped and left stale status, and its checkpoint was retained.

```powershell
python -m Tantra.data.migrate_vocabulary --tokenizer runs/tokenizer-extended-64k-v1/tokenizer.json --output runs/new-model-64k-initial
python -m Tantra.data.prepare_corpus --output runs/new-64k/corpus-1m --model-dir runs/new-model-64k-initial --train-tokens 1000000
python -m Tantra.data.train_base_pilot --corpus runs/new-64k/corpus-1m --output runs/new-64k/base-pilot --model-dir runs/new-model-64k-initial --checkpoint-updates 10
# For an interrupted existing candidate, repeat its exact paths with --resume.
```

Use new output directories for initialization and data preparation. A stale lock must be inspected against its recorded PID before removal. Resume has no automatic corrupted-checkpoint fallback. This is a one-million-token adaptation pilot, not full pretraining or a 1B model. No candidate is automatically deployed.

Validation: full suite 51 passed; after adding the configurable checkpoint interval, eight relevant source/resume/migration tests passed. Native DLL hashes matched installed package metadata after a temporarily slow PyTorch import. No dependency reinstall or OS/security changes were made.
