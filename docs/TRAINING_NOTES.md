# Training notes: conversation updates and the 64K vocabulary migration

Two chronological records, kept because `AGENTS.md` requires reading them before
language/tokenizer changes: the conversation/identity audit (older) and the 64K
vocabulary migration (newer). Figures describe local runs that were removed from
the tree in the 2026-11 reset; treat them as recorded history, not current state.

## Part 1 — Conversation, identity and vocabulary update (earlier record)

Checked all 18,300 legacy curated conversation rows across eleven experts. The heuristic audit found 371 flag occurrences: 141 unsolicited greeting questions, 63 long simple greetings and 167 deployment-dependent identity claims. Flags can overlap and require contextual review; they are not 371 confirmed incorrect replies. Legacy records lack the strict provenance/review contract and remain excluded from prepared training.

The clarification corpus has 1,500 rows but six distinct final responses; refusal has 1,500 rows but four. Repeated short greetings may be correct. Do not inflate coverage through spelling variants. Original datasets were preserved.

Ran 28 generated-response probes across every restored conversation expert. These force one expert at a time and do not measure router performance. Identity replies remain irrelevant or garbled. Local artifacts: runs/conversation-audit-v2/audit.json and runs/conversation-probes-v2/responses.json.

### New examples and identity policy

The tracked source docs/data/conversation_seed.v2.json contains 705 English exchanges with transparent AI content review and 19 Hindi/Hinglish drafts awaiting fluent review. No independent human/native review is claimed. Several exchanges are multi-turn.

| Expert | English AI-reviewed | Multilingual pending |
| --- | ---: | ---: |
| apology | 68 | 0 |
| clarification | 88 | 0 |
| empathy | 51 | 0 |
| farewell | 91 | 2 |
| greeting | 86 | 6 |
| humor | 67 | 0 |
| identity_capabilities | 67 | 2 |
| identity_limitations | 38 | 2 |
| identity_self | 93 | 5 |
| smalltalk | 56 | 2 |

- Greetings: Hi -> Hi! and Hello -> Hello!, without an automatic help question.
- identity_self: Tantra, developed by Atulya AI, a language model. No impersonation or unprompted provider comparisons.
- identity_capabilities: modest capabilities; tool availability depends on the application. Never invent tool execution.
- identity_limitations: uncertainty, unavailable metadata, limited context, no personal experience or automatic learning claims.

The reviewed seed now has 705 accepted records and remains needs_data. Pending drafts are outside incoming/. These samples have not been trained into deployed weights. They are training data, not a runtime response lookup table. Do not launch an English-only conversation release to hide missing Hindi/Hinglish coverage.

### Balance and tokenizer expansion

Tantra/training_scope.json now sets Hindi/English base token shares to 50/50, replacing 60/40. Base windows alternate languages. Hinglish remains a separate conversation/evaluation track. Reviewed conversation coverage is not balanced yet.

The tokenizer experiment used exactly 17,417,313 training Unicode characters per language. Equal tokenizer characters and equal base-training tokens are different explicit budgets. Document/tree families were split before tokenizer fitting.

Built a separate 65,536-token vocabulary extension, preserving all existing IDs and BPE merge priorities. Artifact: runs/tokenizer-extended-64k-v1/tokenizer.json. SHA-256: a9d2915fa0000758f766995f301787e5386db970c093f7d97cd761ae568fd80f. The original tokenizer remains unchanged.

| Holdout | Current tokens | Extended tokens | Reduction |
| --- | ---: | ---: | ---: |
| hi | 929,236 | 754,675 | 18.8% |
| en | 340,522 | 258,897 | 24.0% |
| hinglish | 113 | 107 | 5.3% |

All measured texts round-trip. Hindi holdout: 468 documents; English: 1,543; Hinglish: only eight original draft texts totaling 301 characters. Hinglish compression is provisional, not proof of fluency. A standalone new 64K BPE regressed on that tiny Hinglish sample; the extension avoids the measured regression. No automatic promotion under the within-5%-of-best-every-category policy was claimed.

Vocabulary counts subword/byte pieces, not known words. Byte-level encoding supports arbitrary text without proving comprehension. Context is separately limited to 512 total tokens on the restored model.

At width 512, the tied embedding/output table grows from 5,261,824 to 33,554,432 parameters. The same transformer with 64K would have approximately 57,156,096 base parameters and needs a CPU benchmark. Preserved IDs do not remove the need to expand/initialize the matrix and train/re-evaluate the base and adapters. Do not attach the larger tokenizer to unchanged 10,277-row embeddings.

### Reproduce

```powershell
python -m Tantra quality_cli audit --output runs/new-conversation-audit
python -m Tantra quality_cli probe --output runs/new-conversation-probes
python -m Tantra quality_cli export --output Datasets/workbench-conversation-new
python -m Tantra workbench_cli --out Datasets/workbench-conversation-new prepare
python -m Tantra candidates_cli --output runs/new-balanced-tokenizers --chars-per-language 20000000
python -m Tantra candidates_cli --output runs/new-extended-64k --extend-from runs/new-balanced-tokenizers/tokenizer-65536.json --texts runs/new-balanced-tokenizers/texts.jsonl --target-vocab 65536
```

Use new immutable directories. The background base pilot keeps its existing frozen tokenizer; the 64K model migration is a separate stage. Its durable worker was restarted with resumable source acquisition and the 50/50 scope. Check runs/public-campaign-v1/status.json for actual progress.

Validation: 50 tests passed, including exact base resume, equal bilingual quotas, reviewed/pending boundaries and token-ID preservation. Fluent review, representative Hinglish evaluation, weight migration and quality validation remain outstanding.

## Part 2 — 64K model migration and CPU pilot

The isolated model now has 65,536 vocabulary entries and 57,156,096 base parameters. Initialized model: runs/model-64k-initial-v1/. Trained candidate/checkpoint directory: runs/vocabulary-64k-v1/base-pilot/. The original Model/ assets are preserved and remain the deployed configuration.

### Initialization

All original token IDs and embedding rows were retained. Each new embedding is initialized as the mean of its old-vocabulary BPE component embeddings. Transformer weights are copied unchanged. No expert adapters or router state were transferred. New output logits change probabilities, so preserved old rows do not imply identical generated replies.

Migration provenance and hashes are in runs/model-64k-initial-v1/migration.json. Existing experts need re-evaluation or retraining against the exact trained 64K base before any release.

### Measured first updates

The first two optimizer updates completed with 8,192 supervised next-token targets. Each update uses 4,096 targets; the selected microbatch is two with four CPU threads. The original base and migrated initialization are unchanged by candidate training.

| Validation language | Before training | After two updates |
| --- | ---: | ---: |
| hi | 13.8899 | 9.7806 |
| en | 10.7212 | 9.9272 |

Sampled peak RSS was 1,931,198,464 bytes (about 1.8 GiB), below the 8 GiB process threshold. Microbatch benchmarks measured approximately 378/403/396 targets per second for sizes 1/2/4. These backward-pass benchmarks are not end-to-end completion estimates. The first two updates plus final validation took 48.81 seconds by the training timer, excluding setup and baseline validation.

The frozen corpus has 500,000 Hindi and 500,000 English training tokens, including document-boundary EOS tokens. Text-token counts are separately recorded. Validation/test documents remain separate. This pilot is a narrow source-shard sample with 53 Hindi and 61 English training documents and only a few held-out documents per language; lower loss does not establish broad world knowledge or conversational accuracy. Hinglish conversation training still awaits suitable reviewed data.

### Continuing job

The hidden vocabulary_training_job worker resumes from the two-update optimizer/RNG/cursor checkpoint and runs one pass over the remaining corpus. Checkpoints are now saved every ten optimizer updates or ten minutes, whichever comes first. Status is runs/vocabulary-64k-v1/status.json; metrics are base-pilot/metrics.jsonl; stdout/stderr are worker.stdout.log and worker.stderr.log. Check process liveness as well as saved status: an earlier 10K worker stopped and left stale status, and its checkpoint was retained.

```powershell
python -m Tantra migrate_cli --tokenizer runs/tokenizer-extended-64k-v1/tokenizer.json --output runs/new-model-64k-initial
python -m Tantra prepare_corpus_cli --output runs/new-64k/corpus-1m --model-dir runs/new-model-64k-initial --train-tokens 1000000
python -m Tantra pilot_cli --corpus runs/new-64k/corpus-1m --output runs/new-64k/base-pilot --model-dir runs/new-model-64k-initial --checkpoint-updates 10
# For an interrupted existing candidate, repeat its exact paths with --resume.
```

Use new output directories for initialization and data preparation. A stale lock must be inspected against its recorded PID before removal. Resume has no automatic corrupted-checkpoint fallback. This is a one-million-token adaptation pilot, not full pretraining or a 1B model. No candidate is automatically deployed.

Validation: full suite 51 passed; after adding the configurable checkpoint interval, eight relevant source/resume/migration tests passed. Native DLL hashes matched installed package metadata after a temporarily slow PyTorch import. No dependency reinstall or OS/security changes were made.
