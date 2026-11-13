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

The tracked source docs/data/conversation_seed.v2.json contains 1129 English exchanges with transparent AI content review and 19 Hindi/Hinglish drafts awaiting fluent review. No independent human/native review is claimed. Several exchanges are multi-turn. Batch 3 (127 rows, 58 multi-turn) was authored on 2026-11-12 to cover multi-turn name recall, provider challenge, farewell-after-thanks and topical small talk.

| Expert | English AI-reviewed | Multilingual pending |
| --- | ---: | ---: |
| apology | 92 | 0 |
| clarification | 112 | 0 |
| empathy | 87 | 0 |
| farewell | 122 | 2 |
| greeting | 134 | 6 |
| humor | 79 | 0 |
| identity_capabilities | 103 | 2 |
| identity_limitations | 75 | 2 |
| identity_self | 192 | 5 |
| smalltalk | 133 | 2 |

- Greetings: Hi -> Hi! and Hello -> Hello!, without an automatic help question.
- identity_self: Tantra, developed by Atulya AI, a language model. No impersonation or unprompted provider comparisons.
- identity_capabilities: modest capabilities; tool availability depends on the application. Never invent tool execution.
- identity_limitations: uncertainty, unavailable metadata, limited context, no personal experience or automatic learning claims.

The reviewed seed now has 1129 accepted records, meeting the 1000-row pilot minimum while remaining needs_data until 100/100 validation and test checks exist per expert. Pending drafts are outside incoming/. These samples have not been trained into deployed weights. They are training data, not a runtime response lookup table. Do not launch an English-only conversation release to hide missing Hindi/Hinglish coverage.

The 1002-row bundle was trained end-to-end on 2026-11-12 (all 10 conversation experts, rank 8, 800 steps, champion base). Every candidate improved validation loss, but only 4/10 were structurally clean on held-out replies and both integrations regressed the frozen suites (strict 19/32 single + 8/16 chat; all-improved 15/32 + 5/16; champion control 22/32 + 10/16). A second bounded round with 300 steps on the 7 deployed experts saturated validation loss without fixing the drift (strict 20/32 + 10/16, chat mean 0.75). The rung-3 champion was retained both times. Record: validation loss is not a selection signal at this scale; behavior-level gates decide. Training on this bundle is stopped. No deployable weights changed.

### Batch 3, rank and the 30M base (2026-11-12)

Batch 3 adds 127 reviewed English rows (78 families, 58 multi-turn) to the 1129-row bundle: multi-turn name recall, name-after-farewell, provider challenge, identity chain, address-me, farewell-after-thanks, mood flow, thanks-then-close, plus topical small talk (weather, food, music, sport, weekend, pets, tiredness) and capability/limit detail. The authoring pass rejects any user turn or conversation that matches a held-out eval case, and any duplicate exchange. Disclose plainly: the batch targets the behaviors the chat suite measures, so it is not independent evidence by itself; the bundle's own held-out test rows are the cross-check.

Measured results on the frozen suites (champion control 22/32 at 0.7812 and 10/16 at 0.7375, reproduced in the same run):

| candidate | single-turn | chat |
| --- | ---: | ---: |
| all seven retrained, integrated | 13/32 (0.6188) | 9/16 (0.7) |
| identity_self swap, 600 steps | 17/32 (0.6875) | 9/16 (0.7) |
| identity_self swap, 250 steps | 18/32 (0.6813) | 7/16 (0.6) |
| smalltalk swap | 22/32 (0.7875) | 10/16 (0.7375) |
| empathy swap | 21/32 (0.7625) | 10/16 (0.7375) |
| greeting swap | 21/32 (0.7688) | 10/16 (0.7375) |
| farewell swap | 21/32 (0.7562) | 9/16 (0.7) |
| identity_capabilities swap | 22/32 (0.7812) | 10/16 (0.75) |
| identity_limitations swap | 22/32 (0.7812) | 10/16 (0.7375) |

Nothing adds a passed case over the champion. The identity_capabilities swap ties at 10/16 with a slightly higher mean (0.75 vs 0.7375) and no case-level gain; it is recorded as a tie, not frozen, because equal pass counts are not a capability gain.

Two further capacity checks, both null:

- Adapter rank 16 (smalltalk, 600 steps) cut validation loss from 5.44 to 3.55 and still scored 21/32 + 10/16, losing `good-news`. Higher rank fits the same data better without transferring.
- The 30M preset with all seven experts retrained on this bundle scored 17/32 single + 5/16 chat; the untouched same base scores 18/32 + 4/16. It remains underfit at this CPU budget, so scale alone does not buy the missing behavior.

Root cause finally measured directly: name recall is a copy-from-context task this base cannot do. On the bundle's own held-out name-recall rows (never trained on), the retrained identity_self adapter answers "What did I say my name was?" with "Atul Vij." and "Your name is fact or call me Tantra."; the champion answers "eyes? Atulya AI built me — I'm Tantra." Both fail in-distribution, not on the eval suite. Until the base can copy a token from earlier context, no adapter, rank, router or prompt change fixes chat memory.

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

## Part 3 - Rung 6: bilingual corpus builder and chat SFT (2026-11-13)

The 64K migration record in Part 2 refers to local directories that were later
removed from 
uns/; only the 2,897-token champion lineage survives in this
workspace. Do not treat Part 2 model paths as present artifacts.

### Pinned corpus preparation

prepare_corpus_cli was used unchanged, on the pinned wikipedia-en and
wikipedia-hi sources, for two corpora:

- 
uns/rung6-corpus-6m - 6M train tokens (later excluded from its successor)
- 
uns/rung6-corpus-16m - 8M English + 8M Hindi train tokens, 444,444 val and
  444,444 test tokens per language, built with
  --exclude-corpus runs/rung6-corpus-6m so no text repeats.

Split assignment is sha256(family) mod 100 (90/5/5) and family exclusion is
checked before acceptance, so excluding a prior corpus removes whole article
families rather than individual rows. That is why a later corpus cannot leak
held-out families from an earlier one.

### Continued pretraining

3,500 updates = 14,336,000 target tokens at microbatch 4 (~3,275 tokens/s on
this CPU), exported to 
uns/rung6-pilot-16m. Held-out spread-v1 validation
over 32 windows:

| language | loss before | loss after | top-1 before -> after |
| --- | ---: | ---: | --- |
| en | 11.508 | 4.096 | 0.42% -> 22.35% |
| hi | 15.329 | 0.901 | 0.00% -> 72.25% |

Three bad evals were recorded by the trainer during the run (the resumable
trainer's own counter); the exported best base is checksummed in 
eport.json.
The persistent gap between Hindi and English is a property of the 2,897-token
vocabulary, which was built with a Hindi-heavy budget, not of the training loop.

### Supervised chat tuning after pretraining

Regressing a pretrained base back to conversation requires full-parameter SFT;
rank-8 LoRA on the drifted base collapsed to 0/32 + 0/16
(
uns/rung6-adapted). The pretraining export writes an empty router registry,
so the chat stage used a staged directory (
uns/rung6-base-fromwiki) that
restores the champion's 7-name router and 
outer_ngram.pt before SFT.

Early stopping matters at this data size: 3,000 steps at LR 1e-4 reached train
loss 0.02 while validation climbed 3.62 -> 5.13. With evaluation every 100 steps
and patience 3, the best epoch was step 500 at validation 3.6173
(
uns/rung6-sft2, recorded in sft-report.json).

Result: base-only 11/32 + 9/16; with seven retrained experts 14/32 + 4/16 versus
the frozen champion 22/32 + 10/16. Six of seven experts validated worse than the
base they were trained on. The base is more fluent in free conversation but
scores lower on the suite, and no candidate was deployed.

### Copy-from-context: what actually trains it (rung 7)

Three negative results and one positive, all on the champion 2,897-token base:

1. rank-8 adapter on 768 extra name rows -> 0/24 unseen names, despite val 7.83 -> 3.66
2. full-parameter training on the same 768 rows -> 0/24, despite val 8.73 -> 0.51
3. chat SFT with only the reviewed bundle -> 0/24
4. 4,000 generated syllable names with 200 held out, full-parameter, lr 3e-4,
   2,500 steps, 8,000 rows -> copy accuracy 0.00, 0.00, 0.025, 0.15, 0.40, 0.625 at
   steps 0/500/1000/1500/2000/2500, final 0.660 on 100 held-out names

The model is not incapable of copying; it only learns the circuit when the corpus
makes memorisation impossible. With 64 names the optimiser memorises the name list
because that is enough to drive loss down.

Transfer: continuing supervised tuning on the reviewed bundle plus the real-name
curriculum keeps the circuit and yields 11/24 then 13/24 unseen-name recall, at the
cost of chat-suite parity (6/16 vs the champion's 10/16). Both curriculum sets are
template-generated, marked not_reviewed, and are not seed data.

Selection note: bundle validation loss cannot select for a capability the bundle
does not contain. The runs above select on a composite of held-out dev-name recall
and validation loss, keeping the eval names untouched until the final measurement.
