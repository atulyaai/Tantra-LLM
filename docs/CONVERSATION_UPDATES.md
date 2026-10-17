# Conversation, identity and vocabulary update

Checked all 18,300 legacy curated conversation rows across eleven experts. The heuristic audit found 371 flag occurrences: 141 unsolicited greeting questions, 63 long simple greetings and 167 deployment-dependent identity claims. Flags can overlap and require contextual review; they are not 371 confirmed incorrect replies. Legacy records lack the strict provenance/review contract and remain excluded from prepared training.

The clarification corpus has 1,500 rows but six distinct final responses; refusal has 1,500 rows but four. Repeated short greetings may be correct. Do not inflate coverage through spelling variants. Original datasets were preserved.

Ran 28 generated-response probes across every restored conversation expert. These force one expert at a time and do not measure router performance. Identity replies remain irrelevant or garbled. Local artifacts: runs/conversation-audit-v2/audit.json and runs/conversation-probes-v2/responses.json.

## New examples and identity policy

The tracked source docs/data/conversation_seed.v2.json contains 98 English exchanges with transparent AI content review and 19 Hindi/Hinglish drafts awaiting fluent review. No independent human/native review is claimed. Several exchanges are multi-turn.

| Expert | English AI-reviewed | Multilingual pending |
| --- | ---: | ---: |
| apology | 5 | 0 |
| clarification | 7 | 0 |
| empathy | 6 | 0 |
| farewell | 7 | 2 |
| greeting | 21 | 6 |
| humor | 3 | 0 |
| identity_capabilities | 13 | 2 |
| identity_limitations | 14 | 2 |
| identity_self | 15 | 5 |
| smalltalk | 7 | 2 |

- Greetings: Hi -> Hi! and Hello -> Hello!, without an automatic help question.
- identity_self: Tantra, developed by Atulya AI, a language model. No impersonation or unprompted provider comparisons.
- identity_capabilities: modest capabilities; tool availability depends on the application. Never invent tool execution.
- identity_limitations: uncertainty, unavailable metadata, limited context, no personal experience or automatic learning claims.

The bundle has 98 accepted records and remains needs_data. Pending drafts are outside incoming/. These samples have not been trained into deployed weights. They are training data, not a runtime response lookup table. Do not launch an English-only conversation release to hide missing Hindi/Hinglish coverage.

## Balance and tokenizer expansion

Tantra/config/training_scope.json now sets Hindi/English base token shares to 50/50, replacing 60/40. Base windows alternate languages. Hinglish remains a separate conversation/evaluation track. Reviewed conversation coverage is not balanced yet.

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

## Reproduce

```powershell
python -m Tantra.data.conversation_quality audit --output runs/new-conversation-audit
python -m Tantra.data.conversation_quality probe --output runs/new-conversation-probes
python -m Tantra.data.conversation_quality export --output Datasets/workbench-conversation-new
python -m Tantra.data.expert_workbench --out Datasets/workbench-conversation-new prepare
python -m Tantra.data.tokenizer_candidates --output runs/new-balanced-tokenizers --chars-per-language 20000000
python -m Tantra.data.tokenizer_candidates --output runs/new-extended-64k --extend-from runs/new-balanced-tokenizers/tokenizer-65536.json --texts runs/new-balanced-tokenizers/texts.jsonl --target-vocab 65536
```

Use new immutable directories. The background base pilot keeps its existing frozen tokenizer; the 64K model migration is a separate stage. Its durable worker was restarted with resumable source acquisition and the 50/50 scope. Check runs/public-campaign-v1/status.json for actual progress.

Validation: 50 tests passed, including exact base resume, equal bilingual quotas, reviewed/pending boundaries and token-ID preservation. Fluent review, representative Hinglish evaluation, weight migration and quality validation remain outstanding.
