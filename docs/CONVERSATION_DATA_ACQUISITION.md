# Conversation-data acquisition and approval boundary

A proposed trilingual self-reviewed import was rejected by automatic approval
review because it would attest fluent review of AI-authored Hindi/Hinglish data
without independent/native-human review. The rejected command did not execute.
No new reviewed v3 dataset was created. Do not bypass the rejection by relabeling
metadata or using another import route. The existing 19 multilingual drafts remain
pending; the strict supervised training gates remain unchanged.

Safer work completed: the Aya human-annotated instruction dataset was verified
through its official dataset card/API, pinned at commit
f9ea04583f02a8f86404ff6c58bf75fe637df8a2 under Apache-2.0. Hindi/English train
and original test files have exact sizes and SHA-256s in
`docs/data/conversation_sources.lock.json`. This is a separate source lock, so
existing pretraining manifests are not changed.

The acquisition command is:

```powershell
python -m Tantra.data.instruction_candidates --output runs/aya-conversation-review-v1
```

A visible CMD acquisition worker was launched. It downloads bounded verified
ranges, keeps source attribution/cards, streams Parquet and writes pending
Hindi/English candidates. It reserves the source test first and excludes exact
question/answer duplicates from candidates. No source demographic data is fetched.
No record is assigned an expert, marked reviewed or made trainable automatically.
Hindi is not relabeled Hinglish. Semantic/source-family grouping, content safety,
correctness, expert identity conflicts and actual fluent review still require work.

Live download progress is in the visible acquisition console. On successful
completion, index.json records counts at runs/aya-conversation-review-v1/. A
candidate count is not a reviewed-data or independent-family count. The original
source test must never be used as training data. No paid services were used.

The supervised trainer, readiness report and review pack remain available. Real
chat training is not launched while the balanced reviewed-data gate fails.
Tests confirm pending status, no fake expert labels and reserved-test exclusion.
