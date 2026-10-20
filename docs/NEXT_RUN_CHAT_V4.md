# Next run: chat-tune the serving base on the bulk batch (after the owner's spot-check)

Blocked on one human step: the owner marks `runs/conversation-bulk-v1/queue/spot-check.csv`
(90 stratified rows, one per topic family; Y/N in the `ok (Y/N)` column). Nothing below is
run before that, and nothing here creates review marks.

```powershell
# 1. Approve the batch under docs/BULK_DATA_POLICY.md (refuses unless the exact recorded draw is fully marked and >= 95% Y)
.venv\Scripts\python.exe -m Tantra.data.bulk_review approve --queue runs/conversation-bulk-v1/queue --sample runs/conversation-bulk-v1/queue/spot-check.csv --reviewer "<owner name>" --rationale "<what was checked>" --output Datasets/conversation-reviewed-bulk-v1

# 2. Merge with the row-reviewed persona data, drop persona-rule violations, prepare frozen splits
.venv\Scripts\python.exe scripts\build_chat_bundle.py --out Datasets/workbench-chat-en-v4 --sources Datasets/workbench-chat-en-pilot-v3/incoming/conversation Datasets/conversation-reviewed-bulk-v1/conversation

# 3. Chat-tune the pretrained 64K base (English pilot mode; best.pt kept at lowest validation loss)
.venv\Scripts\python.exe -m Tantra.conversation_train train --english-pilot --epochs 3 --bundle Datasets/workbench-chat-en-v4 --source runs/kaggle-output/base-training --output runs/chat-en-v4

# 4. Compare against the current best chat model on fresh prompts, then serve it
.venv\Scripts\python.exe scripts\compare_chat_pilot.py runs\chat-en-v4 runs\kaggle-output\base-training best.pt
```

Expected size: ~9.4K bulk rows + ~500 persona rows; on the 4-core CPU one epoch is roughly
3,300 updates (~2 h), so 3 epochs fit an overnight run, or use the Kaggle helpers in
`scripts/kaggle/` (`chat_pilot_upload.ps1 -Version v4` after copying the bundle name).
After this, freeze the result as the serving base (roadmap M3) and train experts only on
domain data the base has not seen (roadmap M4/M7; see the M3/M4 findings in
docs/ROADMAP_CPU_EXPERTS.md).
