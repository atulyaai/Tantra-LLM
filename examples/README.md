# Small CPU training sample

Run from the repository root:

```bash
python examples/train_small_sample.py --output sample_output/small_cpu
```

The script trains a deliberately tiny one-layer model on a small, fixed toy corpus,
writes `base.pt`, `tokenizer.json`, and `training_report.json`, then reloads the
checkpoint and verifies identical logits.

This is an engineering smoke test only. The toy data is not representative of
Hindi, English, Hinglish, factual knowledge, or real user conversations. A falling
training loss on this corpus does not establish general language capability.
