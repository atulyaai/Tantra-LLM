# Scaling Tantra toward 1B parameters: what this setup allows (2026-10-21)

Owner goal: a 1B-parameter Tantra with broad knowledge, reasoning, fast CPU serving and many
experts. This file records the arithmetic so plans stay honest. Hardware: 4-core i3-9100F,
16 GB RAM, no usable GPU; free Kaggle T4 (~30 GPU-hours/week); no paid compute (AGENTS.md).

## Cost of training from scratch (6·N·D FLOPs, ~20 tokens per parameter)

| Params | Tokens needed | Train memory (fp32 + Adam) | 4-core CPU | Kaggle T4 (best case) |
|---|---|---|---|---|
| 57M (current) | 1.1B | 0.9 GB | ~30 days | ~4 h |
| 150M | 3B | 2.4 GB | ~208 days | ~30 h = 1 Kaggle week |
| 300M | 6B | 4.8 GB | ~833 days | ~120 h = 4 Kaggle weeks |
| 1B | 20B | 16 GB (does not fit a T4 or this PC) | decades | ~1,333 h = 44 Kaggle weeks |

The current base has seen ~19M tokens (about 1/1000 of a 1B model's need). Data is a second,
independent limit: billions of reviewed Hindi/English tokens are required at every row above 57M.

## What is realistic

1. **Make the 57M base chat well first.** The limit today is conversation diversity, not size:
   bulk-v1's 9,240 rows had ~800 distinct replies and collapsed the model (CONVERSATION_TRAINING_STAGE.md).
2. **Grow to ~150M on Kaggle** with `TantraModel.grow_depth` (function-preserving layer growth)
   plus more pretraining tokens (public Wikipedia Hindi/English via PUBLIC_DATA_PIPELINE.md), then
   re-run chat SFT. About one Kaggle week of GPU time.
3. **Keep serving on CPU**: lazy experts, INT8 head (~60 tok/s at 57M; expect ~20-25 tok/s at 150M).
4. **Experts on top of a chat-tuned base**, trained only on domain data the base has not seen
   (ROADMAP_CPU_EXPERTS.md M3/M4 findings). Knowledge-heavy skills belong in retrieval
   (Tantra/memory) rather than in weights at this scale.
5. **1B** only with paid or donated GPU time and billions of curated tokens, or by starting from
   an existing open-weights model (the owner prefers own-trained weights, so this is not planned).

## Experiment: 16K-262K tiny experts on CPU (2026-10-21)

`Tantra/model/peer.py` adds a product-key expert layer (each expert = one hidden neuron; a token
activates 4 heads x 16 experts, found in O(sqrt N) with product keys; sparse gradients touch only
the chosen rows). Tests: `tests/test_peer.py` (exact top-k, no-op start, sparse gradients).
CPU training step, d_model 512, 1,024 tokens, 2 threads, measured while another run shared the CPU:

| Layer | Total params | Active params/token | Train tok/s | Extra RSS |
|---|---|---|---|---|
| Dense FFN (current block) | 1.44M | 1.44M | 7,517 | - |
| 16,384 experts | 16.9M | 0.07M | 700 | +648 MB |
| 65,536 experts | 67.3M | 0.07M | 523 | +872 MB |
| 262,144 experts | 268.7M | 0.07M | 382 | +2.3 GB |

Finding: capacity scales cheaply (16x more experts cost ~1.8x time), so a ~1B-total-parameter
model with tiny active compute is a real design for this PC. But in plain PyTorch the gather and
sparse-optimizer overhead makes it ~10x slower per token than the dense block, so it is not yet a
training speed-up. Next: fused gather kernels or merging the down/up lookups, then a small
pretraining A/B against the dense FFN at equal wall-clock time. Capacity still needs data: each
expert must see enough tokens, so this does not remove the data requirement above.
