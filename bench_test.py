import sys
import time
import torch

from Tantra.config import get_preset
from Tantra.data.expert_dataset import EncodedDataset
from Tantra.inference.inference_engine import TantraEngine
from Tantra.model.tantra_model import TantraModel
from Tantra.reasoning import Reasoner
from Tantra.training.tantra_trainer import TantraTrainer, TrainArgs, setup_cpu

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

setup_cpu()

print("=" * 60, flush=True)
print("1. INFERENCE BENCHMARK (Tokens / Sec)", flush=True)
print("=" * 60, flush=True)

for mdir in ["Model", "Model_small"]:
    print(f"\n--- Model Directory: {mdir} ---", flush=True)
    try:
        e_fp32 = TantraEngine(mdir, int8=False)
        res_fp32 = e_fp32.benchmark(n_tokens=64)
        print(f"FP32: {res_fp32['tokens_per_sec']} tok/s  (Base: {res_fp32['base_params_M']}M params, {res_fp32['experts']} experts)", flush=True)

        e_int8 = TantraEngine(mdir, int8=True)
        res_int8 = e_int8.benchmark(n_tokens=64)
        print(f"INT8: {res_int8['tokens_per_sec']} tok/s", flush=True)
    except Exception as exc:
        print(f"Error testing {mdir}: {exc}", flush=True)

print("\n" + "=" * 60, flush=True)
print("2. TRAINING SPEED BENCHMARK (Base vs Expert)", flush=True)
print("=" * 60, flush=True)

t = TantraTrainer("Model")
model = t.load_model()
tok = t.tokenizer()

# Benchmark Expert training step speed
print("\nBenchmarking Expert Training Step Speed (rank 8 LoRA)...", flush=True)
t0 = time.perf_counter()
targs = TrainArgs(steps=20, lr=3e-3, batch_size=16, max_len=128, log_every=100)
res_expert = t.train_expert("greeting", model, targs)
dt_expert = time.perf_counter() - t0
steps_per_sec_expert = 20 / dt_expert
tokens_per_sec_expert = (20 * 16 * 128) / dt_expert
print(f"Expert training (20 steps): {dt_expert:.2f}s -> {steps_per_sec_expert:.2f} steps/s (~{tokens_per_sec_expert:.0f} tokens/s)", flush=True)

# Benchmark Base training step speed
print("\nBenchmarking Base Training Step Speed (8M base model)...", flush=True)
rows = t.general_rows(limit=50)
data = EncodedDataset(tok, rows, max_len=128, assistant_only=False)
opt = torch.optim.AdamW(model.base_parameters(), lr=1e-3)
for p in model.base_parameters():
    p.requires_grad = True
model.set_active([])
model.train()
t0 = time.perf_counter()
steps = 10
for i, (ids, labels, _) in enumerate(data.batches(batch_size=8, shuffle=True)):
    if i >= steps:
        break
    loss = model.loss(ids, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
dt_base = time.perf_counter() - t0
steps_per_sec_base = steps / dt_base
tokens_per_sec_base = (steps * 8 * 128) / dt_base
print(f"Base training ({steps} steps): {dt_base:.2f}s -> {steps_per_sec_base:.2f} steps/s (~{tokens_per_sec_base:.0f} tokens/s)", flush=True)

print("\n" + "=" * 60, flush=True)
print("3. LIVE REASONING & QUERY TESTS", flush=True)
print("=" * 60, flush=True)
queries = [
    "who are you?",
    "what is 24 times 15?",
    "hello, good morning",
    "आप कौन हैं?",
]
engine = TantraEngine("Model")
reasoner = Reasoner(engine, ["Datasets/curated"], use_web=False, use_teacher=False)
for q in queries:
    msgs = [{"role": "user", "content": q}]
    ans = reasoner.answer(msgs, temperature=0.7, max_new_tokens=64)
    print(f"\nUser: {q}", flush=True)
    print(f"Route: {ans['route']} | Source: {ans['source']}", flush=True)
    print(f"Reply: {ans['reply']}", flush=True)

print("\n" + "=" * 60, flush=True)
print("All tests completed successfully.", flush=True)
print("=" * 60, flush=True)
